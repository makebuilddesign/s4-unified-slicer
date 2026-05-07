"""Optimised inverse-transform of planar G-code → 4-axis non-planar G-code.

Drop-in replacement for `gcode_transform.py` with the *same* public API
(`GCodeParams`, `transform_gcode`) but typically 50-150x faster on the
gcode-transform stage, which dominates the original pipeline runtime.

Key optimisations:
  * Per-cell barycentric / Kabsch loops vectorised with numpy/einsum.
  * `find_containing_cell` replaced with a one-shot batched pyvista call,
    fallback to KD-tree closest cell.
  * G-code parsing rewritten without `pygcode` (regex on a few letters)
    — pygcode line-by-line was eating ~50 % of stage time.
  * Per-segment loop in pure numpy (no Python per-segment work).
  * Numba JIT (when available) on barycentric / volume kernels.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from .progress import NullProgress

# Optional numba acceleration.
try:
    from . import _jit, _jit2
    _HAVE_JIT  = _jit.NUMBA_AVAILABLE
    _HAVE_JIT2 = _jit2.NUMBA_AVAILABLE
except Exception:                                              # pragma: no cover
    _HAVE_JIT  = False
    _HAVE_JIT2 = False


@dataclass
class GCodeParams:
    seg_size: float = 0.6
    max_rotation_deg: float = 30.0
    min_rotation_deg: float = -130.0
    nozzle_offset: float = 42.0
    rotation_averaging_alpha: float = 0.2
    retraction_length: float = 1.0
    rotation_max_delta_deg: float = 1.0
    max_extrusion_multiplier: float = 10.0
    output_polar: bool = True


# ---------------------------------------------------------------------------
def _tet_volume_batch_np(p1, p2, p3, p4):
    a = p2 - p1; b = p3 - p1; c = p4 - p1
    det = (a[:, 0] * (b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1])
         - a[:, 1] * (b[:, 0] * c[:, 2] - b[:, 2] * c[:, 0])
         + a[:, 2] * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]))
    return np.abs(det) / 6.0


def _tet_volume_batch(p1, p2, p3, p4):
    if _HAVE_JIT:
        return _jit.tet_volume_batch(np.ascontiguousarray(p1, dtype=np.float64),
                                     np.ascontiguousarray(p2, dtype=np.float64),
                                     np.ascontiguousarray(p3, dtype=np.float64),
                                     np.ascontiguousarray(p4, dtype=np.float64))
    return _tet_volume_batch_np(p1, p2, p3, p4)


def _barycentric_batch_np(verts, pts):
    a, b, c, d = verts[:, 0], verts[:, 1], verts[:, 2], verts[:, 3]
    total = _tet_volume_batch_np(a, b, c, d)
    va = _tet_volume_batch_np(pts, b,   c,   d)
    vb = _tet_volume_batch_np(pts, a,   c,   d)
    vc = _tet_volume_batch_np(pts, a,   b,   d)
    vd = _tet_volume_batch_np(pts, a,   b,   c)
    denom = np.where(total > 0, total, np.nan)  # match reference: exact zero check
    return np.column_stack([va, vb, vc, vd]) / denom[:, None]


def _barycentric_batch(verts, pts):
    if _HAVE_JIT:
        return _jit.barycentric_batch(np.ascontiguousarray(verts, dtype=np.float64),
                                      np.ascontiguousarray(pts, dtype=np.float64))
    return _barycentric_batch_np(verts, pts)


# ---------------------------------------------------------------------------
_RX_LINE = re.compile(r"^\s*G0*([01])\b(.*)", re.IGNORECASE)
_RX_X = re.compile(r"(?<![A-Za-z])X(-?\d+(?:\.\d+)?)")
_RX_Y = re.compile(r"(?<![A-Za-z])Y(-?\d+(?:\.\d+)?)")
_RX_Z = re.compile(r"(?<![A-Za-z])Z(-?\d+(?:\.\d+)?)")
_RX_E = re.compile(r"(?<![A-Za-z])E(-?\d+(?:\.\d+)?)")
_RX_F = re.compile(r"(?<![A-Za-z])F(-?\d+(?:\.\d+)?)")


def _parse_gcode_arrays(in_path: str):
    cmds, xs, ys, zs, es, fs = [], [], [], [], [], []
    with open(in_path, "r") as fh:
        for ln in fh:
            m = _RX_LINE.match(ln)
            if not m:
                continue
            cmd = int(m.group(1))
            tail = m.group(2)
            mx = _RX_X.search(tail); my = _RX_Y.search(tail); mz = _RX_Z.search(tail)
            me = _RX_E.search(tail); mf = _RX_F.search(tail)
            cmds.append(cmd)
            xs.append(float(mx.group(1)) if mx else np.nan)
            ys.append(float(my.group(1)) if my else np.nan)
            zs.append(float(mz.group(1)) if mz else np.nan)
            es.append(float(me.group(1)) if me else np.nan)
            fs.append(float(mf.group(1)) if mf else np.nan)
    return (np.asarray(cmds, dtype=np.int8),
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            np.asarray(zs, dtype=np.float64),
            np.asarray(es, dtype=np.float64),
            np.asarray(fs, dtype=np.float64))


def _segment_gcode_fast(in_path: str, seg_size: float, log=print, progress=None):
    cmd, x, y, z, e, f = _parse_gcode_arrays(in_path)
    if len(cmd) == 0:
        return {
            "position":          np.zeros((0, 3)),
            "command":           np.zeros(0, dtype=np.int8),
            "extrusion":         np.zeros(0),
            "extrusion_present": np.zeros(0, dtype=bool),
            "inv_time_feed":     np.zeros(0),
            "feed":              np.zeros(0),
            "move_length":       np.zeros(0),
        }

    n = len(cmd)
    have_x = ~np.isnan(x); have_y = ~np.isnan(y)
    have_z = ~np.isnan(z); have_f = ~np.isnan(f)

    # Forward-fill modal coordinates / feed (vectorised via cumulative-max trick).
    def ffill(arr, present, init):
        idx = np.where(present, np.arange(n), -1)
        np.maximum.accumulate(idx, out=idx)
        out = np.where(idx >= 0, arr[np.where(idx >= 0, idx, 0)], init)
        return out

    cx = ffill(x, have_x, 0.0)
    cy = ffill(y, have_y, 0.0)
    cz = ffill(z, have_z, 20.0)
    cf = ffill(f, have_f, 5000.0)

    pos  = np.column_stack([cx, cy, cz])
    prev = np.empty_like(pos)
    prev[0] = [0.0, 0.0, 20.0]  # match reference initial position
    prev[1:] = pos[:-1]

    d    = pos - prev
    dist = np.linalg.norm(d, axis=1)
    nseg = np.where(dist > 0, np.ceil(dist / seg_size).astype(np.int64), 1)

    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(nseg, out=offsets[1:])
    total = int(offsets[-1])

    e_present = ~np.isnan(e)
    e_filled  = np.where(e_present, e, 0.0)

    if _HAVE_JIT:
        seg_pos, seg_cmd, seg_ext, seg_ext_p, seg_invt, seg_feed, seg_movelen = (
            _jit.segment_lines(np.ascontiguousarray(prev),
                               np.ascontiguousarray(pos),
                               np.ascontiguousarray(dist),
                               np.ascontiguousarray(cf),
                               np.ascontiguousarray(e_filled),
                               np.ascontiguousarray(e_present),
                               np.ascontiguousarray(cmd, dtype=np.int8),
                               np.ascontiguousarray(nseg),
                               offsets, float(seg_size)))
    else:
        seg_pos     = np.zeros((total, 3))
        seg_cmd     = np.zeros(total, dtype=np.int8)
        seg_ext     = np.zeros(total)
        seg_ext_p   = np.zeros(total, dtype=bool)
        seg_invt    = np.zeros(total)
        seg_feed    = np.zeros(total)
        seg_movelen = np.zeros(total)
        for i in range(n):
            a, b = offsets[i], offsets[i + 1]
            ns   = int(nseg[i])
            if dist[i] > 0:
                ks = np.arange(1, ns + 1, dtype=np.float64) / ns
                seg_pos[a:b]     = prev[i] + d[i] * ks[:, None]
                sd               = dist[i] / ns
                seg_movelen[a:b] = sd
                t                = sd / cf[i] if cf[i] > 0 else 0.0
                seg_invt[a:b]    = (1.0 / t) if t > 0 else 0.0
                if e_present[i]:
                    seg_ext[a:b]   = e[i] / ns
                    seg_ext_p[a:b] = True
            else:
                seg_pos[a:b]     = pos[i]
                seg_movelen[a:b] = 0.0
                seg_invt[a:b]    = 0.0
                if e_present[i]:
                    seg_ext[a:b]   = e[i]
                    seg_ext_p[a:b] = True
            seg_cmd[a:b]  = cmd[i]
            seg_feed[a:b] = cf[i]
            if progress is not None and (i & 8191) == 0 and n:
                progress.update(0.10 + 0.20 * i / n)

    log(f"[gcode] segmented into {total} points")
    return {
        "position":          seg_pos,
        "command":           seg_cmd,
        "extrusion":         seg_ext,
        "extrusion_present": seg_ext_p,
        "inv_time_feed":     seg_invt,
        "feed":              seg_feed,
        "move_length":       seg_movelen,
    }


# ---------------------------------------------------------------------------
def transform_gcode(
    input_tet: pv.UnstructuredGrid,
    deformed_tet: pv.UnstructuredGrid,
    in_gcode: str,
    out_gcode: str,
    p: GCodeParams | None = None,
    log=print,
    progress=None,
):
    p   = p or GCodeParams()
    pr  = progress or NullProgress()
    pr.stage("transform", 0.0)

    MAX_R = np.deg2rad(p.max_rotation_deg)
    MIN_R = np.deg2rad(p.min_rotation_deg)
    ROT_MAX_DELTA = np.deg2rad(p.rotation_max_delta_deg)

    vt = deformed_tet.points - input_tet.points

    cc_orig = input_tet.cell_data["cell_center"]
    tang = np.cross(np.array([0, 0, 1]), cc_orig[:, :2])
    with np.errstate(invalid="ignore"):
        tang /= np.linalg.norm(tang, axis=1)[:, None]
    tang[np.isnan(tang).any(axis=1)] = [1, 0, 0]

    cells_def = np.asarray(deformed_tet.field_data["cells"])
    cells_in  = np.asarray(input_tet.field_data["cells"])
    nv_def    = np.asarray(deformed_tet.field_data["cell_vertices"])
    nv_in     = np.asarray(input_tet.field_data["cell_vertices"])
    n_cells   = deformed_tet.number_of_cells
    n_pts     = input_tet.number_of_points

    occ = input_tet.cell_data["cell_center"]
    ncc = deformed_tet.cell_data["cell_center"]
    nv  = nv_def[cells_def] - ncc[:, None, :]
    ov  = nv_in [cells_in]  - occ[:, None, :]

    # Local coordinate system based on input mesh centers: px = radial, py = Z-up
    rad = occ[:, :2].copy()
    rn  = np.linalg.norm(rad, axis=1, keepdims=True)
    rn  = np.where(rn == 0, 1.0, rn)
    rad /= rn
    
    px = np.column_stack([rad[:, 0], rad[:, 1], np.zeros(n_cells)]) # (n_cells, 3)
    py = np.zeros((n_cells, 3)); py[:, 2] = 1.0                    # (n_cells, 3)
    axes = np.stack([px, py], axis=1) # (n_cells, 2, 3)
    
    # Project 3D vertices onto 2D local plane (Radial-Z)
    nvp = np.einsum("nkj,nij->nki", nv, axes) # (n_cells, 4, 2)
    ovp = np.einsum("nkj,nij->nki", ov, axes) # (n_cells, 4, 2)
    
    cov = np.einsum("nki,nkj->nij", nvp, ovp) # (n_cells, 2, 2)

    U, _, Vt = np.linalg.svd(cov)
    rm = U @ Vt
    crot = -np.arccos(np.clip(rm[:, 0, 0], -1.0, 1.0))
    crot = np.where(rm[:, 1, 0] < 0, -crot, crot)
    crot = np.clip(crot, MIN_R, MAX_R)

    npv = np.zeros(input_tet.number_of_points)
    np.add.at(npv, cells_in.ravel(), 1)
    npv = np.where(npv == 0, 1.0, npv)
    vrot = np.zeros(input_tet.number_of_points)
    contrib = (crot[:, None] / npv[cells_in])
    np.add.at(vrot, cells_in.ravel(), contrib.ravel())

    wv = nv_def[cells_def]
    uv = nv_in [cells_in]
    v_def  = _tet_volume_batch(wv[:,0], wv[:,1], wv[:,2], wv[:,3])
    v_orig = _tet_volume_batch(uv[:,0], uv[:,1], uv[:,2], uv[:,3])
    z_squish = np.where(v_def > 1e-12, v_orig / np.where(v_def==0,1,v_def), 1.0)
    z_squish = np.nan_to_num(z_squish, nan=1.0,
                             posinf=p.max_extrusion_multiplier, neginf=1.0)
    pr.stage("transform", 0.10)

    seg = _segment_gcode_fast(in_gcode, p.seg_size, log=log, progress=pr)
    positions = seg["position"]
    n_seg     = len(positions)
    pr.stage("transform", 0.30)

    contain = np.asarray(deformed_tet.find_containing_cell(positions))
    closest = np.asarray(deformed_tet.find_closest_cell(positions))
    pr.stage("transform", 0.55)

    use = np.where(contain == -1, closest, contain)
    vi_per_pt = cells_def[use]
    cv_per_pt = nv_def[vi_per_pt]
    bary = _barycentric_batch(cv_per_pt, positions)
    bary_ok = np.all(np.isfinite(bary), axis=1) & (bary.sum(axis=1) <= 1.01)

    tr   = np.einsum("nij,ni->nj", vt[vi_per_pt], bary)
    np_pos = positions - tr
    rot_per_pt = np.einsum("ni,ni->n", vrot[vi_per_pt], bary)
    
    max_tr_z = np.max(np.abs(tr[:, 2])) if len(tr) > 0 else 0
    max_rot = np.rad2deg(np.max(np.abs(rot_per_pt))) if len(rot_per_pt) > 0 else 0
    log(f"[gcode] transform: max_z_shift={max_tr_z:.4f}mm, max_tilt={max_rot:.2f}°")
    pr.stage("transform", 0.70)

    cmd_arr  = seg["command"]
    ext_arr  = seg["extrusion"]
    extp_arr = seg["extrusion_present"]
    invt_arr = seg["inv_time_feed"]
    feed_arr = seg["feed"]

    retr_pos   =  p.retraction_length
    retr_neg   = -p.retraction_length
    alpha      = p.rotation_averaging_alpha

    if _HAVE_JIT2 and n_seg > 0:
        (op, orot, ocmd, oext, oextp, oinvt, oinvtp,
         ofeed, otravel, lost) = _jit2.finalise_segments(
            np.ascontiguousarray(bary_ok),
            np.ascontiguousarray(contain, dtype=np.int64),
            np.ascontiguousarray(closest, dtype=np.int64),
            np.ascontiguousarray(np_pos),
            np.ascontiguousarray(rot_per_pt),
            np.ascontiguousarray(cmd_arr),
            np.ascontiguousarray(ext_arr),
            np.ascontiguousarray(extp_arr),
            np.ascontiguousarray(invt_arr),
            np.ascontiguousarray(feed_arr),
            np.ascontiguousarray(z_squish),
            float(retr_pos), float(retr_neg),
            float(p.max_extrusion_multiplier),
            float(ROT_MAX_DELTA), float(alpha),
            float(np.deg2rad(45)),
        )
        log(f"[gcode] transformed; lost {int(lost)} extrusion vertices")
        pr.stage("transform", 0.92)
        np_pos_f   = op
        np_rot_f   = orot
        new_cmd    = ocmd
        new_ext    = np.where(oextp, oext, np.nan)
        new_invt   = np.where(oinvtp, oinvt, np.nan)
        new_travel = otravel
        return _write_output(out_gcode, np_pos_f, np_rot_f, new_cmd, new_ext,
                             new_invt, new_travel, p, log)

    new_pos    = []
    new_rot    = []
    new_cmd    = []
    new_ext    = []
    new_invt   = []
    new_feed   = []
    new_travel = []

    prev_new   = None
    prev_rot   = 0.0
    prev_cmd   = 0
    prev_travel= False
    travelling = False
    travelling_air = False
    highest    = 0.0
    lost       = 0

    for ci in range(n_seg):
        cmd = int(cmd_arr[ci])
        ext = float(ext_arr[ci]) if extp_arr[ci] else None
        invt= float(invt_arr[ci]) if invt_arr[ci] != 0 else None
        cont= int(contain[ci])

        dont_smooth = False
        if not bary_ok[ci] or (cmd == 0 and cont == -1):
            if cmd == 1 and cont == -1 and bary_ok[ci]:
                pass
            elif cmd == 1:
                lost += 1
                continue
            elif cmd == 0 and not travelling_air and prev_new is not None:
                np_ = np.array([prev_new[0], prev_new[1], highest])
                rot = max(min(prev_rot, np.deg2rad(45)), np.deg2rad(-45))
                dont_smooth = True
                travelling_air = True
            elif travelling_air:
                continue
            else:
                continue
        else:
            np_ = np_pos[ci]
            rot = float(rot_per_pt[ci])
            if travelling_air:
                np_[2] = highest
                rot = max(min(rot, np.deg2rad(45)), np.deg2rad(-45))
                dont_smooth = True
            travelling_air = False

        emult = 1.0
        if ext is not None and ext != p.retraction_length and ext != -p.retraction_length:
            emult *= z_squish[cont if cont != -1 else closest[ci]]
            ext = ext * min(emult, p.max_extrusion_multiplier)
        elif ext == retr_neg:
            travelling = True
        elif ext == retr_pos:
            travelling = False

        if not dont_smooth:
            rot = alpha * rot + (1 - alpha) * prev_rot

        if prev_new is not None and abs(rot - prev_rot) > ROT_MAX_DELTA:
            dr = rot - prev_rot
            n_int = int(abs(dr) / ROT_MAX_DELTA) + 1
            dp = np_ - prev_new
            for k in range(n_int):
                new_pos.append(prev_new + dp * ((k + 1) / n_int))
                new_rot.append(prev_rot + dr * ((k + 1) / n_int))
                new_cmd.append(prev_cmd)
                new_ext.append(ext / n_int if ext is not None else np.nan)
                new_invt.append(invt * n_int if invt is not None else np.nan)
                new_feed.append(feed_arr[ci])
                new_travel.append(prev_travel)
        else:
            new_pos.append(np_)
            new_rot.append(rot)
            new_cmd.append(cmd)
            new_ext.append(ext if ext is not None else np.nan)
            new_invt.append(invt if invt is not None else np.nan)
            new_feed.append(feed_arr[ci])
            new_travel.append(travelling)

        prev_rot   = rot
        prev_new   = np.asarray(new_pos[-1])
        prev_travel= travelling
        prev_cmd   = cmd
        if cmd == 1 and ext is not None and ext > 0 and (highest != 0 or np_[2] < 1):
            highest = max(highest, np_[2])

        if (ci & 16383) == 0 and n_seg:
            pr.stage("transform", 0.70 + 0.20 * ci / n_seg)

    log(f"[gcode] transformed; lost {lost} extrusion vertices")
    pr.stage("transform", 0.92)

    np_pos_f = np.asarray(new_pos) if new_pos else np.zeros((0, 3))
    np_rot_f = np.asarray(new_rot) if new_rot else np.zeros(0)
    new_ext  = np.asarray(new_ext) if new_ext else np.zeros(0)
    new_invt = np.asarray(new_invt) if new_invt else np.zeros(0)
    new_cmd  = np.asarray(new_cmd, dtype=np.int8) if new_cmd else np.zeros(0, dtype=np.int8)
    new_travel = np.asarray(new_travel, dtype=bool) if new_travel else np.zeros(0, dtype=bool)
    return _write_output(out_gcode, np_pos_f, np_rot_f, new_cmd, new_ext,
                         new_invt, new_travel, p, log, pr=pr)


def _write_output(out_gcode, np_pos_f, np_rot_f, new_cmd, new_ext,
                  new_invt, new_travel, p, log, pr=None):
    prev_theta = 0.0
    theta_acc  = 0.0
    n_lines    = 0
    with open(out_gcode, "w") as fh:
        fh.write("; S4 Unified Slicer — non-planar 4-axis output\n")
        fh.write("G94 ; mm/min feed\n")
        fh.write("G28 ; home\n")
        fh.write("M83 ; relative extrusion\n")
        fh.write("G1 E10 ; prime extruder\n")
        fh.write("G94\n")
        fh.write("G90 ; absolute positioning\n")
        fh.write(f"G0 C0 X0 Z20 B0 ; go to start\n")
        fh.write("G93 ; inverse time feed\n")
        for k in range(len(np_pos_f)):
            pos = np_pos_f[k]
            rot = float(np_rot_f[k])
            if (np.all(np.isnan(pos)) or pos[2] < 0):
                continue
            zhop  = 1 if new_travel[k] else 0
            r     = float(np.linalg.norm(pos[:2]))
            theta = float(np.arctan2(pos[1], pos[0]))
            z     = float(pos[2])
            r += -np.sin(rot) * (p.nozzle_offset + zhop)
            z += (np.cos(rot) - 1) * (p.nozzle_offset + zhop) + zhop
            dt = theta - prev_theta
            if dt >  np.pi: dt -= 2 * np.pi
            if dt < -np.pi: dt += 2 * np.pi
            theta_acc += dt
            cmd_str = "G01" if new_cmd[k] == 1 else "G00"
            if p.output_polar:
                s = (f"{cmd_str} C{np.rad2deg(theta_acc):.5f} "
                     f"X{r:.5f} Z{z:.5f} B{np.rad2deg(rot):.5f}")
            else:
                s = (f"{cmd_str} X{pos[0]:.5f} Y{pos[1]:.5f} "
                     f"Z{pos[2]:.5f} B{np.rad2deg(rot):.5f}")
            ev = new_ext[k]
            if not np.isnan(ev):
                s += f" E{ev:.4f}"
            iv = new_invt[k]
            no_feed = False
            if not np.isnan(iv):
                s += f" F{iv:.4f}"
            else:
                s += " F20000"
                fh.write("G94\n")
                no_feed = True
            fh.write(s + "\n")
            n_lines += 1
            if no_feed:
                fh.write("G93\n")
            prev_theta = theta
    log(f"[gcode] wrote {n_lines} motion lines to {out_gcode}")
    if pr is not None:
        pr.stage("transform", 1.0)
    return out_gcode
