"""Inverse-transform a planar G-code through the deformation field.

This produces the final 4-axis (R, theta, Z, B) G-code for the polar
Core R-Theta printer used in the original S4 Slicer project.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv
from pygcode import Line


@dataclass
class GCodeParams:
    seg_size: float = 0.6                 # mm — segmentation length
    max_rotation_deg: float = 30.0
    min_rotation_deg: float = -130.0
    nozzle_offset: float = 42.0           # mm — radial offset of nozzle from B axis
    rotation_averaging_alpha: float = 0.2  # EMA smoothing of B axis
    retraction_length: float = 1.0
    rotation_max_delta_deg: float = 1.0
    max_extrusion_multiplier: float = 10.0
    output_polar: bool = True             # else cartesian X/Y/Z/B


def _tet_volume(p1, p2, p3, p4):
    mat = np.vstack([p2 - p1, p3 - p1, p4 - p1])
    return abs(np.linalg.det(mat)) / 6.0


def _barycentric(a, b, c, d, p):
    total = _tet_volume(a, b, c, d)
    if total == 0:
        return None
    va = _tet_volume(p, b, c, d)
    vb = _tet_volume(p, a, c, d)
    vc = _tet_volume(p, a, b, d)
    vd = _tet_volume(p, a, b, c)
    return np.array([va, vb, vc, vd]) / total


def _project(plane_x, plane_y, pts):
    return np.column_stack([np.sum(plane_x * pts, axis=1), np.sum(plane_y * pts, axis=1)])


def _segment_gcode(in_path: str, seg_size: float, log=print):
    pos = np.array([0.0, 0.0, 20.0])
    feed = 5000.0
    points = []
    with open(in_path, "r") as fh:
        for ln in fh.readlines():
            try:
                line = Line(ln)
            except Exception:
                continue
            if not line.block.gcodes:
                continue
            for gc in sorted(line.block.gcodes):
                if gc.word not in ("G01", "G00"):
                    continue
                prev = pos.copy()
                if gc.X is not None:
                    pos[0] = gc.X
                if gc.Y is not None:
                    pos[1] = gc.Y
                if gc.Z is not None:
                    pos[2] = gc.Z

                for w in line.block.words:
                    if w.letter == "F":
                        feed = float(w.value)

                extrusion = None
                for prm in line.block.modal_params:
                    if prm.letter == "E":
                        extrusion = float(prm.value)

                d = pos - prev
                dist = np.linalg.norm(d)
                if dist > 0:
                    nseg = int(-(-dist // seg_size))
                    sd = dist / nseg
                    t = (1 / feed) * sd
                    inv_t = 1 / t if t else None
                    for i in range(nseg):
                        points.append({
                            "position": prev + d * (i + 1) / nseg,
                            "command": gc.word,
                            "extrusion": extrusion / nseg if extrusion is not None else None,
                            "inv_time_feed": inv_t,
                            "move_length": sd,
                            "feed": feed,
                            "travelling": False,
                        })
                else:
                    t = (1 / feed) * dist
                    inv_t = 1 / t if t else None
                    points.append({
                        "position": pos.copy(),
                        "command": gc.word,
                        "extrusion": extrusion,
                        "inv_time_feed": inv_t,
                        "move_length": dist,
                        "feed": feed,
                        "travelling": False,
                    })
    log(f"[gcode] segmented into {len(points)} points")
    return points


def transform_gcode(
    input_tet: pv.UnstructuredGrid,
    deformed_tet: pv.UnstructuredGrid,
    in_gcode: str,
    out_gcode: str,
    p: GCodeParams | None = None,
    log=print,
):
    p = p or GCodeParams()
    MAX_R = np.deg2rad(p.max_rotation_deg)
    MIN_R = np.deg2rad(p.min_rotation_deg)
    ROT_MAX_DELTA = np.deg2rad(p.rotation_max_delta_deg)

    # vertex-level transformation field (deformed - undeformed)
    vt = deformed_tet.points - input_tet.points

    # tangential vector per cell
    tang = np.cross(np.array([0, 0, 1]), input_tet.cell_data["cell_center"][:, :2])
    with np.errstate(invalid="ignore"):
        tang /= np.linalg.norm(tang, axis=1)[:, None]
    tang[np.isnan(tang).any(axis=1)] = [1, 0, 0]

    n_cells = deformed_tet.number_of_cells
    n_pts = input_tet.number_of_points

    # rotation per cell using Kabsch on the radial-Z plane
    npv = np.zeros(n_pts)
    for cell in input_tet.field_data["cells"]:
        npv[cell] += 1

    vrot = np.zeros(deformed_tet.number_of_points)
    crot = np.zeros(n_cells)
    for ci, cell in enumerate(deformed_tet.field_data["cells"]):
        nv = deformed_tet.field_data["cell_vertices"][cell].copy()
        ncc = deformed_tet.cell_data["cell_center"][ci]
        ov = input_tet.field_data["cell_vertices"][cell].copy()
        occ = input_tet.cell_data["cell_center"][ci]
        nv -= ncc
        ov -= occ
        rad = occ[:2]
        rn = np.linalg.norm(rad)
        if rn == 0:
            px = np.array([1.0, 0.0, 0.0])
        else:
            px = np.array([rad[0] / rn, rad[1] / rn, 0.0])
        py = np.array([0, 0, 1.0])
        nvp = _project(px, py, nv)
        ovp = _project(px, py, ov)
        cov = nvp.T @ ovp
        U, _, Vt = np.linalg.svd(cov)
        rm = U @ Vt
        rot = -np.arccos(min(max(rm[0, 0], -1), 1))
        if rm[1, 0] < 0:
            rot = -rot
        rot = max(min(rot, MAX_R), MIN_R)
        crot[ci] = rot
        for vi in cell:
            vrot[vi] += rot / npv[vi]

    # z-squish scales (volume ratio)
    from .deform import S4Deformer
    rmats = S4Deformer.calculate_rotation_matrices(input_tet, crot)
    z_squish = np.full(n_cells, np.nan)
    for ci, cell in enumerate(deformed_tet.field_data["cells"]):
        wv = deformed_tet.field_data["cell_vertices"][cell]
        uv = input_tet.field_data["cell_vertices"][cell]
        try:
            v_def = _tet_volume(*wv)
            v_orig = _tet_volume(*uv)
            if v_def > 1e-12:
                z_squish[ci] = v_orig / v_def
            else:
                z_squish[ci] = 1.0
        except Exception:
            z_squish[ci] = 1.0
    z_squish = np.nan_to_num(z_squish, nan=1.0, posinf=p.max_extrusion_multiplier, neginf=1.0)

    # segment the gcode
    points = _segment_gcode(in_gcode, p.seg_size, log)
    positions = [pt["position"] for pt in points]
    contain = deformed_tet.find_containing_cell(positions)
    closest = deformed_tet.find_closest_cell(positions)

    new_points = []
    prev_new = None
    travelling_air = False
    travelling = False
    prev_rot = 0.0
    prev_travel = False
    prev_cmd = "G00"
    highest = 0.0
    lost = 0

    for ci, (pt, cont) in enumerate(zip(points, contain)):
        pos = pt["position"]
        cmd = pt["command"]
        ext = pt["extrusion"]
        invt = pt["inv_time_feed"]

        def interp(pos, cont, cmd, ci):
            if cmd == "G00" and cont == -1:
                return None, None
            if cmd == "G01" and cont == -1:
                cont = closest[ci]
            vi = deformed_tet.field_data["cells"][cont]
            cv = deformed_tet.field_data["cell_vertices"][vi]
            bary = _barycentric(cv[0], cv[1], cv[2], cv[3], pos)
            if bary is None or np.sum(bary) > 1.01:
                return None, None
            tr = vt[vi] * bary[:, None]
            tr = np.sum(tr, axis=0)
            np_ = pos - tr
            r = np.sum(vrot[vi] * bary)
            return np_, r

        dont_smooth = False
        np_, rot = interp(pos, cont, cmd, ci)
        if np_ is None:
            if cmd == "G01":
                lost += 1
                continue
            if cmd == "G00" and not travelling_air and prev_new is not None:
                np_ = np.array([prev_new[0], prev_new[1], highest])
                rot = max(min(prev_rot, np.deg2rad(45)), np.deg2rad(-45))
                dont_smooth = True
                travelling_air = True
            elif travelling_air:
                continue
            else:
                continue
        else:
            if travelling_air:
                np_[2] = highest
                rot = max(min(rot, np.deg2rad(45)), np.deg2rad(-45))
                dont_smooth = True
            travelling_air = False

        emult = 1.0
        if ext is not None and ext != p.retraction_length and ext != -p.retraction_length:
            emult *= z_squish[cont if cont != -1 else closest[ci]]
            ext = ext * min(emult, p.max_extrusion_multiplier)
        elif ext == -p.retraction_length:
            travelling = True
        elif ext == p.retraction_length:
            travelling = False

        if not dont_smooth:
            rot = p.rotation_averaging_alpha * rot + (1 - p.rotation_averaging_alpha) * prev_rot

        if prev_new is not None and abs(rot - prev_rot) > ROT_MAX_DELTA:
            dr = rot - prev_rot
            n_int = int(abs(dr) / ROT_MAX_DELTA) + 1
            dp = np_ - prev_new
            for i in range(n_int):
                new_points.append({
                    "position": prev_new + dp * ((i + 1) / n_int),
                    "rotation": prev_rot + dr * ((i + 1) / n_int),
                    "command": prev_cmd,
                    "extrusion": ext / n_int if ext is not None else None,
                    "inv_time_feed": invt * n_int if invt is not None else None,
                    "feed": pt["feed"],
                    "travelling": prev_travel,
                })
        else:
            new_points.append({
                "position": np_, "rotation": rot, "command": cmd,
                "extrusion": ext, "inv_time_feed": invt,
                "feed": pt["feed"], "travelling": travelling,
            })

        prev_rot = rot
        prev_new = np_.copy()
        prev_travel = travelling
        prev_cmd = cmd
        if cmd == "G01" and ext is not None and ext > 0 and (highest != 0 or np_[2] < 1):
            highest = max(highest, np_[2])

    log(f"[gcode] transformed; lost {lost} extrusion vertices")

    # write final gcode
    prev_theta = 0.0
    theta_acc = 0.0
    n_lines = 0
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
        for pt in new_points:
            pos = pt["position"]
            rot = pt["rotation"]
            if np.all(np.isnan(pos)) or pos[2] < 0:
                continue
            zhop = 1 if pt["travelling"] else 0
            r = np.linalg.norm(pos[:2])
            theta = np.arctan2(pos[1], pos[0])
            z = pos[2]
            r += -np.sin(rot) * (p.nozzle_offset + zhop)
            z += (np.cos(rot) - 1) * (p.nozzle_offset + zhop) + zhop
            dt = theta - prev_theta
            if dt > np.pi:
                dt -= 2 * np.pi
            if dt < -np.pi:
                dt += 2 * np.pi
            theta_acc += dt
            if p.output_polar:
                s = (f"{pt['command']} C{np.rad2deg(theta_acc):.5f} "
                     f"X{r:.5f} Z{z:.5f} B{np.rad2deg(rot):.5f}")
            else:
                s = (f"{pt['command']} X{pos[0]:.5f} Y{pos[1]:.5f} "
                     f"Z{pos[2]:.5f} B{np.rad2deg(rot):.5f}")
            if pt["extrusion"] is not None:
                s += f" E{pt['extrusion']:.4f}"
            no_feed = False
            if pt["inv_time_feed"] is not None:
                s += f" F{pt['inv_time_feed']:.4f}"
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
    return out_gcode
