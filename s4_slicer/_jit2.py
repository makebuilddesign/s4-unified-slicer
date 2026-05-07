"""Second-tier numba kernels: per-segment finalisation loop.

Splits out the long Python `for ci in range(n_seg)` loop in
`gcode_transform_fast.transform_gcode` so it can be JIT-compiled.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:                                       # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        def deco(f): return f
        return deco


@njit(cache=True, fastmath=False)
def finalise_segments(
    bary_ok, contain, closest, np_pos, rot_per_pt,
    cmd_arr, ext_arr, extp_arr, invt_arr, feed_arr,
    z_squish, retr_pos, retr_neg, max_emult,
    rot_max_delta, alpha, deg45,
):
    """Returns (out_pos, out_rot, out_cmd, out_ext, out_extp,
                out_invt, out_feed, out_travel, lost).
    Output arrays are over-allocated and trimmed via returned `n_out`.
    """
    n = bary_ok.shape[0]
    cap = n * 4 + 8                                # generous over-allocate
    out_pos    = np.zeros((cap, 3), dtype=np.float64)
    out_rot    = np.zeros(cap, dtype=np.float64)
    out_cmd    = np.zeros(cap, dtype=np.int8)
    out_ext    = np.zeros(cap, dtype=np.float64)
    out_extp   = np.zeros(cap, dtype=np.bool_)
    out_invt   = np.zeros(cap, dtype=np.float64)
    out_invtp  = np.zeros(cap, dtype=np.bool_)
    out_feed   = np.zeros(cap, dtype=np.float64)
    out_travel = np.zeros(cap, dtype=np.bool_)

    n_out = 0
    have_prev = False
    prev_x = 0.0; prev_y = 0.0; prev_z = 0.0
    prev_rot   = 0.0
    prev_cmd   = np.int8(0)
    prev_travel= False
    travelling = False
    travelling_air = False
    highest    = 0.0
    lost       = 0

    for ci in range(n):
        cmd = cmd_arr[ci]
        has_ext  = extp_arr[ci]
        ext      = ext_arr[ci] if has_ext else 0.0
        invt     = invt_arr[ci]
        has_invt = invt != 0.0
        cont     = contain[ci]
        nx_ = np_pos[ci, 0]; ny_ = np_pos[ci, 1]; nz_ = np_pos[ci, 2]
        rot = rot_per_pt[ci]
        dont_smooth = False
        skip = False

        if not bary_ok[ci] or (cmd == 0 and cont == -1):
            if cmd == 1 and cont == -1 and bary_ok[ci]:
                pass
            elif cmd == 1:
                lost += 1
                skip = True
            elif cmd == 0 and (not travelling_air) and have_prev:
                nx_ = prev_x; ny_ = prev_y; nz_ = highest
                if prev_rot > deg45:    rot = deg45
                elif prev_rot < -deg45: rot = -deg45
                else:                   rot = prev_rot
                dont_smooth = True
                travelling_air = True
            elif travelling_air:
                skip = True
            else:
                skip = True
        else:
            if travelling_air:
                nz_ = highest
                if rot > deg45:    rot = deg45
                elif rot < -deg45: rot = -deg45
                dont_smooth = True
            travelling_air = False

        if skip:
            continue

        emult = 1.0
        if has_ext and ext != retr_pos and ext != retr_neg:
            cell_for_squish = cont if cont != -1 else closest[ci]
            emult *= z_squish[cell_for_squish]
            ext = ext * (emult if emult < max_emult else max_emult)
        elif has_ext and ext == retr_neg:
            travelling = True
        elif has_ext and ext == retr_pos:
            travelling = False

        if not dont_smooth:
            rot = alpha * rot + (1.0 - alpha) * prev_rot

        # Path-split if rotation changes too sharply.
        diff = rot - prev_rot
        if have_prev and (diff if diff > 0 else -diff) > rot_max_delta:
            adiff = diff if diff > 0 else -diff
            n_int = int(adiff / rot_max_delta) + 1
            dpx = nx_ - prev_x; dpy = ny_ - prev_y; dpz = nz_ - prev_z
            for kk in range(n_int):
                t = (kk + 1) / n_int
                if n_out >= cap:
                    break
                out_pos[n_out, 0] = prev_x + dpx * t
                out_pos[n_out, 1] = prev_y + dpy * t
                out_pos[n_out, 2] = prev_z + dpz * t
                out_rot[n_out]    = prev_rot + diff * t
                out_cmd[n_out]    = prev_cmd
                if has_ext:
                    out_ext[n_out]  = ext / n_int
                    out_extp[n_out] = True
                if has_invt:
                    out_invt[n_out]  = invt * n_int
                    out_invtp[n_out] = True
                out_feed[n_out]   = feed_arr[ci]
                out_travel[n_out] = prev_travel
                n_out += 1
        else:
            if n_out < cap:
                out_pos[n_out, 0] = nx_
                out_pos[n_out, 1] = ny_
                out_pos[n_out, 2] = nz_
                out_rot[n_out]    = rot
                out_cmd[n_out]    = cmd
                if has_ext:
                    out_ext[n_out]  = ext
                    out_extp[n_out] = True
                if has_invt:
                    out_invt[n_out]  = invt
                    out_invtp[n_out] = True
                out_feed[n_out]   = feed_arr[ci]
                out_travel[n_out] = travelling
                n_out += 1

        prev_rot = rot
        if n_out > 0:
            prev_x = out_pos[n_out - 1, 0]
            prev_y = out_pos[n_out - 1, 1]
            prev_z = out_pos[n_out - 1, 2]
            have_prev = True
        prev_travel = travelling
        prev_cmd    = cmd
        if cmd == 1 and has_ext and ext > 0 and (highest != 0 or nz_ < 1):
            if nz_ > highest:
                highest = nz_

    return (out_pos[:n_out],   out_rot[:n_out],
            out_cmd[:n_out],   out_ext[:n_out],
            out_extp[:n_out],  out_invt[:n_out],
            out_invtp[:n_out], out_feed[:n_out],
            out_travel[:n_out], lost)
