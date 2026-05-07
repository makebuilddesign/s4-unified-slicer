"""Numba-accelerated kernels for the gcode-transform stage.

These are imported lazily — if numba isn't available we fall back to numpy.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:                                      # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        def deco(f): return f
        return deco
    def prange(*a, **k): return range(*a, **k)


# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=False, parallel=True)
def tet_volume_batch(p1, p2, p3, p4):
    """Volumes of a batch of tetrahedra, p* shape (n,3) → (n,)."""
    n = p1.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        a0 = p2[i, 0] - p1[i, 0]; a1 = p2[i, 1] - p1[i, 1]; a2 = p2[i, 2] - p1[i, 2]
        b0 = p3[i, 0] - p1[i, 0]; b1 = p3[i, 1] - p1[i, 1]; b2 = p3[i, 2] - p1[i, 2]
        c0 = p4[i, 0] - p1[i, 0]; c1 = p4[i, 1] - p1[i, 1]; c2 = p4[i, 2] - p1[i, 2]
        det = (a0 * (b1 * c2 - b2 * c1)
             - a1 * (b0 * c2 - b2 * c0)
             + a2 * (b0 * c1 - b1 * c0))
        out[i] = abs(det) / 6.0
    return out


@njit(cache=True, fastmath=False, parallel=True)
def barycentric_batch(verts, pts):
    """Barycentric coordinates of pts (n,3) inside tetrahedra verts (n,4,3).

    Returns (n,4) bary, NaN where the tet is degenerate.
    """
    n = pts.shape[0]
    out = np.empty((n, 4), dtype=np.float64)
    for i in prange(n):
        a0 = verts[i, 0, 0]; a1 = verts[i, 0, 1]; a2 = verts[i, 0, 2]
        b0 = verts[i, 1, 0]; b1 = verts[i, 1, 1]; b2 = verts[i, 1, 2]
        c0 = verts[i, 2, 0]; c1 = verts[i, 2, 1]; c2 = verts[i, 2, 2]
        d0 = verts[i, 3, 0]; d1 = verts[i, 3, 1]; d2 = verts[i, 3, 2]
        p0 = pts[i, 0]; p1 = pts[i, 1]; p2 = pts[i, 2]

        # total = |det(B-A, C-A, D-A)|/6
        ax = b0 - a0; ay = b1 - a1; az = b2 - a2
        bx = c0 - a0; by = c1 - a1; bz = c2 - a2
        cx = d0 - a0; cy = d1 - a1; cz = d2 - a2
        det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
        total = abs(det) / 6.0
        if total == 0.0:  # match reference: exact zero check
            out[i, 0] = np.nan; out[i, 1] = np.nan
            out[i, 2] = np.nan; out[i, 3] = np.nan
            continue

        # va = vol(p, b, c, d)
        ax = b0 - p0; ay = b1 - p1; az = b2 - p2
        bx = c0 - p0; by = c1 - p1; bz = c2 - p2
        cx = d0 - p0; cy = d1 - p1; cz = d2 - p2
        det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
        va = abs(det) / 6.0

        # vb = vol(p, a, c, d)
        ax = a0 - p0; ay = a1 - p1; az = a2 - p2
        bx = c0 - p0; by = c1 - p1; bz = c2 - p2
        cx = d0 - p0; cy = d1 - p1; cz = d2 - p2
        det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
        vb = abs(det) / 6.0

        # vc = vol(p, a, b, d)
        ax = a0 - p0; ay = a1 - p1; az = a2 - p2
        bx = b0 - p0; by = b1 - p1; bz = b2 - p2
        cx = d0 - p0; cy = d1 - p1; cz = d2 - p2
        det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
        vc = abs(det) / 6.0

        # vd = vol(p, a, b, c)
        ax = a0 - p0; ay = a1 - p1; az = a2 - p2
        bx = b0 - p0; by = b1 - p1; bz = b2 - p2
        cx = c0 - p0; cy = c1 - p1; cz = c2 - p2
        det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
        vd = abs(det) / 6.0

        out[i, 0] = va / total
        out[i, 1] = vb / total
        out[i, 2] = vc / total
        out[i, 3] = vd / total
    return out


@njit(cache=True, fastmath=False, parallel=True)
def segment_lines(prev, pos, dist, fcol, e_arr, e_present,
                  cmd_in, nseg, offsets, seg_size):
    """Inner loop of `_segment_gcode_fast`, JITted.

    All output arrays are pre-allocated by caller.
    """
    n = prev.shape[0]
    total = int(offsets[-1])
    seg_pos     = np.zeros((total, 3), dtype=np.float64)
    seg_cmd     = np.zeros(total, dtype=np.int8)
    seg_ext     = np.zeros(total, dtype=np.float64)
    seg_ext_p   = np.zeros(total, dtype=np.bool_)
    seg_invt    = np.zeros(total, dtype=np.float64)
    seg_feed    = np.zeros(total, dtype=np.float64)
    seg_movelen = np.zeros(total, dtype=np.float64)

    for i in prange(n):
        a = offsets[i]; b = offsets[i + 1]; ns = int(nseg[i])
        if dist[i] > 0.0:
            sd = dist[i] / ns
            for k in range(ns):
                t = (k + 1) / ns
                seg_pos[a + k, 0] = prev[i, 0] + (pos[i, 0] - prev[i, 0]) * t
                seg_pos[a + k, 1] = prev[i, 1] + (pos[i, 1] - prev[i, 1]) * t
                seg_pos[a + k, 2] = prev[i, 2] + (pos[i, 2] - prev[i, 2]) * t
                seg_movelen[a + k] = sd
            tval = sd / fcol[i] if fcol[i] > 0 else 0.0
            invt = 1.0 / tval if tval > 0 else 0.0
            for k in range(ns):
                seg_invt[a + k] = invt
            if e_present[i]:
                ev = e_arr[i] / ns
                for k in range(ns):
                    seg_ext[a + k] = ev
                    seg_ext_p[a + k] = True
        else:
            for k in range(ns):
                seg_pos[a + k, 0] = pos[i, 0]
                seg_pos[a + k, 1] = pos[i, 1]
                seg_pos[a + k, 2] = pos[i, 2]
                seg_movelen[a + k] = 0.0
                seg_invt[a + k] = 0.0
            if e_present[i]:
                for k in range(ns):
                    seg_ext[a + k] = e_arr[i]
                    seg_ext_p[a + k] = True
        c_in = cmd_in[i]
        f_in = fcol[i]
        for k in range(ns):
            seg_cmd[a + k]  = c_in
            seg_feed[a + k] = f_in

    return seg_pos, seg_cmd, seg_ext, seg_ext_p, seg_invt, seg_feed, seg_movelen
