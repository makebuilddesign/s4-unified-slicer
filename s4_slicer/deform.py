"""Mesh deformation pipeline.

Re-implements the tetrahedral-mesh rotation field optimization from the
original S4_Slicer notebook (`main.ipynb`) as a clean, importable module.
"""
from __future__ import annotations

import base64
import pickle
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pyvista as pv
import tetgen
import trimesh
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as R

UP = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _encode(obj):
    return base64.b64encode(pickle.dumps(obj)).decode("utf-8")


def _decode(s):
    return pickle.loads(base64.b64decode(s))


def _plane_fit(points: np.ndarray):
    points = np.reshape(points, (np.shape(points)[0], -1))
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)
    return ctr, np.linalg.svd(M)[0][:, -1]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DeformParams:
    # Rotation-field optimisation
    neighbour_loss_weight: float = 20.0
    max_overhang_deg: float = 30.0
    rotation_multiplier: float = 2.0
    set_initial_rotation_to_zero: bool = False
    initial_rotation_field_smoothing: int = 30
    max_pos_rotation_deg: float = 3600.0
    max_neg_rotation_deg: float = -3600.0
    rot_iterations: int = 100
    steep_overhang_compensation: bool = True
    # Mesh deformation
    deform_iterations: int = 1000
    # How many full deform passes (each builds on the last)
    num_passes: int = 1
    # Pre-processing
    rotate_x_deg: float = 0.0
    rotate_y_deg: float = 0.0
    rotate_z_deg: float = 0.0
    scale: float = 1.0
    part_offset: tuple = (0.0, 0.0, 0.0)
    make_manifold: bool = False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class S4Deformer:
    """Tetrahedralise an STL mesh, build the rotation field, and deform it."""

    def __init__(self, params: DeformParams | None = None, log=print):
        self.p = params or DeformParams()
        self.log = log
        self.cell_neighbour_dict = None
        self.cell_neighbour_graph = None
        self.bottom_cells = None
        self.bottom_cells_mask = None

    # ------------------------------------------------------------------
    # Mesh I/O
    # ------------------------------------------------------------------
    def load_stl(self, stl_path: str) -> pv.UnstructuredGrid:
        self.log(f"[load] reading STL: {stl_path}")
        mesh = trimesh.load(stl_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Could not load mesh from {stl_path}")
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris = np.asarray(mesh.faces, dtype=np.int64)
        self.log(f"[load] vertices={len(verts)} faces={len(tris)}")
        tg = tetgen.TetGen(verts, tris)
        if self.p.make_manifold:
            try:
                tg.make_manifold()
            except Exception as e:
                self.log(f"[load] make_manifold failed: {e}")
        tg.tetrahedralize()
        tet = tg.grid
        self.log(f"[load] tet cells={tet.number_of_cells} pts={tet.number_of_points}")

        # transforms
        if self.p.rotate_x_deg:
            tet = tet.rotate_x(self.p.rotate_x_deg)
        if self.p.rotate_y_deg:
            tet = tet.rotate_y(self.p.rotate_y_deg)
        if self.p.rotate_z_deg:
            tet = tet.rotate_z(self.p.rotate_z_deg)
        if self.p.scale != 1.0:
            tet = tet.scale(self.p.scale)

        # center bottom of bbox at origin (+ user offset)
        x_min, x_max, y_min, y_max, z_min, _ = tet.bounds
        offset = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]) + np.array(self.p.part_offset)
        tet.points -= offset

        # neighbours + graph
        self._build_neighbours(tet)
        return tet

    def _build_neighbours(self, tet):
        n_cells = tet.number_of_cells
        self.cell_neighbour_dict = {nt: {f: [] for f in range(n_cells)} for nt in ["point", "edge", "face"]}
        for nt in ["point", "edge", "face"]:
            edges = []
            for c in range(n_cells):
                for nb in tet.cell_neighbors(c, f"{nt}s"):
                    if nb > c:
                        edges.append((c, nb))
            for a, b in np.array(edges) if edges else []:
                self.cell_neighbour_dict[nt][a].append(b)
                self.cell_neighbour_dict[nt][b].append(a)
            tet.field_data[f"cell_{nt}_neighbours"] = np.array(edges) if edges else np.zeros((0, 2), int)

        g = nx.Graph()
        centers = tet.cell_centers().points
        for a, b in tet.field_data["cell_point_neighbours"]:
            g.add_weighted_edges_from([(int(a), int(b), float(np.linalg.norm(centers[a] - centers[b])))])
        self.cell_neighbour_graph = g

    # ------------------------------------------------------------------
    # Tet attributes
    # ------------------------------------------------------------------
    def update_tet_attributes(self, tet):
        surface = tet.extract_surface()
        cell_to_face = _decode(tet.field_data["cell_to_face"])

        cells = tet.cells.reshape(-1, 5)[:, 1:]
        tet.add_field_data(cells, "cells")
        tet.add_field_data(tet.points, "cell_vertices")
        faces = surface.faces.reshape(-1, 4)[:, 1:]
        tet.add_field_data(faces, "faces")
        tet.add_field_data(surface.points, "face_vertices")

        n = tet.number_of_cells
        tet.cell_data["face_normal"] = np.full((n, 3), np.nan)
        normals = surface.face_normals
        for ci, fi in cell_to_face.items():
            fns = normals[fi]
            tet.cell_data["face_normal"][ci] = fns[np.argmin(fns[:, 2])]
        with np.errstate(invalid="ignore"):
            tet.cell_data["face_normal"] /= np.linalg.norm(tet.cell_data["face_normal"], axis=1)[:, None]

        tet.cell_data["face_center"] = np.full((n, 3), np.nan)
        fcenters = surface.cell_centers().points
        for ci, fi in cell_to_face.items():
            fcs = fcenters[fi]
            tet.cell_data["face_center"][ci] = fcs[np.argmin(fcs[:, 2])]

        tet.cell_data["cell_center"] = tet.cell_centers().points

        bottom_thr = np.nanmin(tet.cell_data["face_center"][:, 2]) + 0.3
        bot_mask = tet.cell_data["face_center"][:, 2] < bottom_thr
        tet.cell_data["is_bottom"] = bot_mask
        bottom_cells = np.where(bot_mask)[0]

        fnorms = tet.cell_data["face_normal"].copy()
        fnorms[bot_mask] = np.nan
        tet.cell_data["overhang_angle"] = np.arccos(np.dot(fnorms, UP))
        od = fnorms[:, :2].copy()
        od /= np.linalg.norm(od, axis=1)[:, None]
        tet.cell_data["overhang_direction"] = od

        IN_AIR = 1.0
        tet.cell_data["in_air"] = np.full(n, False)
        _, paths = nx.multi_source_dijkstra(self.cell_neighbour_graph, set(bottom_cells.tolist()))
        max_len = max((len(p) for p in paths.values()), default=1)
        tet.cell_data["path_to_bottom"] = np.full((n, max_len), -1)
        for ci, p in paths.items():
            tet.cell_data["path_to_bottom"][ci, : len(p)] = p
            if len(p) > 1:
                heights = tet.cell_data["cell_center"][p, 2]
                if np.any(heights > tet.cell_data["cell_center"][ci, 2] + IN_AIR):
                    tet.cell_data["in_air"][ci] = True
        return tet

    def calculate_tet_attributes(self, tet):
        surface = tet.extract_surface()
        cells = tet.cells.reshape(-1, 5)[:, 1:]
        tet.add_field_data(cells, "cells")
        tet.add_field_data(tet.points, "cell_vertices")
        faces = surface.faces.reshape(-1, 4)[:, 1:]
        tet.add_field_data(faces, "faces")
        tet.add_field_data(surface.points, "face_vertices")
        face_verts = surface.points

        cell_to_face = {}
        face_to_cell = {fi: [] for fi in range(len(faces))}
        c2fv, f2cv = {}, {}
        for cvi, cv in enumerate(tet.field_data["cell_vertices"].reshape(-1, 3)):
            fvi = np.where((face_verts == cv).all(axis=1))[0]
            if len(fvi) == 1:
                c2fv[cvi] = fvi[0]
                f2cv[fvi[0]] = cvi
        for ci, cell in enumerate(tet.field_data["cells"]):
            fvis = [c2fv[cv] for cv in cell if cv in c2fv]
            if len(fvis) >= 3:
                ext = surface.extract_points(fvis, adjacent_cells=False)
                if ext.number_of_cells >= 1:
                    cell_to_face[ci] = list(ext.cell_data["vtkOriginalCellIds"])
                    for fi in ext.cell_data["vtkOriginalCellIds"]:
                        face_to_cell[fi].append(ci)
        tet.add_field_data(_encode(cell_to_face), "cell_to_face")
        tet.add_field_data(_encode(face_to_cell), "face_to_cell")

        tet.cell_data["has_face"] = np.zeros(tet.number_of_cells)
        for ci in cell_to_face:
            tet.cell_data["has_face"][ci] = 1

        tet = self.update_tet_attributes(tet)
        bot_mask = tet.cell_data["is_bottom"]
        bottom_cells = np.where(bot_mask)[0]
        tet.cell_data["overhang_angle"][bottom_cells] = np.nan
        return tet, bot_mask, bottom_cells

    # ------------------------------------------------------------------
    # Rotation field
    # ------------------------------------------------------------------
    def calculate_path_length_gradient(self, tet, MAX_OVERHANG, smoothing, set_zero):
        n = tet.number_of_cells
        grad = np.zeros(n)
        cdb = np.full(n, np.nan)
        dists, paths = nx.multi_source_dijkstra(self.cell_neighbour_graph, set(self.bottom_cells.tolist()))
        closest = np.zeros(n, dtype=int)
        for ci in range(n):
            fn = tet.cell_data["face_normal"][ci]
            overhang = np.arccos(np.dot(fn, [0, 0, 1])) > np.deg2rad(90 + MAX_OVERHANG)
            if overhang and ci not in self.bottom_cells:
                closest[ci] = paths[ci][0]
                cdb[ci] = dists[ci]
        tet.cell_data["cell_distance_to_bottom"] = cdb

        for ci in range(n):
            if not np.isnan(cdb[ci]):
                local = list(self.cell_neighbour_dict["edge"][ci]) + [ci]
                lpl = np.array([cdb[c] for c in local])
                local = np.array(local)[~np.isnan(lpl)]
                lpl = lpl[~np.isnan(lpl)]
                if len(lpl) < 3:
                    target = tet.cell_data["cell_center"][closest[ci], :2]
                    d = target - tet.cell_data["cell_center"][ci, :2]
                    nrm = np.linalg.norm(d)
                    if nrm == 0:
                        grad[ci] = 0
                        continue
                    d /= nrm
                    cc = tet.cell_data["cell_center"][ci, :2].copy()
                    ccn = np.linalg.norm(cc)
                    if ccn == 0:
                        grad[ci] = 0
                        continue
                    cc /= ccn
                    dot = np.dot(cc, d)
                    grad[ci] = dot / abs(dot) if abs(dot) > 1e-9 else 0
                else:
                    pts = np.hstack((tet.cell_data["cell_center"][local, :2], lpl[:, None]))
                    _, n_vec = _plane_fit(pts.T)
                    cc = tet.cell_data["cell_center"][ci, :2]
                    ccn = np.linalg.norm(cc)
                    if ccn == 0:
                        g = 0
                    else:
                        g = np.dot(cc / ccn, n_vec[:2])
                    if np.isnan(g):
                        nbrs = grad[local]
                        nbrs = nbrs[~np.isnan(nbrs)]
                        g = np.mean(nbrs) if len(nbrs) else 0
                        if np.isnan(g):
                            g = 0
                    grad[ci] = g

        if smoothing:
            for _ in range(smoothing):
                sm = np.zeros(n)
                for ci in range(n):
                    if grad[ci] != 0:
                        nbrs = list(self.cell_neighbour_dict["point"][ci])
                        local = nbrs.copy()
                        for nb in nbrs:
                            local.extend(self.cell_neighbour_dict["point"][nb])
                        local = np.array(list(set(local)))
                        local = local[grad[local] != 0]
                        sm[ci] = np.mean(grad[local]) if len(local) else 0
                grad = sm
        if not set_zero:
            grad[grad == 0] = np.nan
        tet.cell_data["path_length_to_base_gradient"] = grad
        return grad

    def initial_rotation_field(self, tet):
        p = self.p
        f = np.abs(np.deg2rad(90 + p.max_overhang_deg) - tet.cell_data["overhang_angle"])
        grad = self.calculate_path_length_gradient(
            tet, p.max_overhang_deg, p.initial_rotation_field_smoothing, p.set_initial_rotation_to_zero
        )
        if p.steep_overhang_compensation:
            f[tet.cell_data["in_air"]] += 2 * (np.deg2rad(180) - tet.cell_data["overhang_angle"][tet.cell_data["in_air"]])
        f *= grad
        f = np.clip(f * p.rotation_multiplier, -np.deg2rad(360), np.deg2rad(360))
        f = np.clip(f, np.deg2rad(p.max_neg_rotation_deg), np.deg2rad(p.max_pos_rotation_deg))
        n_init = np.sum(~np.isnan(f))
        mx_deg = np.rad2deg(np.nanmax(np.abs(f))) if n_init > 0 else 0
        self.log(f"[opt] overhang cells detected: {n_init}/{tet.number_of_cells}, max initial tilt: {mx_deg:.2f}°")
        tet.cell_data["initial_rotation_field"] = f
        return f

    @staticmethod
    def calculate_rotation_matrices(tet, rotation_field):
        tang = np.cross(np.array([0, 0, 1]), tet.cell_data["cell_center"][:, :2])
        with np.errstate(invalid="ignore"):
            tang /= np.linalg.norm(tang, axis=1)[:, None]
        tang[np.isnan(tang).any(axis=1)] = [1, 0, 0]
        return R.from_rotvec(rotation_field[:, None] * tang).as_matrix()

    @staticmethod
    def _unique_vertices_rotated(tet, rf):
        rmat = S4Deformer.calculate_rotation_matrices(tet, rf)
        n = tet.number_of_cells
        uv = np.zeros((n, 4, 3))
        for ci, cell in enumerate(tet.field_data["cells"]):
            uv[ci] = tet.field_data["cell_vertices"][cell]
        cc = tet.cell_data["cell_center"]
        return cc.reshape(-1, 1, 3, 1) + rmat.reshape(-1, 1, 3, 3) @ (uv.reshape(-1, 4, 3, 1) - cc.reshape(-1, 1, 3, 1))

    def apply_rotation_field_unique(self, tet, rf):
        uv = self._unique_vertices_rotated(tet, rf)
        n = tet.number_of_cells
        cells = np.zeros((n, 5), dtype=int)
        cells[:, 0] = 4
        cells[:, 1:] = np.arange(n * 4).reshape(-1, 4)
        return pv.UnstructuredGrid(cells.flatten(), np.full(n, pv.CellType.TETRA), uv.reshape(-1, 3))

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    def optimize_rotations(self, tet):
        p = self.p
        irf = self.initial_rotation_field(tet)
        n_init = int(np.sum(~np.isnan(irf)))
        cfn = tet.field_data["cell_face_neighbours"]

        def obj(rf):
            diffs = rf[cfn[:, 0]] - rf[cfn[:, 1]]
            res_nl = np.sqrt(p.neighbour_loss_weight) * diffs
            valid = np.where(~np.isnan(irf))[0]
            res_il = rf[valid] - irf[valid]
            return np.concatenate((res_nl, res_il))

        def jac(rf):
            J = lil_matrix((len(cfn) + n_init, tet.number_of_cells), dtype=np.float32)
            c1, c2 = cfn[:, 0], cfn[:, 1]
            J[range(len(cfn)), c1] =  np.sqrt(p.neighbour_loss_weight)
            J[range(len(cfn)), c2] = -np.sqrt(p.neighbour_loss_weight)
            valid = np.where(~np.isnan(irf))[0]
            J[len(cfn) + np.arange(len(valid)), valid] = 1.0
            return J.tocsr()

        def jac_sparsity():
            sp = lil_matrix((len(cfn) + n_init, tet.number_of_cells), dtype=np.int8)
            for i, (a, b) in enumerate(cfn):
                sp[i, a] = 1
                sp[i, b] = 1
            valid = np.where(~np.isnan(irf))[0]
            for i, ci in enumerate(valid):
                sp[len(cfn) + i, ci] = 1
            return sp.tocsr()

        x0 = np.zeros(tet.number_of_cells)
        self.log(f"[opt] running least-squares (cells={tet.number_of_cells} init={n_init} iter={p.rot_iterations})")
        t0 = time.time()
        res = least_squares(
            obj, x0, jac=jac, max_nfev=p.rot_iterations,
            jac_sparsity=jac_sparsity(), method="trf", ftol=1e-6, verbose=0,
        )
        self.log(f"[opt] done in {time.time()-t0:.1f}s, cost={res.cost:.4f}")
        return res.x

    # ------------------------------------------------------------------
    # Vertex deformation
    # ------------------------------------------------------------------
    def deform_vertices(self, tet, rotation_field):
        p = self.p
        N = np.eye(4) - 1 / 4 * np.ones((4, 4))
        rmat = self.calculate_rotation_matrices(tet, rotation_field)
        old = tet.field_data["cell_vertices"][tet.field_data["cells"]]
        old_t = np.einsum("ijk,ikl->ijl", rmat, (N @ old).transpose(0, 2, 1))

        x0 = tet.points.copy().flatten()

        def obj(params):
            nv = params[: tet.number_of_points * 3].reshape(-1, 3)
            new_t = (N @ nv[tet.field_data["cells"]]).transpose(0, 2, 1)
            return np.linalg.norm(new_t - old_t, axis=(1, 2)) ** 2

        def jac(params):
            J = lil_matrix((tet.number_of_cells, len(params)), dtype=np.float32)
            nv = params[: tet.number_of_points * 3].reshape(-1, 3)
            new_t = (N @ nv[tet.field_data["cells"]]).transpose(0, 2, 1)
            diff = (new_t - old_t).transpose(0, 2, 1)
            ci = np.repeat(np.arange(tet.number_of_cells), len(tet.field_data["cells"][0]))
            vi = np.ravel(tet.field_data["cells"])
            for d in range(3):
                J[ci, vi * 3 + d] = 2 * diff[:, :, d].ravel()
            return J.tocsr()

        def jac_sparsity():
            sp = lil_matrix((tet.number_of_cells, len(x0)), dtype=np.int8)
            ci = np.repeat(np.arange(tet.number_of_cells), len(tet.field_data["cells"][0]))
            vi = np.ravel(tet.field_data["cells"])
            for d in range(3):
                sp[ci, vi * 3 + d] = 1
            return sp.tocsr()

        self.log(f"[deform] running least-squares (pts={tet.number_of_points} iter={p.deform_iterations})")
        t0 = time.time()
        res = least_squares(
            obj, x0, jac=jac, jac_sparsity=jac_sparsity(),
            max_nfev=p.deform_iterations, method="trf", x_scale="jac", verbose=0,
        )
        self.log(f"[deform] done in {time.time()-t0:.1f}s")
        return res.x[: tet.number_of_points * 3].reshape(-1, 3)

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------
    def run(self, stl_path: str):
        input_tet = self.load_stl(stl_path)
        input_tet, bot_mask, bot_cells = self.calculate_tet_attributes(input_tet)
        self.bottom_cells_mask = bot_mask
        self.bottom_cells = bot_cells
        original_input_tet = input_tet.copy()

        deformed = input_tet
        for ipass in range(self.p.num_passes):
            self.log(f"[pass {ipass+1}/{self.p.num_passes}] optimising rotation field")
            rf = self.optimize_rotations(deformed)
            self.log(f"[pass {ipass+1}/{self.p.num_passes}] deforming vertices")
            new_pts = self.deform_vertices(deformed, rf)
            new_tet = pv.UnstructuredGrid(
                deformed.cells, np.full(deformed.number_of_cells, pv.CellType.TETRA), new_pts
            )
            for k in deformed.field_data:
                new_tet.field_data[k] = deformed.field_data[k]
            for k in deformed.cell_data:
                new_tet.cell_data[k] = deformed.cell_data[k]
            new_tet = self.update_tet_attributes(new_tet)
            deformed = new_tet

        # Re-center bottom of bbox at origin based on ORIGINAL model
        x_min, x_max, y_min, y_max, z_min, _ = original_input_tet.bounds
        offset = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])
        deformed.points -= offset
        original_input_tet.points -= offset

        # The `cell_vertices` field_data caches were taken from the un-shifted
        # points; refresh them so barycentric / vertex-transform calculations
        # see the same coordinates that find_containing_cell uses.
        deformed.field_data["cell_vertices"] = deformed.points.copy()
        original_input_tet.field_data["cell_vertices"] = original_input_tet.points.copy()

        return original_input_tet, deformed
