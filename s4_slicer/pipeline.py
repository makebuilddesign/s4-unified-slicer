"""End-to-end pipeline runner that wires deform → slice → gcode-transform
together with progress reporting and a thread-safe job model.

Used by both the CLI (`cli.py`) and the web UI (`web/app.py`).
"""
from __future__ import annotations

import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .progress import ProgressReporter, NullProgress


@dataclass
class PipelineConfig:
    """All knobs the pipeline supports.  Mirrors `cli._parse` defaults."""
    # I/O
    output: str = ""
    workdir: str = ""
    keep_intermediates: bool = False
    # pre-processing
    rotate_x: float = 0.0
    rotate_y: float = 0.0
    rotate_z: float = 0.0
    scale: float = 1.0
    offset: tuple = (0.0, 0.0, 0.0)
    make_manifold: bool = False
    # rotation field
    max_overhang: float = 30.0
    neighbour_loss_weight: float = 20.0
    rotation_multiplier: float = 2.0
    smoothing: int = 30
    rot_iter: int = 100
    no_steep: bool = False
    set_initial_rotation_zero: bool = False
    max_pos_rotation: float = 3600.0
    max_neg_rotation: float = -3600.0
    num_passes: int = 1
    # deform
    deform_iter: int = 1000
    # slicer
    layer_height: float = 0.2
    first_layer_height: float = 0.2
    nozzle_diameter: float = 0.4
    filament_diameter: float = 1.75
    perimeters: int = 2
    fill_density: str = "15%"
    top_solid_layers: int = 3
    bottom_solid_layers: int = 0
    skirts: int = 0
    support_material: bool = False
    input_gcode: str = ""
    skip_slice: bool = False
    # 4-axis
    seg_size: float = 0.6
    max_rotation: float = 30.0
    min_rotation: float = -130.0
    nozzle_offset: float = 42.0
    rotation_averaging: float = 0.2
    retraction_length: float = 1.0
    rotation_max_delta: float = 1.0
    max_extrusion_mult: float = 10.0
    cartesian: bool = False
    # backend
    fast: bool = False               # [Experimental] use vectorised modules
    quiet: bool = False


def run_pipeline(stl_in: str,
                 cfg: PipelineConfig,
                 progress: Optional[ProgressReporter] = None) -> dict:
    """Run the full STL → 4-axis G-code pipeline.

    Returns dict with 'output_gcode', 'deformed_stl', 'elapsed', 'stats'.
    """
    progress = progress or NullProgress()

    # Pick fast vs reference implementations.
    if cfg.fast:
        from .deform_fast          import DeformParams, S4Deformer
        from .gcode_transform_fast import GCodeParams, transform_gcode
    else:
        from .deform               import DeformParams, S4Deformer
        from .gcode_transform      import GCodeParams, transform_gcode
    from .slice_engine import SliceParams, find_slicer, slice_stl

    log = (lambda *a, **k: None) if cfg.quiet else (
        lambda *a, **k: progress.log(" ".join(str(x) for x in a)))

    stl_in = Path(stl_in).resolve()
    if not stl_in.is_file():
        raise FileNotFoundError(f"STL not found: {stl_in}")

    out = Path(cfg.output) if cfg.output else stl_in.with_suffix(".gcode")
    out = out.resolve()

    if cfg.workdir:
        wd = Path(cfg.workdir).resolve()
        wd.mkdir(parents=True, exist_ok=True)
        keep_wd = True
    else:
        wd = Path(tempfile.mkdtemp(prefix="s4slicer_"))
        keep_wd = cfg.keep_intermediates

    log(f"[main] working dir: {wd}")
    log(f"[main] input STL: {stl_in}")
    log(f"[main] output gcode: {out}")
    log(f"[main] backend: {'fast (vectorised)' if cfg.fast else 'reference'}")

    # Pre-flight checks
    if not (cfg.skip_slice or cfg.input_gcode):
        if find_slicer() is None:
            raise RuntimeError(
                "No slicer (prusa-slicer / slic3r / CuraEngine) found. "
                "Install one, or provide pre-sliced gcode.")

    t0 = time.time()

    # --- Stage 1: deform -------------------------------------------------
    dp = DeformParams(
        neighbour_loss_weight=cfg.neighbour_loss_weight,
        max_overhang_deg=cfg.max_overhang,
        rotation_multiplier=cfg.rotation_multiplier,
        set_initial_rotation_to_zero=cfg.set_initial_rotation_zero,
        initial_rotation_field_smoothing=cfg.smoothing,
        max_pos_rotation_deg=cfg.max_pos_rotation,
        max_neg_rotation_deg=cfg.max_neg_rotation,
        rot_iterations=cfg.rot_iter,
        steep_overhang_compensation=not cfg.no_steep,
        deform_iterations=cfg.deform_iter,
        num_passes=cfg.num_passes,
        rotate_x_deg=cfg.rotate_x,
        rotate_y_deg=cfg.rotate_y,
        rotate_z_deg=cfg.rotate_z,
        scale=cfg.scale,
        part_offset=tuple(cfg.offset),
        make_manifold=cfg.make_manifold,
    )
    if cfg.fast:
        deformer = S4Deformer(dp, log=log, progress=progress)
    else:
        deformer = S4Deformer(dp, log=log)
    input_tet, deformed_tet = deformer.run(str(stl_in))

    deformed_stl = wd / f"{stl_in.stem}_deformed.stl"
    deformed_tet.extract_surface().save(str(deformed_stl))
    log(f"[main] saved deformed STL: {deformed_stl}")

    if cfg.skip_slice and not cfg.input_gcode:
        elapsed = time.time() - t0
        log(f"[main] --skip-slice — done in {elapsed:.1f}s")
        return {
            "output_gcode": str(deformed_stl),
            "deformed_stl": str(deformed_stl),
            "elapsed": elapsed,
            "stats": {},
        }

    # --- Stage 2: slice --------------------------------------------------
    progress.stage("slice", 0.0)
    if cfg.input_gcode:
        sliced_gcode = Path(cfg.input_gcode).resolve()
        if not sliced_gcode.is_file():
            raise FileNotFoundError(f"--input-gcode not found: {sliced_gcode}")
        log(f"[main] using user-supplied gcode: {sliced_gcode}")
    else:
        # (Slicer already checked in pre-flight)
        sp = SliceParams(
            layer_height=cfg.layer_height,
            first_layer_height=cfg.first_layer_height,
            nozzle_diameter=cfg.nozzle_diameter,
            filament_diameter=cfg.filament_diameter,
            perimeters=cfg.perimeters,
            fill_density=cfg.fill_density,
            top_solid_layers=cfg.top_solid_layers,
            bottom_solid_layers=cfg.bottom_solid_layers,
            skirts=cfg.skirts,
            support_material=cfg.support_material,
        )
        sliced_gcode = wd / f"{stl_in.stem}_sliced.gcode"
        sliced_gcode = slice_stl(str(deformed_stl), str(sliced_gcode), sp, log=log)
    progress.stage("slice", 1.0)

    # --- Stage 3: gcode → 4-axis ----------------------------------------
    progress.stage("transform", 0.0)
    gp = GCodeParams(
        seg_size=cfg.seg_size,
        max_rotation_deg=cfg.max_rotation,
        min_rotation_deg=cfg.min_rotation,
        nozzle_offset=cfg.nozzle_offset,
        rotation_averaging_alpha=cfg.rotation_averaging,
        retraction_length=cfg.retraction_length,
        rotation_max_delta_deg=cfg.rotation_max_delta,
        max_extrusion_multiplier=cfg.max_extrusion_mult,
        output_polar=not cfg.cartesian,
    )
    if cfg.fast:
        transform_gcode(input_tet, deformed_tet, str(sliced_gcode),
                        str(out), gp, log=log, progress=progress)
    else:
        transform_gcode(input_tet, deformed_tet, str(sliced_gcode),
                        str(out), gp, log=log)

    elapsed = time.time() - t0
    log(f"[main] DONE in {elapsed:.1f}s — final 4-axis gcode: {out}")

    # Stats
    try:
        with open(out) as fh:
            n_lines = sum(1 for _ in fh)
    except Exception:
        n_lines = 0

    if not keep_wd:
        try:
            shutil.rmtree(wd)
        except Exception:
            pass

    return {
        "output_gcode": str(out),
        "deformed_stl": str(deformed_stl) if keep_wd else "",
        "elapsed":      elapsed,
        "stats":        {"output_lines": n_lines},
    }
