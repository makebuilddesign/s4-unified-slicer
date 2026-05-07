"""Command-line entry point for the S4 unified non-planar slicer.

Usage:
    s4slicer INPUT.stl -o OUTPUT.gcode [options]

Single-step pipeline:
    STL  →  tetrahedralise  →  rotation field  →  deform mesh
         →  slice deformed mesh (PrusaSlicer)
         →  inverse-transform G-code  →  4-axis non-planar G-code
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .pipeline import PipelineConfig, run_pipeline
from .progress import ProgressReporter, STAGE_LABELS


BANNER = rf"""
 ____  _  _    ____  _ _                 
/ ___|| || |  / ___|| (_) ___ ___ _ __ 
\___ \| || |_ \___ \| | |/ __/ _ \ '__|
 ___) |__   _| ___) | | | (_|  __/ |   
|____/   |_|  |____/|_|_|\___\___|_|   
                                        S4 Unified Slicer v{__version__}
A single-step CLI: STL → non-planar 4-axis G-code.
"""


def _parse():
    ap = argparse.ArgumentParser(
        prog="s4slicer",
        description="S4 Unified Slicer — non-planar 4-axis slicing in one step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("stl", type=str, help="Input STL file")
    ap.add_argument("-o", "--output", type=str, required=False)
    ap.add_argument("--workdir", type=str, default=None)
    ap.add_argument("--keep-intermediates", action="store_true")

    g = ap.add_argument_group("Pre-processing")
    g.add_argument("--rotate-x", type=float, default=0.0)
    g.add_argument("--rotate-y", type=float, default=0.0)
    g.add_argument("--rotate-z", type=float, default=0.0)
    g.add_argument("--scale", type=float, default=1.0)
    g.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("X", "Y", "Z"))
    g.add_argument("--make-manifold", action="store_true")

    g = ap.add_argument_group("Rotation field")
    g.add_argument("--max-overhang", type=float, default=30.0)
    g.add_argument("--neighbour-loss-weight", type=float, default=20.0)
    g.add_argument("--rotation-multiplier", type=float, default=2.0)
    g.add_argument("--smoothing", type=int, default=30)
    g.add_argument("--rot-iter", type=int, default=100)
    g.add_argument("--no-steep-overhang-comp", action="store_true", dest="no_steep")
    g.add_argument("--set-initial-rotation-zero", action="store_true",
                   dest="set_initial_rotation_zero")
    g.add_argument("--max-pos-rotation", type=float, default=3600.0)
    g.add_argument("--max-neg-rotation", type=float, default=-3600.0)
    g.add_argument("--num-passes", type=int, default=1)

    g = ap.add_argument_group("Vertex deformation")
    g.add_argument("--deform-iter", type=int, default=1000)

    g = ap.add_argument_group("Slicer (PrusaSlicer / Cura)")
    g.add_argument("--layer-height", type=float, default=0.2)
    g.add_argument("--first-layer-height", type=float, default=0.2)
    g.add_argument("--nozzle-diameter", type=float, default=0.4)
    g.add_argument("--filament-diameter", type=float, default=1.75)
    g.add_argument("--perimeters", type=int, default=2)
    g.add_argument("--fill-density", type=str, default="15%")
    g.add_argument("--top-solid-layers", type=int, default=3)
    g.add_argument("--bottom-solid-layers", type=int, default=0)
    g.add_argument("--skirts", type=int, default=0)
    g.add_argument("--support-material", action="store_true")
    g.add_argument("--input-gcode", type=str, default="")
    g.add_argument("--skip-slice", action="store_true")

    g = ap.add_argument_group("4-axis G-code conversion")
    g.add_argument("--seg-size", type=float, default=0.6)
    g.add_argument("--max-rotation", type=float, default=30.0)
    g.add_argument("--min-rotation", type=float, default=-130.0)
    g.add_argument("--nozzle-offset", type=float, default=42.0)
    g.add_argument("--rotation-averaging", type=float, default=0.2)
    g.add_argument("--retraction-length", type=float, default=1.0)
    g.add_argument("--rotation-max-delta", type=float, default=1.0)
    g.add_argument("--max-extrusion-mult", type=float, default=10.0)
    g.add_argument("--cartesian", action="store_true")

    g = ap.add_argument_group("Backend")
    g.add_argument("--reference", action="store_true",
                   help="Use the unmodified reference backend (slow). "
                        "Default is the fast vectorised backend.")

    ap.add_argument("--quiet", "-q", action="store_true")
    ap.add_argument("--no-progress", action="store_true",
                    help="Don't print a progress bar to stderr")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap.parse_args()


def main():
    args = _parse()
    if not args.quiet:
        print(BANNER, file=sys.stderr)

    cfg = PipelineConfig(
        output=args.output or "",
        workdir=args.workdir or "",
        keep_intermediates=args.keep_intermediates,
        rotate_x=args.rotate_x, rotate_y=args.rotate_y, rotate_z=args.rotate_z,
        scale=args.scale, offset=tuple(args.offset), make_manifold=args.make_manifold,
        max_overhang=args.max_overhang,
        neighbour_loss_weight=args.neighbour_loss_weight,
        rotation_multiplier=args.rotation_multiplier,
        smoothing=args.smoothing,
        rot_iter=args.rot_iter,
        no_steep=args.no_steep,
        set_initial_rotation_zero=args.set_initial_rotation_zero,
        max_pos_rotation=args.max_pos_rotation,
        max_neg_rotation=args.max_neg_rotation,
        num_passes=args.num_passes,
        deform_iter=args.deform_iter,
        layer_height=args.layer_height,
        first_layer_height=args.first_layer_height,
        nozzle_diameter=args.nozzle_diameter,
        filament_diameter=args.filament_diameter,
        perimeters=args.perimeters,
        fill_density=args.fill_density,
        top_solid_layers=args.top_solid_layers,
        bottom_solid_layers=args.bottom_solid_layers,
        skirts=args.skirts,
        support_material=args.support_material,
        input_gcode=args.input_gcode or "",
        skip_slice=args.skip_slice,
        seg_size=args.seg_size,
        max_rotation=args.max_rotation,
        min_rotation=args.min_rotation,
        nozzle_offset=args.nozzle_offset,
        rotation_averaging=args.rotation_averaging,
        retraction_length=args.retraction_length,
        rotation_max_delta=args.rotation_max_delta,
        max_extrusion_mult=args.max_extrusion_mult,
        cartesian=args.cartesian,
        fast=not args.reference,
        quiet=args.quiet,
    )

    progress = ProgressReporter(job_id="cli")

    # Background drainer so log lines and progress reach stderr.
    import threading, queue as _q, time as _t
    stop = threading.Event()
    def _drain():
        last_pct = -1
        while not stop.is_set():
            try:
                evt = progress.queue.get(timeout=0.2)
            except _q.Empty:
                continue
            if args.quiet:
                continue
            t = evt.get("type")
            if t == "log":
                print(evt["message"], file=sys.stderr, flush=True)
            elif t == "progress" and not args.no_progress:
                pct = int(evt["global_progress"] * 100)
                if pct != last_pct:
                    last_pct = pct
                    bar = "█" * (pct // 4) + "░" * (25 - pct // 4)
                    print(f"\r  [{bar}] {pct:3d}%  {evt['stage_label']:<28}",
                          end="", file=sys.stderr, flush=True)
            elif t == "error":
                print(f"\nERROR: {evt['message']}", file=sys.stderr, flush=True)
            elif t == "done":
                if not args.no_progress:
                    print(f"\r  [{'█'*25}] 100%  done in {evt['elapsed']:.1f}s          ",
                          file=sys.stderr, flush=True)
    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    try:
        result = run_pipeline(args.stl, cfg, progress=progress)
        progress.done(output_path=result["output_gcode"],
                      deformed_stl=result.get("deformed_stl", ""),
                      input_stl=args.stl,
                      extras={"elapsed": result["elapsed"],
                              "stats":   result["stats"]})
        # Allow the drainer to flush.
        _t.sleep(0.3)
        stop.set()
        t.join(timeout=1.0)
        print(result["output_gcode"])
        return 0
    except FileNotFoundError as e:
        progress.error(str(e))
        _t.sleep(0.2); stop.set()
        return 2
    except RuntimeError as e:
        progress.error(str(e))
        _t.sleep(0.2); stop.set()
        return 3
    except Exception as e:
        import traceback
        progress.log(traceback.format_exc(), level="error")
        progress.error(f"{type(e).__name__}: {e}")
        _t.sleep(0.2); stop.set()
        return 1


if __name__ == "__main__":
    sys.exit(main())
