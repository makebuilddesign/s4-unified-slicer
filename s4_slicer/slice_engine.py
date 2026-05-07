"""Wrapper around an external planar slicer (PrusaSlicer / Slic3r / CuraEngine).

The S4 pipeline needs a *planar* G-code of the *deformed* STL; that gcode is
then inverse-transformed into a non-planar 4-axis program.  We use whatever
slicer is available on the system, falling back through a small priority list.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SliceParams:
    layer_height: float = 0.2
    first_layer_height: float = 0.2
    nozzle_diameter: float = 0.4
    filament_diameter: float = 1.75
    perimeters: int = 2
    fill_density: str = "15%"
    top_solid_layers: int = 3
    bottom_solid_layers: int = 0          # bottom is non-planar in S4 — skip
    skirts: int = 0
    support_material: bool = False
    bed_size: float = 400.0               # mm — used for prusa-slicer placement
    extra_args: tuple = ()


def find_slicer() -> tuple[str, str] | None:
    """Return (binary_path, kind) or None."""
    # 1. Check PATH
    for name in ("prusa-slicer", "prusaslicer", "slic3r", "superslicer"):
        path = shutil.which(name)
        if path:
            return path, "prusa"
    for name in ("CuraEngine", "curaengine"):
        path = shutil.which(name)
        if path:
            return path, "cura"

    # 2. macOS-specific app paths
    import sys
    if sys.platform == "darwin":
        mac_paths = [
            ("/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer", "prusa"),
            ("/Applications/SuperSlicer.app/Contents/MacOS/SuperSlicer", "prusa"),
            ("/Applications/UltiMaker Cura.app/Contents/MacOS/CuraEngine", "cura"),
        ]
        for path, kind in mac_paths:
            if os.path.exists(path):
                return path, kind

    return None


def slice_stl(stl_path: str, out_gcode: str, p: SliceParams | None = None, log=print) -> str:
    p = p or SliceParams()
    found = find_slicer()
    if not found:
        raise RuntimeError(
            "No slicer found on PATH. Install prusa-slicer (preferred) or slic3r,\n"
            "or supply your own gcode with `--input-gcode` and `--skip-slice`."
        )
    binpath, kind = found
    if kind == "prusa":
        return _slice_prusa(binpath, stl_path, out_gcode, p, log)
    elif kind == "cura":
        return _slice_cura(binpath, stl_path, out_gcode, p, log)
    raise RuntimeError(f"Unknown slicer kind: {kind}")


def _slice_prusa(binpath: str, stl: str, out: str, p: SliceParams, log) -> str:
    # Pick a bed size big enough to hold the model (defaulted huge),
    # place the model at its centre, and post-translate the gcode back
    # so origin = bed centre (which is what the S4 gcode-transform expects).
    bed = p.bed_size
    cx = cy = bed / 2.0
    bed_shape = f"0x0,{bed}x0,{bed}x{bed},0x{bed}"

    # Ensure output path is safe for CLI (esp. on macOS)
    # Some versions of PrusaSlicer have issues with spaces in --output
    op = Path(out)
    if " " in op.name:
        out = str(op.parent / op.name.replace(" ", "_"))
        log(f"[slice] sanitised output filename to {Path(out).name}")

    # Detect if the deformed mesh is floating or sinking
    import pyvista as pv
    import numpy as np
    m = pv.read(stl)
    z_min_def = m.bounds[4]
    
    # Force the model to sit exactly on the bed (Z=0)
    # This is critical for PrusaSlicer to generate a valid first layer.
    if abs(z_min_def) > 0.001:
        log(f"[slice] shifting model by {-z_min_def:.3f}mm to sit on bed")
        m.points[:, 2] -= z_min_def
        stl_on_bed = stl.replace(".stl", "_on_bed.stl")
        m.save(stl_on_bed)
        stl = stl_on_bed

    # Try with the requested first-layer-height, then retry with thicker
    # values if PrusaSlicer fails with "no extrusions in first layer".
    attempts = [p.first_layer_height, 0.4, 0.6, 1.0]
    seen = set()
    attempts = [h for h in attempts if not (h in seen or seen.add(h))]

    for attempt_idx, flh in enumerate(attempts):
        cmd = [
            binpath, "--slice",
            "--output", out,
            "--layer-height", str(p.layer_height),
            "--first-layer-height", str(flh),
            "--nozzle-diameter", str(p.nozzle_diameter),
            "--filament-diameter", str(p.filament_diameter),
            "--perimeters", str(p.perimeters),
            "--fill-density", p.fill_density,
            "--top-solid-layers", str(p.top_solid_layers),
            "--bottom-solid-layers", str(p.bottom_solid_layers),
            "--skirts", str(p.skirts),
            "--center", f"{cx},{cy}",
            "--bed-shape", bed_shape,
            "--gcode-flavor", "marlin",
        ]
        if p.support_material:
            cmd.append("--support-material")
        cmd.extend(p.extra_args)
        cmd.append(stl)
        
        if attempt_idx == 0:
            log(f"[slice] running prusa-slicer (bed={bed}x{bed}, centre={cx},{cy})")
        else:
            log(f"[slice] retrying with flh={flh}mm (attempt {attempt_idx+1}/{len(attempts)})")
            
        res = subprocess.run(cmd, capture_output=True, text=True)
        combined = (res.stdout or "") + (res.stderr or "")
        
        if os.path.exists(out) and os.path.getsize(out) > 100:
            break
            
        if "no extrusions in the first layer" in combined and attempt_idx < len(attempts) - 1:
            log(f"[slice] empty first layer detected")
            continue

        # If we failed but rc=0, log the output to see what happened
        log(f"[slice] prusa-slicer failed (rc={res.returncode})")
        if res.stdout: log("[stdout] " + res.stdout[-1000:])
        if res.stderr: log("[stderr] " + res.stderr[-1000:])
        
        if not os.path.exists(out):
            raise RuntimeError(f"prusa-slicer failed and produced no gcode at {out}")

    if not os.path.exists(out):
        raise RuntimeError(f"prusa-slicer failed after {len(attempts)} attempts — no gcode at {out}")
    # Translate the gcode by (-cx, -cy) so origin = bed centre = STL origin.
    _shift_gcode_xy(out, -cx, -cy)
    log(f"[slice] gcode written to {out} ({os.path.getsize(out)/1024:.1f} KB), recentred to origin")
    return out


def _shift_gcode_xy(path: str, dx: float, dy: float):
    """Re-write each G0/G1 X/Y coordinate by adding (dx,dy). In-place."""
    import re
    rx = re.compile(r"(?<![A-Za-z])X(-?\d+(?:\.\d+)?)")
    ry = re.compile(r"(?<![A-Za-z])Y(-?\d+(?:\.\d+)?)")
    with open(path, "r") as fh:
        lines = fh.readlines()
    with open(path, "w") as fh:
        for ln in lines:
            stripped = ln.lstrip()
            if stripped.startswith(("G1", "G0", "G01", "G00")):
                ln = rx.sub(lambda m: f"X{float(m.group(1)) + dx:.5f}", ln)
                ln = ry.sub(lambda m: f"Y{float(m.group(1)) + dy:.5f}", ln)
            fh.write(ln)


def _slice_cura(binpath: str, stl: str, out: str, p: SliceParams, log) -> str:
    cmd = [
        binpath, "slice", "-v",
        "-o", out,
        "-s", f"layer_height={p.layer_height}",
        "-s", f"machine_nozzle_size={p.nozzle_diameter}",
        "-s", f"material_diameter={p.filament_diameter}",
        "-s", f"wall_line_count={p.perimeters}",
        "-s", f"infill_sparse_density={p.fill_density.rstrip('%')}",
        "-s", f"top_layers={p.top_solid_layers}",
        "-s", f"bottom_layers={p.bottom_solid_layers}",
        "-s", f"skirt_line_count={p.skirts}",
        "-l", stl,
    ]
    log(f"[slice] running CuraEngine")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log(res.stdout[-2000:])
        log(res.stderr[-2000:])
        raise RuntimeError(f"CuraEngine failed: rc={res.returncode}")
    if not os.path.exists(out):
        raise RuntimeError(f"CuraEngine produced no gcode at {out}")
    log(f"[slice] gcode written to {out} ({os.path.getsize(out)/1024:.1f} KB)")
    return out
