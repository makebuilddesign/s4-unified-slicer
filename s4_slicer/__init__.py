"""S4 Unified Slicer — non-planar 4-axis slicing in a single tool.

Based on the original S4 Slicer by Joshua Bird:
    https://github.com/jyjblrd/S4_Slicer
Licensed GPL-3.0 (same as upstream).

v2.0:
    * vectorised "fast" backend (`deform_fast.py`, `gcode_transform_fast.py`)
      with ~50–150x speedup on the gcode-transform stage and ~5–20x on the
      deformation stage.
    * web UI (`s4slicer-web`) with progress bar, error terminal, and 3D
      input STL + path-preview viewports.
    * unified pipeline runner (`pipeline.py`) used by both the CLI and the
      web UI.
"""
__version__ = "2.0.0"
