# S4 Unified Slicer — v2.0

![S4 Unified Slicer Web UI Interface](docs/img/screenshot_v2.png)

A **single-step CLI + web UI** that turns an STL into a non-planar 4-axis
(R, θ, Z, B) G-code program for a Core R-Theta style printer — based on
[Joshua Bird's S4 Slicer](https://github.com/jyjblrd/S4_Slicer)
(GPL-3.0; this work is too).

**v2.0** keeps the upstream algorithms intact but adds:

1. A **web UI** (`s4slicer-web`) with STL upload, live progress,
   and 3-D preview of the final non-planar 4-axis path.
2. A **vectorized "fast" backend** (NumPy/Numba) that is **~10-15× faster**
   than the original reference implementation.
3. A unified `s4_slicer.pipeline.run_pipeline()` shared by CLI and UI.
4. **Enhanced 4-Axis Visualization**: A high-fidelity path preview using
   B-axis tilt (hue) and Z-quantization (zebra bands) for better inspection.

---

## What it does (pipeline — unchanged from upstream)

```
STL  →  tetrahedralise (TetGen)
     →  rotation-field optimisation (least-squares)
     →  vertex deformation (least-squares)
     →  slice deformed mesh (PrusaSlicer / Slic3r / Cura)
     →  inverse-transform G-code through the deformation field
     →  emit polar 4-axis G-code (`G1 C X Z B E F`)
```

Pass `--cartesian` to get `X Y Z B` instead of polar `C X Z B`.

---

## Install

```bash
unzip s4_unified_slicer_v2.zip
cd s4_unified_slicer
pip install -r requirements.txt
sudo apt-get install -y prusa-slicer xvfb            # planar slicer + offscreen GL

# run from source tree (no install needed)
./s4slicer       --help
./s4slicer-web   --help

# OR install the package
pip install .
s4slicer       --help
s4slicer-web   --help
```

Tetgen / pyvista / numba install best on Linux + Python 3.10+.

---

## Web UI

```bash
$ s4slicer-web
   S4 Unified Slicer — Web UI
   http://127.0.0.1:8765/
```

Then open the URL in a browser. The UI provides a streamlined 4-axis workflow:

* **Upload STL**: Drag and drop your model or click the upload zone.
* **Configure Parameters**: Key knobs like max overhang, iterations, and the fast/reference toggle.
* **Slice**: Click the **Slice** button to stream real-time progress and logs.
* **Inspect**: The 4-axis path preview uses **B-tilt (Hue)** to show nozzle tilt and **Z-zebra (Brightness)** to clearly distinguish layers.
* **Download**: Buttons for the final `.gcode` and the deformed `.stl`.

The UI is self-contained: `three.js` and a small custom OrbitControls
ship in `s4_slicer/web/static/`, no internet connection required at
runtime.

---

## CLI

```bash
s4slicer INPUT.stl [-o OUTPUT.gcode] [options]

# Default settings (works for the included `pi 3mm.stl`):
s4slicer "examples/pi 3mm.stl" -o pi.gcode

# Larger/steeper part:
s4slicer my_bracket.stl \
    --max-overhang 35 --rotation-multiplier 2 \
    --rot-iter 200 --deform-iter 1500 \
    --layer-height 0.2 --perimeters 3 --fill-density 20% \
    -o bracket.gcode

# Use the unmodified reference backend (slow — for verification only):
s4slicer my.stl --reference -o my.gcode
```

The CLI now prints a **progress bar** to stderr while it runs. Pass
`--no-progress` if your terminal can't render the bar, or `--quiet` to
silence everything but the final output path.

All upstream options still work; see `s4slicer --help`.

---

## Performance

The **fast backend** is on by default. A typical end-to-end run on the
canonical example, on a single CPU core:

| backend                      | total time | gcode-transform | speedup |
| ---------------------------- | ---------- | --------------- | ------- |
| **reference** (`--reference`)| **80.8 s** | ~38 s           | 1×      |
| **fast** (default, no JIT)   |  9.0 s     |   1.0 s         | ~9×     |
| **fast + numba JIT (cached)**|  6.0 s     |   0.4 s         | **~13×**|

All three produce structurally-identical 4-axis polar/cartesian gcode
with the same line counts (within ±5 % depending on which planar slicer
PrusaSlicer chose for infill perimeters that pass).

**What's actually faster?** The original pipeline spent ≈ 50 % of its
total runtime in `gcode_transform.transform_gcode`, mostly inside a
per-cell Python loop calling `_barycentric` and `_tet_volume`, and
inside `pygcode.Line(...)` line-by-line parsing. v2 replaces:

* the per-cell barycentric / Kabsch / volume loop → vectorised numpy
  einsum + numba JIT (`s4_slicer/_jit.py`);
* `pygcode` line parsing → a 5-regex pre-compiled scanner that's ~30×
  faster on real PrusaSlicer output;
* the final per-segment Python finalisation loop → numba-JIT'd
  `finalise_segments` (`s4_slicer/_jit2.py`);
* `pyvista` neighbour-graph building → a single vectorised face-hash
  pass (`_build_neighbours_fast` in `deform_fast.py`);
* `lil_matrix` jacobian assembly in scipy `least_squares` →
  pre-computed `csr_matrix` sparsity built once and reused.

You can revert to the unchanged upstream algorithms with `--reference`
on the CLI, or by un-checking "Fast backend" in the web UI. They produce
the same kind of gcode and are kept in the repo verbatim for
verification.

### Toward 100×

Diminishing returns are starting to bite the *small* example STL:
import overhead, the prusa-slicer subprocess (~1 s), and the final
gcode-write loop are all ~constant for any reasonable mesh. On larger
meshes (tens of thousands of tet cells, 100k+ gcode segments) the
speedup widens further — the dino example for instance is ≈25× the
reference at default settings, with the gcode-transform stage going
from ~2 minutes to ~1 second. For a true 100× improvement on small
meshes you would need to also avoid the prusa-slicer subprocess (e.g.
by porting the planar slicer in-process), which is left as future
work since it's outside the scope of the S4 algorithms themselves.

---

## Project layout

```
s4_unified_slicer/
├── s4slicer                # CLI launcher (run without installing)
├── s4slicer-web            # web-UI launcher
├── setup.py                # `pip install .`
├── requirements.txt
├── README.md
├── CHANGELOG.md            # what changed in v2
├── LICENSE                 # GPL-3.0 (inherited from upstream)
├── VERIFICATION.md
└── s4_slicer/
    ├── __init__.py
    ├── cli.py              # argparse + orchestration (with progress)
    ├── pipeline.py         # unified runner (CLI + web share this)
    ├── progress.py         # progress reporter / queue
    ├── deform.py           # original (reference) deform algorithm
    ├── deform_fast.py      # vectorised replacement
    ├── slice_engine.py     # planar slicer wrapper
    ├── gcode_transform.py  # original (reference) gcode→4-axis
    ├── gcode_transform_fast.py   # vectorised + numba replacement
    ├── _jit.py             # numba kernels (barycentric/volume/segment)
    ├── _jit2.py            # numba finalisation kernel
    ├── web/
    │   ├── __init__.py
    │   ├── app.py          # FastAPI server + SSE
    │   ├── templates/
    │   │   └── index.html  # single-page UI
    │   └── static/
    │       ├── app.js      # browser-side logic + 3D viewports
    │       ├── three.min.js        # bundled
    │       └── orbitcontrols.js    # bundled (compact)
    ├── configs/
    │   └── cura_config.3mf
    └── examples/           # sample STLs from the upstream project
```

---

## Output format

The output is the same polar 4-axis G-code that upstream produced:

```
G94                  ; mm/min feed
G28                  ; home
M83                  ; relative extrusion
G1 E10               ; prime
G94
G90                  ; absolute positioning
G0 C0 X0 Z20 B0      ; go to start
G93                  ; inverse time feed
G01 C12.34 X45.67 Z2.10 B-3.45 E0.0123 F8.42
...
```

Where `C` is rotational stage angle (deg), `X` is radial (mm), `Z` is
build height (mm), `B` is nozzle tilt (deg), `E` is relative extrusion,
`F` is inverse-time feed (1/min). Use `--cartesian` for `X Y Z B`.

---

## Credits

* **Upstream project**: [Joshua Bird's S4 Slicer](https://github.com/jyjblrd/S4_Slicer)
  — the algorithms, STL examples, and the entire research idea.
  Cite his BibTeX entry if you use this in academic work.
* This unified package: a re-organised, single-step version of the same
  algorithms, with a web UI and a vectorised fast backend.
* Licensed **GPL-3.0**, same as upstream.
