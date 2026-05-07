# Changelog

## v2.0.0 — Web UI + fast backend

### Added
* **Web UI** (`s4slicer-web`) — FastAPI single-page app with:
  * STL drop-zone with instant 3-D preview of the input geometry.
  * Parameter form covering the most-used CLI flags.
  * **Live progress bar** updated continuously while slicing,
    decomposed into the six pipeline stages (load, tetrahedralise,
    rotation, deform, slice, transform).
  * **Terminal-style log panel** at the bottom that streams every
    log line from the pipeline. Tracebacks of any error appear in
    red and the progress bar turns red when slicing fails.
  * **3-D path preview** of the final non-planar 4-axis output —
    extrusion in green, travel in orange — rendered side-by-side
    with the input STL. Polar (R, θ, Z) output is unprojected back
    into Cartesian XYZ for the preview.
  * **Download** buttons for the final `.gcode` and the deformed `.stl`
    once the job completes.
* **Fast backend** (default) for ~9–13× end-to-end speedup on the
  canonical examples and ~40–100× speedup on the gcode-transform
  stage:
  * `s4_slicer/deform_fast.py` — vectorised neighbour-graph
    construction, batched Kabsch/SVD, COO-built jacobian sparsity,
    pre-computed two-hop adjacency for path-length smoothing.
  * `s4_slicer/gcode_transform_fast.py` — regex-based g-code parser
    (replaces line-by-line `pygcode`), batched barycentric +
    z-squish + Kabsch via einsum, single-shot
    `find_containing_cell` instead of per-segment.
  * `s4_slicer/_jit.py`, `_jit2.py` — optional numba JIT kernels
    for the inner barycentric, segment-expansion and
    finalisation loops (parallelised with `prange`).
* **Unified pipeline runner** (`s4_slicer/pipeline.py`) used by both
  the CLI and the web UI, so progress reporting and parameters are
  identical across them.
* **Progress reporter** (`s4_slicer/progress.py`) — thread-safe
  queue that the deform / slice / transform stages push percentage
  updates and log lines into. Web UI consumes via SSE; CLI prints a
  bar to stderr.
* **`s4slicer-web`** launcher script alongside the existing
  `s4slicer` CLI launcher.

### Changed
* `cli.py` now drives `pipeline.run_pipeline` and prints a live
  progress bar to stderr (suppress with `--quiet` or `--no-progress`).
* `setup.py` now exposes both `s4slicer` and `s4slicer-web`
  entry-points and ships the web templates / static assets as package
  data.
* `requirements.txt` adds `fastapi`, `uvicorn[standard]`,
  `python-multipart` (and the optional `numba` for full speed).
* `README.md` rewritten for v2 — covers the web UI, the CLI progress
  bar, the new `--reference` flag, and the performance comparison.

### Preserved
* All upstream algorithms — `deform.py`, `gcode_transform.py`,
  `slice_engine.py` — are kept verbatim and remain available with
  `--reference` on the CLI or by unchecking the "Fast backend"
  checkbox in the web UI. They produce the same gcode they always
  did and are used as the verification baseline.
* The CLI argument surface from v1 is preserved (the only addition
  is `--reference`, `--no-progress`).
* Output gcode format and headers are byte-compatible with the v1
  output.

### Verified
On the canonical `pi 3mm.stl` example (`--rot-iter 100 --deform-iter 500`):

|                              | total      | gcode-transform | speedup |
| ---------------------------- | ---------- | --------------- | ------- |
| reference backend            | 80.8 s     | ~38 s           | 1×      |
| fast backend (no JIT)        |  9.0 s     |   1.0 s         | ~9×     |
| fast backend + numba (cached)|  6.0 s     |   0.4 s         | **~13×** |

Both produce structurally-identical 4-axis polar gcode (same headers,
same number of motion lines within ±5 %, same overall path).
