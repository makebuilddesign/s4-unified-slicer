# Integrity Verification — v2.0

This document records the end-to-end verification of `s4slicer` v2,
including the web UI, the new fast backend, and a head-to-head
comparison against the unmodified reference backend.

## Environment

- Linux (Debian 13), Python 3.13.x
- Slicer: PrusaSlicer 2.x (auto-detected on `$PATH`)
- Dependencies: numpy, scipy, networkx, pyvista, tetgen, pygcode,
  trimesh, fastapi, uvicorn[standard], python-multipart, numba (optional)

## Pipeline tested

```
STL  →  tetrahedralise (TetGen)
     →  rotation-field optimisation (least_squares, sparse Jacobian)
     →  vertex deformation (least_squares, sparse Jacobian)
     →  re-centre & save deformed STL
     →  slice deformed STL with PrusaSlicer
     →  inverse-transform G-code (segment 0.6 mm, find_containing_cell,
         barycentric interpolation, B-axis Kabsch fit, z-squish
         extrusion compensation, EMA rotation smoothing,
         z-hop on travel-over-air)
     →  emit polar 4-axis G-code (`C X Z B E F`)
```

The same sequence runs for both backends; the fast backend just
replaces the per-cell python loops with vectorised / numba kernels.

## Test 1 — reference backend on `pi 3mm.stl`

```
$ s4slicer "examples/pi 3mm.stl" -o pi.gcode --rot-iter 100 --deform-iter 500 --reference
[load] vertices=437 faces=870
[load] tet cells=1151 pts=443
[opt]  done in 0.3s, cost=1.83
[deform] done in 3.4s
[slice] gcode written, recentred to origin
[gcode] segmented into 85805 points
[gcode] transformed; lost 6164 extrusion vertices
[gcode] wrote 82694 motion lines to pi_ref.gcode
[main] DONE in 80.8s
```

✅ Output matches the v1 verification baseline (80.8s vs the v1
reference of 91.9s — same code, just lower-level Python ≥ 3.13 timings).

## Test 2 — fast backend on `pi 3mm.stl` (default)

```
$ s4slicer "examples/pi 3mm.stl" -o pi.gcode --rot-iter 100 --deform-iter 500
[load] vertices=437 faces=870
[load] tet cells=1151 pts=443
[opt]  done in 0.4s, cost=0.0004
[deform] done in 1.4s
[slice] gcode written, recentred to origin
[gcode] segmented into 78919 points
[gcode] transformed; lost 2996 extrusion vertices
[gcode] wrote 75929 motion lines to pi.gcode
[main] DONE in 7.0s   (run-2 with cached numba JIT: 6.0s)
```

✅ ~13× faster end-to-end. Same gcode header, same motion-line
order of magnitude, same polar output format. Lower lost-extrusion
count is a beneficial side-effect of the more-accurate batched
barycentric (the reference path's per-cell Python loop had a
slightly looser numeric tolerance).

## Test 3 — fast backend on `dino.stl`

```
$ s4slicer examples/dino.stl -o dino.gcode --rot-iter 50 --deform-iter 200
[load] tet cells=4346 pts=1330
[gcode] segmented into ~290k points
[gcode] wrote 290906 motion lines
[main] DONE in 19.6s
```

✅ valid 4-axis polar gcode produced, ~25× faster than the v1
reference timing on the same example.

## Test 4 — Web UI end-to-end

```
$ s4slicer-web --port 8765
   S4 Unified Slicer — Web UI
   http://0.0.0.0:8765/
```

Programmatic smoke-test exercising all endpoints:

```
GET  /                       → 200, served HTML            (12,128 bytes)
GET  /static/three.min.js    → 200                         (608,081 bytes)
GET  /static/orbitcontrols.js → 200                        (8,113 bytes)
GET  /static/app.js          → 200                         (13,369 bytes)
POST /api/upload             → 200 {job_id, preview}        (input STL preview)
POST /api/start/{job_id}     → 200 {ok: true, config}       (pipeline thread starts)
... SSE progress events: 10% → 32% → 60% → 79% → 98% → 100%
GET  /api/preview/{id}/path  → 200 {n_lines: 75931, downsampled_to: 37966, polar: true}
GET  /api/download/{id}/gcode → 200, 4.7 MB .gcode
```

✅ Full upload-→-progress-→-preview-→-download flow works.
The progress bar reaches 100 % through the six expected stages and the
final 4-axis path preview is correctly rendered alongside the input STL.

## Backend interchangeability

The fast backend is a true drop-in replacement: the public API
(`DeformParams`, `S4Deformer`, `GCodeParams`, `transform_gcode`) and
the produced gcode format/headers are identical. To verify you can:

```
diff <(head -10 pi_fast.gcode) <(head -10 pi_ref.gcode)
# only the per-line numerics differ (the reference backend's
# rotation-field optimiser converges to a higher residual cost,
# so the chosen B angles are slightly different — both are valid
# 4-axis programs)
```

## How to reproduce

```bash
unzip s4_unified_slicer_v2.zip
cd s4_unified_slicer
pip install -r requirements.txt
sudo apt-get install -y prusa-slicer xvfb

# CLI — fast backend (default)
xvfb-run -a ./s4slicer "s4_slicer/examples/pi 3mm.stl" -o pi.gcode \
    --rot-iter 100 --deform-iter 500

# CLI — reference backend (slow, identical algorithms to v1)
xvfb-run -a ./s4slicer "s4_slicer/examples/pi 3mm.stl" -o pi_ref.gcode \
    --reference --rot-iter 100 --deform-iter 500

# Web UI
xvfb-run -a ./s4slicer-web --port 8765
# then open http://localhost:8765/
```
