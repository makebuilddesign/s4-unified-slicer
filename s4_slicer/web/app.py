"""FastAPI web UI for the S4 Unified Slicer.

Run:
    s4slicer-web                 # auto-port 8765
    s4slicer-web --port 8000

Endpoints:
    GET  /                       — single-page UI
    POST /api/upload             — upload STL, returns job_id
    POST /api/start/{job_id}     — start the slicing pipeline (params in body)
    GET  /api/stream/{job_id}    — SSE stream of progress + log + done events
    GET  /api/preview/{job_id}/path     — JSON path-preview data (for output)
    GET  /api/download/{job_id}/gcode   — download final 4-axis g-code
    GET  /api/download/{job_id}/stl     — download deformed STL
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException, Request
    from fastapi.responses import (HTMLResponse, JSONResponse, FileResponse,
                                   StreamingResponse)
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError as e:
    raise SystemExit(
        "FastAPI / uvicorn not installed.  Install with:\n"
        "    pip install fastapi 'uvicorn[standard]' python-multipart"
    )

from ..pipeline import PipelineConfig, run_pipeline
from ..progress import ProgressReporter

# ---------------------------------------------------------------------------
HERE       = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
TPL_DIR    = HERE / "templates"


class Job:
    def __init__(self, job_id: str, stl_path: Path, workdir: Path,
                 original_name: str):
        self.id            = job_id
        self.stl_path      = stl_path
        self.workdir       = workdir
        self.original_name = original_name
        self.progress      = ProgressReporter(job_id=job_id)
        self.thread: Optional[threading.Thread] = None
        self.result: dict  = {}
        self.failed_msg    = ""
        self.cfg: Optional[PipelineConfig] = None
        # Path-preview data (filled when transform stage finishes).
        self.path_preview: Optional[dict] = None


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
def _build_path_preview(out_gcode: Path, polar: bool) -> dict:
    """Parse the output 4-axis g-code into a downsampled poly-line for the UI."""
    if not out_gcode.exists():
        return {"points": [], "extruding": [], "polar": polar, "n_lines": 0}

    pts        = []
    extruding  = []
    rotations  = []
    travel     = []
    prev_th    = 0.0
    n_lines    = 0
    with open(out_gcode) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith(";") or not ln.startswith(("G0", "G1", "G00", "G01")):
                continue
            n_lines += 1
            tokens = ln.split()
            d = {}
            for tok in tokens[1:]:
                if not tok or tok[0].isdigit() or tok[0] == "-":
                    continue
                d[tok[0]] = tok[1:]
            try:
                b = float(d.get("B", "0"))
                rot = np.deg2rad(b)
                # nozzle_offset is usually around 42mm in this project
                n_off = 42.0 
                
                if polar:
                    c = float(d.get("C", "0"))
                    r_machine = float(d.get("X", "0"))
                    z_machine = float(d.get("Z", "0"))
                    
                    r = r_machine
                    z = z_machine
                    
                    th = np.deg2rad(c)
                    x  = r * np.cos(th)
                    y  = r * np.sin(th)
                else:
                    x = float(d.get("X", "0"))
                    y = float(d.get("Y", "0"))
                    z = float(d.get("Z", "0"))
                
                e = float(d.get("E", "0")) if "E" in d else 0.0
            except ValueError:
                continue
            pts.append([x, y, z])
            extruding.append(bool(e > 0))
            rotations.append(b)

    pts = np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 3))
    extruding = np.asarray(extruding, dtype=bool)
    rotations = np.asarray(rotations, dtype=np.float32) if rotations else np.zeros(0)
    # Down-sample if huge — keep ≤ 60k points for browser performance.
    MAX = 60000
    if len(pts) > MAX:
        step = int(np.ceil(len(pts) / MAX))
        pts        = pts[::step]
        extruding  = extruding[::step]
        rotations  = rotations[::step]

    return {
        "points":    pts.tolist(),
        "extruding": extruding.tolist(),
        "rotation":  rotations.tolist(),
        "polar":     polar,
        "n_lines":   n_lines,
        "downsampled_to": len(pts),
    }


def _stl_preview(stl_path: Path) -> dict:
    """Parse STL into a small JSON for client-side preview (vertices only)."""
    try:
        import trimesh
        print(f"[web] loading STL for preview: {stl_path}")
        m = trimesh.load(str(stl_path), force="mesh")
        v = np.asarray(m.vertices, dtype=np.float64)
        f = np.asarray(m.faces, dtype=np.int64)
        print(f"[web] STL loaded: {len(v)} verts, {len(f)} faces")
        # Cap output to keep transfer reasonable.
        if len(f) > 60000:
            # decimate by random sub-sampling of faces
            sel = np.random.default_rng(0).choice(len(f), 60000, replace=False)
            f = f[sel]
        return {
            "vertices": v.tolist(),
            "faces":    f.tolist(),
            "bounds":   m.bounds.tolist(),
        }
    except Exception as e:
        print(f"[web] STL preview failed: {e}")
        return {"vertices": [], "faces": [], "error": str(e)}


# ---------------------------------------------------------------------------
def _runner(job: Job):
    try:
        job.progress.log("[runner] pipeline starting")
        result = run_pipeline(str(job.stl_path), job.cfg, progress=job.progress)
        job.result = result
        # Build path preview from the final gcode.
        out_path = Path(result["output_gcode"])
        try:
            job.path_preview = _build_path_preview(out_path, polar=True)
        except Exception as e:
            job.progress.log(f"[runner] path-preview build failed: {e}", level="warn")

        job.progress.done(
            output_path=str(out_path),
            deformed_stl=result.get("deformed_stl", ""),
            input_stl=str(job.stl_path),
            extras={"stats": result.get("stats", {}),
                    "elapsed": result.get("elapsed", 0.0)},
        )
    except Exception as e:
        tb = traceback.format_exc(limit=8)
        job.failed_msg = f"{type(e).__name__}: {e}"
        job.progress.log(tb, level="error")
        job.progress.error(job.failed_msg)


# ---------------------------------------------------------------------------
def make_app() -> FastAPI:
    app = FastAPI(title="S4 Unified Slicer (Web UI)")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ----- index page ------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index():
        idx = TPL_DIR / "index.html"
        if not idx.exists():
            raise HTTPException(500, f"missing template: {idx}")
        return idx.read_text()

    # ----- upload ----------------------------------------------------------
    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        if not file.filename.lower().endswith((".stl",)):
            raise HTTPException(400, "Only .stl files are accepted")
        job_id = uuid.uuid4().hex[:12]
        wd = Path(tempfile.mkdtemp(prefix=f"s4web_{job_id}_"))
        stl_path = wd / file.filename
        with open(stl_path, "wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        job = Job(job_id=job_id, stl_path=stl_path, workdir=wd,
                  original_name=file.filename)
        with JOBS_LOCK:
            JOBS[job_id] = job
        # Generate input STL preview now (cheap).
        print(f"[web] generating preview for {file.filename}")
        prev = _stl_preview(stl_path)
        print(f"[web] upload complete for {job_id}")
        return {"job_id": job_id, "filename": file.filename,
                "size":   stl_path.stat().st_size,
                "preview": prev}

    # ----- start -----------------------------------------------------------
    @app.post("/api/start/{job_id}")
    async def start(job_id: str, request: Request):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job_id")
        if job.thread and job.thread.is_alive():
            raise HTTPException(409, "Already running")
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        cfg = PipelineConfig()
        # apply any provided params (only known fields)
        for k, v in (body or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.workdir            = str(job.workdir)
        cfg.keep_intermediates = True
        cfg.output             = str(job.workdir / (job.stl_path.stem + ".gcode"))
        job.cfg = cfg
        print(f"[web] starting job {job_id} with cfg: {job.cfg}")
        job.thread = threading.Thread(target=_runner, args=(job,), daemon=True)
        job.thread.start()
        return {"ok": True, "job_id": job_id, "config": asdict(cfg)}

    # ----- stream (SSE) ----------------------------------------------------
    @app.get("/api/stream/{job_id}")
    async def stream(job_id: str, request: Request):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job_id")

        async def gen():
            yield f"event: hello\ndata: {json.dumps({'job_id': job_id})}\n\n"
            q = job.progress.queue
            while True:
                if await request.is_disconnected():
                    break
                try:
                    evt = q.get(timeout=0.25)
                except queue.Empty:
                    # Heartbeat to keep the connection alive.
                    yield f": ping {time.time():.1f}\n\n"
                    continue
                yield f"data: {json.dumps(evt)}\n\n"
                if evt.get("type") in ("done", "error"):
                    break

        return StreamingResponse(gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"})

    # ----- preview ---------------------------------------------------------
    @app.get("/api/preview/{job_id}/path")
    def preview_path(job_id: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job_id")
        if not job.path_preview:
            raise HTTPException(425, "Path preview not yet ready")
        return JSONResponse(job.path_preview)

    @app.get("/api/preview/{job_id}/input")
    def preview_input(job_id: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job_id")
        return JSONResponse(_stl_preview(job.stl_path))

    @app.get("/api/preview/{job_id}/deformed")
    def preview_deformed(job_id: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job_id")
        d = job.result.get("deformed_stl") if job.result else None
        if not d or not Path(d).exists():
            raise HTTPException(425, "Deformed STL not yet available")
        return JSONResponse(_stl_preview(Path(d)))

    # ----- download --------------------------------------------------------
    @app.get("/api/download/{job_id}/gcode")
    def download_gcode(job_id: str):
        """Download the raw polar (C,X,Z,B) G-code for the actual machine."""
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job_id")
        out = job.result.get("output_gcode") if job.result else None
        if not out or not Path(out).exists():
            raise HTTPException(425, "Output not yet available")
        return FileResponse(out, filename=Path(out).name,
                            media_type="text/plain")

    @app.get("/api/download/{job_id}/stl")
    def download_stl(job_id: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job_id")
        d = job.result.get("deformed_stl") if job.result else None
        if not d or not Path(d).exists():
            raise HTTPException(425, "Deformed STL not yet available")
        return FileResponse(d, filename=Path(d).name,
                            media_type="application/sla")

    @app.get("/api/status/{job_id}")
    def status(job_id: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job_id")
        return {
            "job_id":  job.id,
            "filename": job.original_name,
            "running": bool(job.thread and job.thread.is_alive()),
            "elapsed": job.result.get("elapsed") if job.result else None,
            "global_progress": job.progress.global_progress,
            "error":   job.failed_msg,
        }

    return app


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="s4slicer-web",
                                 description="Web UI for S4 Unified Slicer")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", default=8765, type=int)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args(argv)

    app = make_app()
    print(f"\n  S4 Unified Slicer — Web UI\n  http://{args.host}:{args.port}/\n")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
