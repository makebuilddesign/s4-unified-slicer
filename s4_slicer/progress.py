"""Progress reporting infrastructure for the S4 slicer pipeline.

Provides a ProgressReporter that the deform / slice / gcode-transform stages
push percentage updates and log lines into.  The web UI reads from a queue
and forwards them to the browser over Server-Sent Events.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# Stage weights — how much of the total pipeline each stage takes.
# Tuned roughly from observed timings on a small example STL.
STAGE_WEIGHTS = {
    "load":     0.05,
    "tet":      0.05,
    "rotation": 0.20,
    "deform":   0.30,
    "slice":    0.10,
    "transform":0.30,
}

STAGE_LABELS = {
    "load":      "Loading STL",
    "tet":       "Tetrahedralising",
    "rotation":  "Optimising rotation field",
    "deform":    "Deforming mesh",
    "slice":     "Slicing deformed mesh",
    "transform": "Transforming G-code",
}


@dataclass
class ProgressReporter:
    """Thread-safe progress + log queue for the slicing pipeline."""

    job_id: str = ""
    queue: "queue.Queue" = field(default_factory=queue.Queue)
    _stage: str = ""
    _stage_progress: float = 0.0
    _global_progress: float = 0.0
    _start_time: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _done: bool = False
    _failed: bool = False
    _error: Optional[str] = None

    # ------------------------------------------------------------------
    def stage(self, stage_key: str, sub_progress: float = 0.0):
        """Mark current stage and sub-progress (0..1 within the stage)."""
        with self._lock:
            self._stage = stage_key
            self._stage_progress = max(0.0, min(1.0, sub_progress))
            self._recalc_global()
        self._emit("progress")

    def update(self, sub_progress: float):
        """Update sub-progress within the current stage."""
        with self._lock:
            self._stage_progress = max(0.0, min(1.0, sub_progress))
            self._recalc_global()
        self._emit("progress")

    def log(self, message: str, level: str = "info"):
        msg = str(message).rstrip()
        if not msg:
            return
        self.queue.put({
            "type":  "log",
            "level": level,
            "message": msg,
            "ts":    time.time() - self._start_time,
        })

    def error(self, message: str):
        with self._lock:
            self._failed = True
            self._error  = str(message)
        self.queue.put({
            "type":    "error",
            "message": str(message),
            "ts":      time.time() - self._start_time,
        })

    def done(self, output_path: str = "", deformed_stl: str = "",
             input_stl: str = "", extras: dict | None = None):
        with self._lock:
            self._done = True
            self._global_progress = 1.0
            self._stage_progress  = 1.0
            self._stage           = "done"
        evt = {
            "type":         "done",
            "output_path":  output_path,
            "deformed_stl": deformed_stl,
            "input_stl":    input_stl,
            "elapsed":      time.time() - self._start_time,
        }
        if extras:
            evt.update(extras)
        self.queue.put(evt)

    # ------------------------------------------------------------------
    def _recalc_global(self):
        # Sum complete stages + portion of current.
        order  = list(STAGE_WEIGHTS.keys())
        if self._stage not in STAGE_WEIGHTS:
            return
        idx    = order.index(self._stage)
        before = sum(STAGE_WEIGHTS[s] for s in order[:idx])
        cur    = STAGE_WEIGHTS[self._stage] * self._stage_progress
        self._global_progress = max(0.0, min(0.999, before + cur))

    def _emit(self, kind: str):
        self.queue.put({
            "type":            kind,
            "stage":           self._stage,
            "stage_label":     STAGE_LABELS.get(self._stage, self._stage),
            "stage_progress":  self._stage_progress,
            "global_progress": self._global_progress,
            "ts":              time.time() - self._start_time,
        })

    # ------------------------------------------------------------------
    @property
    def global_progress(self) -> float:
        return self._global_progress


# A no-op reporter so the optimised pipeline modules can be used as
# library calls without having to construct one.
class NullProgress:
    job_id = "null"
    def stage(self, *a, **k):  pass
    def update(self, *a, **k): pass
    def log(self, *a, **k):    pass
    def error(self, *a, **k):  pass
    def done(self, *a, **k):   pass
    @property
    def global_progress(self): return 0.0
