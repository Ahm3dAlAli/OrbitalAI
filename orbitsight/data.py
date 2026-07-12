"""I/O layer: load event arrays / GT boxes, split into 40 ms windows, and
write predictions in the exact format the frozen ``evaluate.py`` expects.

Event array columns (N, 6):
    0 x, 1 y, 2 polarity, 3 timestamp_us, 4 label, 5 relative_timestamp_us
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

COL_X, COL_Y, COL_POL, COL_T, COL_LABEL, COL_RT = range(6)

PRED_HEADER = [
    "window_start_timestamp_us", "window_end_timestamp_us",
    "center_x", "center_y", "width", "height", "confidence",
]
GT_SUFFIX = "_bb_windows_40ms.txt"
EV_SUFFIX = "_labeled_events.npy"


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def load_events(path: str, mmap: bool = False) -> np.ndarray:
    """Load a labeled-events ``.npy`` as float64 (N, 6).

    ``mmap=True`` avoids reading the whole array into RAM (used for probing).
    """
    arr = np.load(path, mmap_mode="r" if mmap else None)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"{path}: expected (N,6) event array, got {arr.shape}")
    return arr


class Events:
    """Memory-compact column store for one sequence.

    Keeps only the columns the pipeline needs, in the smallest safe dtype:
    x/y as float32, polarity/label as int8, timestamps as int64 (microsecond
    absolute values exceed float32 precision, so they must stay integer).
    Cuts per-event footprint from 48 B (float64 ``(N,6)``) to ~14 B — important
    for the 32 GB / multi-GB-EVK4 budget (NFR-6).
    """

    __slots__ = ("x", "y", "pol", "t", "label", "n")

    def __init__(self, x, y, pol, t, label):
        self.x = x; self.y = y; self.pol = pol; self.t = t; self.label = label
        self.n = x.shape[0]

    @classmethod
    def from_npy(cls, path: str) -> "Events":
        arr = np.load(path, mmap_mode="r")          # do not pull all into RAM
        x = np.ascontiguousarray(arr[:, COL_X], dtype=np.float32)
        y = np.ascontiguousarray(arr[:, COL_Y], dtype=np.float32)
        pol = np.ascontiguousarray(arr[:, COL_POL]).astype(np.int8)
        t = np.ascontiguousarray(arr[:, COL_T]).astype(np.int64)
        if arr.shape[1] > COL_LABEL:
            label = np.ascontiguousarray(arr[:, COL_LABEL]).astype(np.int8)
        else:
            label = np.zeros(x.shape[0], dtype=np.int8)
        del arr
        return cls(x, y, pol, t, label)

    def head(self, n: int) -> "Events":
        return Events(self.x[:n], self.y[:n], self.pol[:n], self.t[:n], self.label[:n])


def find_event_file(data_dir: str, seq: str) -> Optional[str]:
    p = os.path.join(data_dir, seq + EV_SUFFIX)
    return p if os.path.exists(p) else None


def load_gt_boxes(path: str) -> list[tuple]:
    """Return list of (ws_us, we_us, cx, cy, w, h)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append((
                int(r["window_start_timestamp_us"]),
                int(r["window_end_timestamp_us"]),
                int(float(r["center_x"])), int(float(r["center_y"])),
                int(float(r["width"])),    int(float(r["height"])),
            ))
    return rows


# --------------------------------------------------------------------------- #
#  Windowing
# --------------------------------------------------------------------------- #
@dataclass
class Window:
    index: int
    start_us: int
    end_us: int
    lo: int          # event slice start (core window)
    hi: int          # event slice end   (core window)


def make_window_grid(ts: np.ndarray, window_us: int) -> list[Window]:
    """Split a (sorted) timestamp vector into consecutive 40 ms windows,
    anchored at the first event — the convention used by the official
    DataLoader snippet.
    """
    if ts.size == 0:
        return []
    t0, t1 = int(ts[0]), int(ts[-1])
    starts = np.arange(t0, t1 + window_us, window_us, dtype=np.int64)
    edges = np.searchsorted(ts, starts)
    wins = []
    for i in range(len(starts) - 1):
        wins.append(Window(i, int(starts[i]), int(starts[i] + window_us),
                            int(edges[i]), int(edges[i + 1])))
    # tail window (covers the final partial slab)
    if edges[-1] < ts.size:
        wins.append(Window(len(starts) - 1, int(starts[-1]),
                           int(starts[-1] + window_us), int(edges[-1]), int(ts.size)))
    return wins


def slice_with_halo(ts: np.ndarray, win: Window, halo_us: int) -> tuple[int, int]:
    """Event index range covering [start-halo, end+halo) for neighborhood
    feature computation without window-edge truncation."""
    lo = int(np.searchsorted(ts, win.start_us - halo_us, side="left"))
    hi = int(np.searchsorted(ts, win.end_us + halo_us, side="left"))
    return lo, hi


# --------------------------------------------------------------------------- #
#  Writing predictions
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    ws: int
    we: int
    cx: int
    cy: int
    w: int
    h: int
    conf: float


def write_predictions(path: str, dets: list[Detection]) -> None:
    """Write a tab-separated prediction file with the canonical header."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(PRED_HEADER)
        for d in dets:
            w.writerow([d.ws, d.we, int(d.cx), int(d.cy),
                        int(d.w), int(d.h), f"{d.conf:.4f}"])
