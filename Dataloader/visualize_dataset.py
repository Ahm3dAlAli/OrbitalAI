#!/usr/bin/env python3
"""
OrbitSight DataLoader — Visualize neuromorphic event stream data with annotated detection of resident space objects (RSOs). This script processes the provided dataset, rendering raw event frames and detection frames with bounding boxes, as well as plotting event statistics.
Input (Dataset_GT/):
  *_labeled_events.npy   — event stream columns: [x, y, polarity, timestamp_us, label, relative_timestamp_us]
  *_bb_windows_40ms.txt  — bounding boxes columns: [window_start_us, window_end_us, center_x, center_y, width, height]

Outputs (output/<sequence>/):
  Raw_Event_Frame/    — one PNG per 40ms window; blue = negative polarity events, red = positive polarity events
  Detection_Frame/    — Event frame + Green bounding box for detected objects in that window (if any)
  Events_statistics/  — line plot of event count per 40ms window
"""

import csv
import os

import cv2
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm


# ── Select one sequence: uncomment the one you want to run ───────────────────
# Files are resolved automatically from Training_sets/ and Testing_sets/.

DATASET_DIR = "../OrbitSight_Dataset/"

# --- Training sequences ---
# SEQ = "2025_12_23_21_12_28_EVK4_mag5.2"                          # ← active
# SEQ = "DAVIS_COSMOS1933_18958_2024-12-04-18-37-01"
# SEQ = "DAVIS_EGS_16908_2024-11-01-19-10-44"
# SEQ = "DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06"
# SEQ = "DAVIS_RESURSDK1_29228_2024-12-04-18-37-01"
# SEQ = "DAVIS_SL12RB2_15772_2024-12-04-18-21-37"
# SEQ = "DAVIS_SL16RB_20625_2024-12-04-19-34-18"
# SEQ = "DAVIS_SL16RB_26070_2024-12-04-19-14-39"
SEQ = "DAVIS_SL8RB_2025-01-13-19-15-36"
# SEQ = "DVX_Filtered_ACS3_59588_2025-01-20-19-35-44"
# SEQ = "DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17"
# SEQ = "DVX_Filtered_NOAA15_25338_2025-01-20-19-25-07"
# SEQ = "DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50"
# SEQ = "DVX_Filtered_NOAA6_11416_2025-01-20-19-11-35"
# SEQ = "DVX_Filtered_Stars_2025-01-20-19-15-10"
# SEQ = "DVX_Filtered_Stars2_2025-01-20-19-57-17"
# SEQ = "DVX_NOAA6_11416_2025-01-20-19-06-31"

# --- Testing sequences ---
# SEQ = "2025_12_23_20_53_46_EVK4_mag7.3"
# SEQ = "DAVIS_SAOCOM1B_46265_2024-12-04-18-21-37"
# SEQ = "DVX_Filtered_Stars3_2025-01-20-20-22-53"
# SEQ = "DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43"

# ─────────────────────────────────────────────────────────────────────────────


def _find_file(seq, suffix):
    """Search Training_sets/ then Testing_sets/ for a sequence file."""
    for subdir in ("Training_sets", "Testing_sets"):
        path = os.path.join(DATASET_DIR, subdir, f"{seq}{suffix}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find '{seq}{suffix}' in Training_sets/ or Testing_sets/"
    )

WINDOW_US  = 40_000   # 40 ms time window
OUTPUT_ROOT = "output"


def load_events(npy_path):
    data = np.load(npy_path)
    if data.shape[1] < 6:
        rel = data[:, 3] - data[0, 3]
        data = np.column_stack([data, rel])
    return data  # [x, y, polarity, timestamp_us, label, relative_timestamp_us]


def load_bb(txt_path):
    rows = []
    with open(txt_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append((
                int(row["window_start_timestamp_us"]),
                int(row["window_end_timestamp_us"]),
                int(row["center_x"]),
                int(row["center_y"]),
                int(row["width"]),
                int(row["height"]),
            ))
    return rows  # list of (start_us, end_us, cx, cy, w, h)


def render_raw_event_frame(chunk, sensor_width, sensor_height):
    frame = np.zeros((sensor_height, sensor_width, 3), dtype=np.uint8)
    if len(chunk) == 0:
        return frame
    xs  = chunk[:, 0].astype(int)
    ys  = chunk[:, 1].astype(int)
    pol = chunk[:, 2].astype(int)
    valid = (xs >= 0) & (xs < sensor_width) & (ys >= 0) & (ys < sensor_height)
    xs, ys, pol = xs[valid], ys[valid], pol[valid]
    frame[ys[pol == 0], xs[pol == 0]] = (255, 0,   0)  # blue  — negative
    frame[ys[pol == 1], xs[pol == 1]] = (0,   0, 255)  # red   — positive
    return frame


def render_detection_frame(chunk, sensor_width, sensor_height, bb=None):
    frame = np.zeros((sensor_height, sensor_width, 3), dtype=np.uint8)
    if len(chunk) > 0:
        xs = chunk[:, 0].astype(int)
        ys = chunk[:, 1].astype(int)
        valid = (xs >= 0) & (xs < sensor_width) & (ys >= 0) & (ys < sensor_height)
        frame[ys[valid], xs[valid]] = (255, 255, 255)  # white — all events
    if bb is not None:
        cx, cy, w, h = bb
        x1 = int(cx - (w - 1) // 2)
        y1 = int(cy - (h - 1) // 2)
        x2 = x1 + w - 1
        y2 = y1 + h - 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame


def save_events_statistics(window_starts, event_counts, out_path):
    t_sec = (np.array(window_starts) - window_starts[0]) * 1e-6  # microseconds → seconds

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_sec, y=event_counts,
        mode="lines", name="Event count",
        line=dict(width=2)
    ))
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Time (s)", font=dict(family="Times New Roman", size=18)),
            tickfont=dict(size=18, family="Times New Roman"),
            mirror=True, ticks="outside", showline=True,
            linecolor="black", gridcolor="lightgrey"
        ),
        yaxis=dict(
            title=dict(text="Event Count", font=dict(family="Times New Roman", size=18)),
            tickfont=dict(size=18, family="Times New Roman"),
            mirror=True, ticks="outside", showline=True,
            linecolor="black", gridcolor="lightgrey"
        ),
        legend=dict(font=dict(family="Times New Roman", size=14)),
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor="white"
    )
    fig.write_image(out_path, format="png", width=1200, height=400, scale=3)
    print(f"[INFO] Statistics plot saved: {out_path}")


def main():
    npy_path = _find_file(SEQ, "_labeled_events.npy")
    bb_path  = _find_file(SEQ, "_bb_windows_40ms.txt")

    print(f"[INFO] Sequence  : {SEQ}")
    print(f"[INFO] Events    : {npy_path}")
    print(f"[INFO] Boxes     : {bb_path}")

    events = load_events(npy_path)
    print(f"[INFO] Loaded {len(events):,} events  shape={events.shape}")

    bb_list   = load_bb(bb_path)
    bb_starts = np.array([b[0] for b in bb_list], dtype=np.int64)
    bb_ends   = np.array([b[1] for b in bb_list], dtype=np.int64)
    print(f"[INFO] Loaded {len(bb_list):,} bounding box windows")

    sensor_width  = int(events[:, 0].max() + 1)
    sensor_height = int(events[:, 1].max() + 1)
    print(f"[INFO] Sensor size: {sensor_width} x {sensor_height}")

    npy_base = os.path.splitext(os.path.basename(npy_path))[0]
    seq_out  = os.path.join(OUTPUT_ROOT, npy_base)
    raw_dir  = os.path.join(seq_out, "Raw_Event_Frame")
    det_dir  = os.path.join(seq_out, "Detection_Frame")
    for d in [seq_out, raw_dir, det_dir]:
        os.makedirs(d, exist_ok=True)

    ev_t          = events[:, 3]
    t0            = int(ev_t[0])
    t1            = int(ev_t[-1])
    window_starts = np.arange(t0, t1, WINDOW_US, dtype=np.int64)
    break_idx     = np.searchsorted(ev_t, window_starts)

    event_counts = []
    det_saved    = 0

    print(f"\n[INFO] Rendering {len(window_starts)} frames ...")
    for i in tqdm(range(len(window_starts)), unit="frame"):
        i0    = break_idx[i]
        i1    = break_idx[i + 1] if i + 1 < len(break_idx) else len(events)
        chunk = events[i0:i1]
        ws    = int(window_starts[i])
        event_counts.append(len(chunk))

        raw_frame = render_raw_event_frame(chunk, sensor_width, sensor_height)
        cv2.imwrite(os.path.join(raw_dir, f"frame_{i:05d}_{ws}.png"), raw_frame)

        ws_end = ws + WINDOW_US
        hits = np.where((bb_starts < ws_end) & (bb_ends > ws))[0]
        bb = (bb_list[hits[0]][2], bb_list[hits[0]][3], bb_list[hits[0]][4], bb_list[hits[0]][5]) if len(hits) else None
        if bb is not None:
            det_frame = render_detection_frame(chunk, sensor_width, sensor_height, bb)
            cv2.imwrite(os.path.join(det_dir, f"{SEQ}_Event_Frame_{i}.png"), det_frame)
            det_saved += 1

    stat_path = os.path.join(seq_out, "Event_Statistics.png")
    save_events_statistics(list(window_starts), event_counts, stat_path)

    print(f"\n[INFO] Done. Output saved to: {seq_out}")
    print(f"  Raw_Event_Frame/   — {len(window_starts)} frames")
    print(f"  Detection_Frame/   — {det_saved} frames (BB only)")
    print(f"  Event_Statistics.png")

# 
if __name__ == "__main__":
    main()
