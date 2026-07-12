# OrbitSight DataLoader

A starter toolkit for loading and visualizing the **OrbitSight Neuromorphic Challenge for Space Objects detection dataset**.  
Use this as your entry point to understand the data format, explore the sequences, and start building your detection algorithm.

---

## Overview

The **TII OrbitSight Challenge** invites solution providers, including startups, students, researchers, academic institutions, and established enterprises, to propose innovative solutions that address a real-world, operational Space Situational Awareness (SSA) challenge.

Participants are asked to develop a detection algorithm for Resident Space Objects (RSOs), i.e. satellites, rocket bodies, and debris, using event-based neuromorphic camera data. Unlike conventional frame-based imaging, event cameras capture per-pixel brightness changes asynchronously with microsecond-level temporal resolution, making them well-suited for tracking fast-moving objects against a star-field background.

The dataset was captured using three event-based sensors:

| Sensor | Type | Resolution |
|--------|------|------------|
| **DAVIS** | Dynamic and Active-pixel Vision Sensor - outputs both events and frames | 346 × 260 |
| **DVX** | Pure event-based camera | 640 × 480 |
| **EVK4** | Prophesee Metavision EVK4 - high-resolution pure event-based camera | 1280 × 720 |

Each sequence is provided as a labeled event stream (`.npy`) paired with ground-truth bounding boxes (`.txt`) at a 40 ms temporal resolution. See the [Submission requirements](#submission-requirements) section for full details on what to submit.

---

## Dataset structure

The dataset lives in `OrbitSight_Dataset/`. Each sequence has two files:

| File | Description |
|------|-------------|
| `<sequence>_labeled_events.npy` | Event stream with ground truth labels |
| `<sequence>_bb_windows_40ms.txt` | Bounding box ground truth per 40 ms window |

### Event stream - `.npy`

Each `.npy` file is a NumPy array of shape `(N, 6)`:

| Column | Name | Description |
|--------|------|-------------|
| 0 | `x` | Pixel column (integer) |
| 1 | `y` | Pixel row (integer) |
| 2 | `polarity` | 0 = negative (brightness decrease), 1 = positive (brightness increase) |
| 3 | `timestamp_us` | Absolute timestamp in microseconds |
| 4 | `label` | 0 = background, 1 = RSO Object |
| 5 | `relative_timestamp_us` | Timestamp relative to the first event in the sequence |

```python
import numpy as np
events = np.load("OrbitSight_Dataset/<sequence>_labeled_events.npy")
# events.shape → (N, 6)
```

### Bounding boxes - `.txt`

Tab-separated file with a header row. One row per 40 ms time window that contains a detection.  
All coordinates are in **pixels**.

| Column | Description |
|--------|-------------|
| `window_start_timestamp_us` | Start of the time window (µs) |
| `window_end_timestamp_us` | End of the time window (µs) |
| `center_x` | Bounding box center X |
| `center_y` | Bounding box center Y |
| `width` | Bounding box width in pixels |
| `height` | Bounding box height in pixels |

```python
import csv
with open("OrbitSight_Dataset/<sequence>_bb_windows_40ms.txt") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        print(row)
```

---

## Available sequences

### Training sequences

Ground truth labels (`label == 1`) and bounding box `.txt` files are provided for all training sequences.

| # | Sequence name | Sensor |
|---|--------------|--------|
| 1 | 2025_12_23_21_12_28_EVK4_mag5.2 | EVK4 |
| 2 | DAVIS_COSMOS1933_18958_2024-12-04-18-37-01 | DAVIS |
| 3 | DAVIS_EGS_16908_2024-11-01-19-10-44 | DAVIS |
| 4 | DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06 | DAVIS |
| 5 | DAVIS_RESURSDK1_29228_2024-12-04-18-37-01 | DAVIS |
| 6 | DAVIS_SL12RB2_15772_2024-12-04-18-21-37 | DAVIS |
| 7 | DAVIS_SL16RB_20625_2024-12-04-19-34-18 | DAVIS |
| 8 | DAVIS_SL16RB_26070_2024-12-04-19-14-39 | DAVIS |
| 9 | DAVIS_SL8RB_2025-01-13-19-15-36 | DAVIS |
| 10 | DVX_Filtered_ACS3_59588_2025-01-20-19-35-44 | DVX |
| 11 | DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17 | DVX |
| 12 | DVX_Filtered_NOAA15_25338_2025-01-20-19-25-07 | DVX |
| 13 | DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50 | DVX |
| 14 | DVX_Filtered_NOAA6_11416_2025-01-20-19-11-35 | DVX |
| 15 | DVX_Filtered_Stars_2025-01-20-19-15-10 | DVX |
| 16 | DVX_Filtered_Stars2_2025-01-20-19-57-17 | DVX |
| 17 | DVX_NOAA6_11416_2025-01-20-19-06-31 | DVX |

### Testing sequences

Event streams (`.npy`) and bounding box `.txt` files are provided with ground truth labels to allow participants to calculate their detection score.

| # | Sequence name | Sensor |
|---|--------------|--------|
| 1 | 2025_12_23_20_53_46_EVK4_mag7.3 | EVK4 |
| 2 | DAVIS_SAOCOM1B_46265_2024-12-04-18-21-37 | DAVIS |
| 3 | DVX_Filtered_Stars3_2025-01-20-20-22-53 | DVX |
| 4 | DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43 | DVX |

---

## Running the visualizer

### 1. Install dependencies

```bash
pip install numpy opencv-python plotly kaleido tqdm tabulate openpyxl
```

### 2. Select a sequence

Open `visualize_dataset.py` and uncomment the sequence you want to run.  
Only one `SEQ = ...` line should be active at a time:

```python
SEQ = "DAVIS_SL12RB2_15772_2024-12-04-18-21-37"   # ← active
# SEQ = "DAVIS_EGS_16908_2024-11-01-19-10-44"
# SEQ = "DVX_NOAA6_11416_2025-01-20-19-06-31"
# ...
```

### 3. Run

```bash
cd OrbitSight_DataLoader
python3 visualize_dataset.py
```

---

## Outputs

Results are saved under `output/<sequence>_labeled_events/`:

```
output/
└── <sequence>_labeled_events/
    ├── Raw_Event_Frame/              ← one PNG per 40ms window
    │   ├── frame_00000_<ts>.png      │   blue pixels  = negative polarity events
    │   ├── frame_00001_<ts>.png      │   red pixels   = positive polarity events
    │   └── ...
    ├── Detection_Frame/              ← one PNG per window that has a GT bounding box
    │   ├── <sequence>_Event_Frame_0.png   │   white pixels = all events
    │   ├── <sequence>_Event_Frame_1.png   │   green rectangle = GT bounding box
    │   └── ...
    └── Event_Statistics.png          ← event count per 40ms window over time
```

---

## Next steps - building a detection algorithm

A simple starting workflow:

```python
import numpy as np

events = np.load("OrbitSight_Dataset/<sequence>_labeled_events.npy")

# Split into 40 ms windows
WINDOW_US = 40_000
ev_t = events[:, 3]
window_starts = np.arange(ev_t[0], ev_t[-1], WINDOW_US)
break_idx = np.searchsorted(ev_t, window_starts)

for i in range(len(window_starts) - 1):
    window = events[break_idx[i]:break_idx[i+1]]
    x, y, polarity, ts, label, rel_ts = window.T

    # --- your detection algorithm here ---
    # Use (x, y, polarity) as input features
    # label == 1 marks ground truth satellite events (training only)
```

The bounding box `.txt` file gives you the GT box per window to train and validate against.

---

## Submission format

For each training and testing sequence, produce one prediction file named `<sequence>_pred.txt`:

```
predictions/
│   # — Training sequences (17 files) —
├── 2025_12_23_21_12_28_EVK4_mag5.2_pred.txt
├── DAVIS_COSMOS1933_18958_2024-12-04-18-37-01_pred.txt
├── DAVIS_EGS_16908_2024-11-01-19-10-44_pred.txt
├── DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06_pred.txt
├── DAVIS_RESURSDK1_29228_2024-12-04-18-37-01_pred.txt
├── DAVIS_SL12RB2_15772_2024-12-04-18-21-37_pred.txt
├── DAVIS_SL16RB_20625_2024-12-04-19-34-18_pred.txt
├── DAVIS_SL16RB_26070_2024-12-04-19-14-39_pred.txt
├── DAVIS_SL8RB_2025-01-13-19-15-36_pred.txt
├── DVX_Filtered_ACS3_59588_2025-01-20-19-35-44_pred.txt
├── DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17_pred.txt
├── DVX_Filtered_NOAA15_25338_2025-01-20-19-25-07_pred.txt
├── DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50_pred.txt
├── DVX_Filtered_NOAA6_11416_2025-01-20-19-11-35_pred.txt
├── DVX_Filtered_Stars_2025-01-20-19-15-10_pred.txt
├── DVX_Filtered_Stars2_2025-01-20-19-57-17_pred.txt
├── DVX_NOAA6_11416_2025-01-20-19-06-31_pred.txt
│   # — Testing sequences (4 files) —
├── 2025_12_23_20_53_46_EVK4_mag7.3_pred.txt
├── DAVIS_SAOCOM1B_46265_2024-12-04-18-21-37_pred.txt
├── DVX_Filtered_Stars3_2025-01-20-20-22-53_pred.txt
└── DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43_pred.txt
```

Each prediction file is a **tab-separated** file with a header row:

| Column | Type | Description |
|--------|------|-------------|
| `window_start_timestamp_us` | int | Start of the predicted time window (µs) |
| `window_end_timestamp_us` | int | End of the predicted time window (µs) |
| `center_x` | int | Predicted bounding box center X (pixels) |
| `center_y` | int | Predicted bounding box center Y (pixels) |
| `width` | int | Predicted bounding box width (pixels) |
| `height` | int | Predicted bounding box height (pixels) |
| `confidence` | float | Detection confidence score *(optional, defaults to 1.0)* |

Example:

```
window_start_timestamp_us	window_end_timestamp_us	center_x	center_y	width	height	confidence
1000000	1040000	213	187	16	16	0.97
1040000	1080000	215	185	16	16	0.83
```

> **Notes:**
> - One row per predicted detection. If your algorithm detects nothing in a window, omit that window entirely.
> - A prediction window is matched to a GT window if they **overlap in time** and the bounding boxes achieve **IoU ≥ 0.5**.
> - `confidence` is used to rank predictions for AP computation. If omitted, all predictions are treated equally.

---

## Evaluation

Participants are asked to submit an **evaluation metric sheet** as part of their submission. To generate it, run the evaluation script to compute detection metrics against ground truth:

```bash
python3 evaluate.py --gt-dir ../OrbitSight_Dataset --pred-dir ../predictions
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--gt-dir` | Directory containing the ground truth `*_bb_windows_40ms.txt` files |
| `--pred-dir` | Directory containing your prediction `*_pred.txt` files |
| `--iou` | IoU threshold for a true positive (default: `0.5`) |

**Metrics reported:**

| Metric | Description |
|--------|-------------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | Harmonic mean of Precision and Recall |
| AP @ IoU 0.5 | Area under the Precision-Recall curve per sequence |
| mAP @ IoU 0.5 | Mean AP across all sequences |

**Example output:**

```
====================================================================================================
  OrbitSight Detection Evaluation - Per Sequence
====================================================================================================
| Sequence                  | GT boxes | Pred boxes | Precision | Recall | F1     | AP@0.5 |
|---------------------------|----------|------------|-----------|--------|--------|--------|
| DVX_Filtered_Thuraya3_... |       42 |         38 |    0.8947 | 0.8095 | 0.8500 | 0.8312 |
| ...                       |      ... |        ... |       ... |    ... |    ... |    ... |

==================================================
  Overall Results
==================================================
| mAP @ IoU 0.5    | 0.8312 |
| Precision        | 0.8947 |
| Recall           | 0.8095 |
| F1 Score         | 0.8500 |
| Total TP         |     34 |
| Total FP         |      4 |
| Total FN (missed)|      8 |
```

---

## Submission requirements

Participants are required to submit **two items**:

1. A **Docker image** (see [Docker image](#docker-image) below)
2. A **5-page technical proposal** in `.pdf` format (see [Technical proposal](#technical-proposal) below)

---

### Technical proposal

Submit a 5-page technical proposal (`.pdf`) covering the following sections:

| # | Section | Description |
|---|---------|-------------|
| 1 | Problem statement and proposed solution | Define the problem and describe your approach |
| 2 | Outcome metrics | Effectiveness measures used to evaluate the solution: mAP (mean Average Precision), precision, recall, F1 score, and inference efficiency |
| 3 | Value proposition and competitive positioning | What makes your solution stand out |
| 4 | Technical approach and solution architecture | Detailed methodology and expected outputs |
| 5 | Team capacity | Participant background, capacity, and capability to develop the solution |
| 6 | Prior work | Details on any proof of concept (POC), additional development, or existing applications of the solution |

---

### Docker image

Participants are required to submit a **Docker image** that meets the following criteria.

### What to upload

Export your image with `docker save` and upload the resulting archive:

```bash
docker save yourimage:tag -o image.tar        # or gzip-compressed: image.tar.gz
```

### Runtime behaviour

- The container must run **non-interactively** and finish on its own - no manual input.
- Provide an automatic entrypoint, e.g. `CMD ["sh", "run.sh"]`. Avoid interactive shells.
- Containers run **offline** (no internet access).

### Mounted paths

| Path | Access | Contents |
|------|--------|----------|
| `/OrbitSight_dataset` | read-only | OrbitSight event recordings (training & testing sequences as `*.npy`), ground-truth metadata (`*.txt`), and the `/OrbitSight_dataloader` folder to load and explore the data |
| `/work/teamName/DDMMYYYY` | write-only | Portal collects this folder for prediction results and scoring sheet |

### Model files

Include inside the Docker image:
- AI model weights
- Model structure file
- Inference script - accepts OrbitSight `*.npy` event recordings as input, processes them through the participant's AI model, and writes a `<sequence>_pred.txt` prediction file (see [Submission format](#submission-format)) that can be passed directly to `evaluate.py` to generate the `Evaluation_Metrics.xlsx` scoring sheet

### Output format

Save all outputs to `/work/teamName/DDMMYYYY`. Required files:

- **`<sequence>_pred.txt`** - one detection per row (same format as the ground truth `.txt` files, plus an optional `confidence` column):

  | Field | Type | Description |
  |-------|------|-------------|
  | `window_start_timestamp_us` | int | Start of the predicted time window (µs) |
  | `window_end_timestamp_us` | int | End of the predicted time window (µs) |
  | `center_x` | int | Bounding box center X (pixels) |
  | `center_y` | int | Bounding box center Y (pixels) |
  | `width` | int | Bounding box width (pixels) |
  | `height` | int | Bounding box height (pixels) |
  | `confidence` | float | Detection confidence score *(optional, defaults to 1.0)* |

- **`Evaluation_Metrics.xlsx`** - scoring sheet generated by the evaluation script (see [Evaluation](#evaluation) section above)

### Portal account

Each team must create a Portal account using their email address as the login username.
