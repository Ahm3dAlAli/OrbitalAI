Dataset Description and Contents

The OrbitSight dataset consists of real event-based recordings captured by neuromorphic vision sensors mounted on a high-end 0.8-meter diameter telescope at the Abu Dhabi Quantum Optical Ground Station (ADQOGS). All data was recorded under real observational conditions, including low-light environments, fast-moving space objects, and the inherent noise characteristics of neuromorphic sensing
Contents
The dataset (8.5-9.0GB) contains sequences of real NVS observations of Resident Space Objects (RSOs), satellites, rocket bodies, stars*, and orbital debris, captured from the ground under varying lighting and atmospheric conditions.
*Stars are included as RSO objects due to their apparent motion in the event stream: see Ground Truth & Labeling for details.
The dataset is split into two subsets:
Training set: 17 sequences, provided with full ground truth labels and bounding box annotations, to be used for model development and optimization
Testing set: 4 sequences, provided with event streams and ground truth bounding box annotations, to be used for final evaluation and scoring
Each sequence is provided as two paired files:
File
Description
<sequence>_labeled_events.npy
Event stream with ground truth labels
<sequence>_bb_windows_40ms.txt
Bounding box ground truth per 40 ms time window


Sensors
The dataset was captured using three event-based sensors:
Sensor
Type
Resolution
DAVIS
DAVIS346c Dynamic and Active-pixel Vision Sensor -  outputs both events and frames
346 × 260
DVX
DVXplore - Pure event-based camera
640 × 480
EVK4
Prophesee Metavision EVK4 -  high-resolution pure event-based camera
1280 × 720


Data Format & Structure
After loading a sequence file, you can access the data as follows:
import numpy as np


events = np.load('OrbitSight_Dataset/<sequence>_labeled_events.npy')
# events.shape → (N, 6)


Each .npy file is a NumPy array of shape (N, 6) with the following columns:
Col
Name
Description
0
x
Pixel column (integer)
1
y
Pixel row (integer)
2
polarity
0 = brightness decrease, 1 = brightness increase
3
timestamp_us
Absolute timestamp in microseconds [µs]
4
label
0 = background, 1 = RSO (Resident Space Object)
5
relative_timestamp_us
Timestamp relative to the first event in the sequence


The bounding box (.txt) file is tab-separated, with one row per 40 ms time window containing a detection:
Column
Description
window_start_timestamp_us
Start of the time window [µs]
window_end_timestamp_us
End of the time window [µs]
center_x
Bounding box center X [pixels]
center_y
Bounding box center Y [pixels]
width
Bounding box width [pixels]
height
Bounding box height [pixels]


Available Sequences
Training Sequences — 17 sequences
Ground truth labels and bounding box files are provided for all training sequences.
#
Sequence Name
Sensor
1
2025_12_23_21_12_28_EVK4_mag5.2
EVK4
2
DAVIS_COSMOS1933_18958_2024-12-04-18-37-01
DAVIS
3
DAVIS_EGS_16908_2024-11-01-19-10-44
DAVIS
4
DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06
DAVIS
5
DAVIS_RESURSDK1_29228_2024-12-04-18-37-01
DAVIS
6
DAVIS_SL12RB2_15772_2024-12-04-18-21-37
DAVIS
7
DAVIS_SL16RB_20625_2024-12-04-19-34-18
DAVIS
8
DAVIS_SL16RB_26070_2024-12-04-19-14-39
DAVIS
9
DAVIS_SL8RB_2025-01-13-19-15-36
DAVIS
10
DVX_Filtered_ACS3_59588_2025-01-20-19-35-44
DVX
11
DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17
DVX
12
DVX_Filtered_NOAA15_25338_2025-01-20-19-25-07
DVX
13
DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50
DVX
14
DVX_Filtered_NOAA6_11416_2025-01-20-19-11-35
DVX
15
DVX_Filtered_Stars_2025-01-20-19-15-10
DVX
16
DVX_Filtered_Stars2_2025-01-20-19-57-17
DVX
17
DVX_NOAA6_11416_2025-01-20-19-06-31
DVX


Testing Sequences — 4 sequences
Ground truth labels and bounding box files are provided to allow participants to evaluate their detection performance.
#
Sequence Name
Sensor
1
2025_12_23_20_53_46_EVK4_mag7.3
EVK4
2
DAVIS_SAOCOM1B_46265_2024-12-04-18-21-37
DAVIS
3
DVX_Filtered_Stars3_2025-01-20-20-22-53
DVX
4
DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43
DVX


Ground Truth & Labeling
Ground truth annotations are provided as bounding boxes around detected RSOs. Each bounding box is defined per 40 ms time window and includes center coordinates and dimensions in pixels.
RSOs in this dataset include satellites, rocket bodies, and orbital debris, as well as stars. Although stars are not man-made objects, their apparent motion across the sensor field of view, caused by Earth’s rotation and telescope tracking, produces an event stream signature indistinguishable from that of a real RSO. Stars are therefore labelled as RSO objects (label = 1) and included in the ground truth annotations.
Ground truth annotations are provided for both the training and test sets. Participants are expected to train their models exclusively on the training sequences. The test sequences should be treated as unseen during training; their ground truth is provided solely to allow participants to self-evaluate their detection scores before submission. Final scoring will be computed against the same ground truth by the challenge organisers.
A prediction is counted as a true positive when it overlaps with a ground truth box at an IoU threshold of ≥ 0.5.

Getting Started
A starter toolkit (OrbitSight DataLoader) is provided to support participants with:
Loading and exploring sequences
Visualizing event streams and bounding boxes
Preparing and formatting submissions, the toolkit outputs <sequence>_pred.txt prediction files ready to pass to the evaluation script


A simple starting workflow to split events into 40 ms windows:
import numpy as np


events = np.load('OrbitSight_Dataset/<sequence>_labeled_events.npy')


WINDOW_US = 40_000
ev_t = events[:, 3]
window_starts = np.arange(ev_t[0], ev_t[-1], WINDOW_US)
break_idx = np.searchsorted(ev_t, window_starts)


for i in range(len(window_starts) - 1):
    window = events[break_idx[i]:break_idx[i+1]]
    x, y, polarity, ts, label, rel_ts = window.T
    # --- your detection algorithm here ---
    # Use (x, y, polarity) as input features
    # label == 1 marks ground truth RSO events (training only)



