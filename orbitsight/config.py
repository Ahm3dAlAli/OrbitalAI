"""Central configuration: sensor geometry and pipeline hyper-parameters.

Every spatial parameter is defined in *normalized* units (fraction of the
sensor's diagonal) so that a single parameter set transfers across the three
sensor resolutions (PRD NFR-3, Tech Report Stage 0 / Section 6).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
import re


# --------------------------------------------------------------------------- #
#  Sensor registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Sensor:
    name: str
    width: int
    height: int

    @property
    def diag(self) -> float:
        return math.hypot(self.width, self.height)


SENSORS = {
    "DAVIS": Sensor("DAVIS", 346, 260),
    "DVX":   Sensor("DVX",   640, 480),
    "EVK4":  Sensor("EVK4",  1280, 720),
}


def sensor_for_sequence(seq_name: str) -> Sensor:
    """Infer the sensor from a sequence file name.

    Names embed the sensor token (DAVIS / DVX / EVK4).  EVK4 is matched first
    because some names contain other tokens.
    """
    s = seq_name.upper()
    if "EVK4" in s:
        return SENSORS["EVK4"]
    if "DAVIS" in s:
        return SENSORS["DAVIS"]
    if "DVX" in s:
        return SENSORS["DVX"]
    # Fall back to the largest sensor (conservative for coordinate clipping).
    return SENSORS["EVK4"]


# --------------------------------------------------------------------------- #
#  Pipeline hyper-parameters  (single set, all sensors)
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # -- windowing --------------------------------------------------------- #
    window_us: int = 40_000          # 40 ms challenge window (Δ)
    halo_us: int = 20_000            # temporal context added on each side for
    #                                  feature/denoise neighborhoods

    # -- Stage 1: background-activity denoise ----------------------------- #
    denoise_dt_us: int = 20_000      # δt: neighbor must have fired within this
    denoise_min_support: int = 1     # k: required supporting neighbors (gentle)
    denoise_radius_px: float = 3.0   # spatial ball radius in pixels

    # -- Stage 2: coherence features -------------------------------------- #
    # Neighborhood radius in NORMALIZED units (fraction of sensor diagonal).
    feat_radius_norm: float = 0.02   # ~ space ball radius
    feat_time_us: int = 20_000       # ± time half-window for the (x,y,t) cloud
    feat_max_neighbors: int = 48     # cap neighbors per event (latency guard)
    feat_min_neighbors: int = 3      # below this, features are degenerate
    feat_query_cap: int = 1500       # max events featurized per window (latency)

    # how strongly time is weighted relative to space when forming the 3-D
    # (x_n, y_n, t_n) point cloud:  t_n = (t / feat_time_us) * time_scale
    time_scale: float = 0.02

    # -- Stage 3: detection / boxing -------------------------------------- #
    score_threshold: float = 0.6     # per-event RSO probability cut (floor)
    cluster_eps_norm: float = 0.03   # cluster eps in normalized units
    cluster_min_samples: int = 3     # min events to seed a cluster
    box_margin_px: int = 3           # padding added around inlier extent
    min_box_px: int = 4              # minimum emitted box side
    min_cluster_events: int = 4      # min events for a candidate cluster
    max_candidates: int = 4          # per-window proposals kept for tracking

    # per-window percentile score gate (recall): instead of a single absolute
    # cut, keep the top (100-keep_percentile)% of events in each window subject
    # to an absolute floor.  Adapts the operating point to per-window event rate
    # so dim windows surface their best events (DAVIS/DVX/EVK4, one param set).
    use_percentile_gate: bool = True
    keep_percentile: float = 80.0    # keep events above this within-window pct
    score_floor: float = 0.45        # absolute minimum score (noise guard)
    sparse_window_max: int = 80      # only relax the cut when a window has <=
    #                                  this many events (dim objects); dense
    #                                  windows keep the strict absolute cut so
    #                                  bright sensors (EVK4) are not flooded

    # -- global tracker --------------------------------------------------- #
    track_gate_norm: float = 0.05    # gating radius (fraction of diagonal)
    track_max_gap: int = 3           # windows a track may coast through
    min_track_len: int = 5           # min detections to accept a track
    min_track_disp_px: float = 12.0  # min net motion -> rejects static clutter
    max_track_residual_px: float = 6.0  # max RMS deviation from smooth motion
    strong_track_evidence: float = 400.0  # evidence overriding the motion gate
    max_tracks_per_seq: int = 40     # cap on tracks (long multi-object sequences)
    tracks_per_kwin: float = 4.0     # + this many tracks per 1000 windows
    track_evidence_frac: float = 0.04  # drop tracks weaker than this × best

    # -- recall recovery: bidirectional track extension ------------------- #
    # Confirmed tracks are extrapolated past their observed ends along a fitted
    # constant-velocity model, converting sub-threshold FN windows -> TP at low
    # precision cost (the track is already confirmed real).
    track_extend: int = 8            # max windows to extrapolate each side
    extend_min_core: int = 4         # min real detections before extrapolating
    extend_conf_decay: float = 0.85  # per-window confidence decay when coasting

    # -- recall: per-sensor box-size IoU calibration ---------------------- #
    # Multiplicative (sw, sh) correction per sensor, fit offline on TP pairs by
    # scripts/calibrate.py; identity by default.  Recovers TPs lost only because
    # boxes were systematically mis-sized despite correct localization.
    box_size_scale: dict = field(default_factory=dict)

    # Per-sensor typical box size (w, h) in pixels, the median GT box for that
    # sensor (learned offline by scripts/calibrate.py).  Dim windows whose event
    # extent is tiny emit a box at this typical size rather than a too-small
    # extent box, which clears IoU>=0.5 far more often.
    box_size_px: dict = field(default_factory=dict)
    # Per-sensor blend weight toward the typical GT size vs. the measured event
    # extent.  Oracle-ceiling analysis (§6.1) shows the optimum is sensor-
    # dependent: EVK4's object FILLS its box, so the tight event extent is best
    # (typical over-sizes it); DAVIS/DVX objects are small/sparse with larger GT
    # boxes, so the typical size is far better (Thuraya3 ceiling 0.97 vs 0.00).
    # EVK4 -> tight (0.2): object fills its box; typical over-sizes it.
    # DVX/DAVIS -> typical-leaning, but NOT pure: the cross-sequence typical
    # (DVX 10x12) under-sizes the larger Stars3 objects (GT 13x13), so retaining
    # ~25% measured extent is better than forcing pure typical.
    box_size_blend: dict = field(default_factory=lambda: {
        "EVK4": 0.2, "DAVIS": 0.85, "DVX": 0.75})
    box_size_blend_default: float = 0.75

    # -- temporal accumulation (synthetic tracking / shift-and-stack) ----- #
    # Recovers dim moving objects whose 2-4 events/window never trip the
    # per-window detector: shift events back along a hypothesized velocity to a
    # common time so the object stacks while background smears.  Used as an
    # optional candidate source that fills windows the classifier left empty.
    stack_block: int = 21            # windows integrated per block
    stack_stride: int = 7            # block hop
    stack_vmax: float = 6.0          # velocity search range (px/window)
    stack_vstep: float = 0.5         # velocity grid step
    stack_cell: float = 6.0          # accumulator cell size (px)
    stack_min_ratio: float = 4.0     # peak / typical-velocity-peak outlier gate
    stack_min_peak: int = 16         # absolute stacked-peak floor
    stack_hot_frac: float = 0.10     # drop pixels active in > this frac of windows

    # -- training --------------------------------------------------------- #
    bg_per_rso: int = 30             # background:RSO sampling ratio for training
    max_events_per_seq: int = 4_000_000  # cap for memory during training
    max_rso_per_seq: int = 8_000     # per-sequence positive cap (balance sensors)
    random_seed: int = 1234

    # Model input features: a brightness-INVARIANT subset (exclude the absolute
    # count features n_neighbors[0] and density[1], which let the classifier
    # take a "dense = RSO" shortcut that fails on dim objects — H2).  The model
    # is forced onto coherence + polarity + shape, which transfer across
    # brightness and resolution.
    model_feature_idx: tuple = field(
        default=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11), init=False)

    # feature names (kept in sync with features.compute_features)
    feature_names: tuple = field(default=(
        "n_neighbors",
        "density",
        "lin_ratio",        # PCA linearity λ1/(λ2+λ3)
        "planar_ratio",     # λ2/λ3
        "lambda1_frac",     # λ1/(λ1+λ2+λ3)
        "flow_consistency",
        "speed_norm",
        "time_spread",
        "space_spread",
        "pol_mean",
        "pol_entropy",
        "aniso",            # 1 - λ3/λ1  (overall anisotropy)
    ), init=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feature_names"] = list(self.feature_names)
        return d


DEFAULT_CONFIG = Config()

# Canonical training/testing sequence lists (from the DataLoader README).
TRAIN_SEQUENCES = [
    "2025_12_23_21_12_28_EVK4_mag5.2",
    "DAVIS_COSMOS1933_18958_2024-12-04-18-37-01",
    "DAVIS_EGS_16908_2024-11-01-19-10-44",
    "DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06",
    "DAVIS_RESURSDK1_29228_2024-12-04-18-37-01",
    "DAVIS_SL12RB2_15772_2024-12-04-18-21-37",
    "DAVIS_SL16RB_20625_2024-12-04-19-34-18",
    "DAVIS_SL16RB_26070_2024-12-04-19-14-39",
    "DAVIS_SL8RB_2025-01-13-19-15-36",
    "DVX_Filtered_ACS3_59588_2025-01-20-19-35-44",
    "DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17",
    "DVX_Filtered_NOAA15_25338_2025-01-20-19-25-07",
    "DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50",
    "DVX_Filtered_NOAA6_11416_2025-01-20-19-11-35",
    "DVX_Filtered_Stars_2025-01-20-19-15-10",
    "DVX_Filtered_Stars2_2025-01-20-19-57-17",
    "DVX_NOAA6_11416_2025-01-20-19-06-31",
]

TEST_SEQUENCES = [
    "2025_12_23_20_53_46_EVK4_mag7.3",
    "DAVIS_SAOCOM1B_46265_2024-12-04-18-21-37",
    "DVX_Filtered_Stars3_2025-01-20-20-22-53",
    "DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43",
]
