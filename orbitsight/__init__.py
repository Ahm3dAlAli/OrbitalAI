"""OrbitSight — Real-time RSO detection & tracking for neuromorphic vision.

A four-stage, CPU-only, O(N) pipeline:

    Stage 0  Normalizer            raw events -> sensor-normalized events
    Stage 1  Denoiser              O(N) background-activity filter
    Stage 2  Coherence classifier  learned per-event RSO score (the novelty)
    Stage 3  Tracker               positive events -> one box per 40 ms window
    Stage 4  Emitter / Visualizer  boxes -> .txt + .xlsx + overlays

See orbitsight/pipeline.py for the orchestration and the companion
Technical Research Report for method rationale.
"""

__version__ = "1.0.0"
