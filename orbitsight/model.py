"""Stage 2 classifier head: a gradient-boosted-tree model over the local
coherence features.  CPU-fast, trains in minutes, no GPU (Tech Report §5.3).

Falls back to a simple, dependency-light scoring rule if no trained model is
present, so the pipeline always runs end-to-end (de-risks the container).
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .config import Config

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:                       # pragma: no cover
    _HAS_LGB = False

import joblib


class CoherenceClassifier:
    """Wraps a LightGBM binary classifier on per-event coherence features."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.idx = list(cfg.model_feature_idx)            # input feature subset
        self.feature_names = [cfg.feature_names[i] for i in self.idx]

    def _select(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.idx] if X.shape[1] != len(self.idx) else X

    # ---- training ------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        if not _HAS_LGB:
            raise RuntimeError("lightgbm not available for training")
        X = self._select(X)
        pos = max(int((y == 1).sum()), 1)
        neg = max(int((y == 0).sum()), 1)
        self.model = lgb.LGBMClassifier(
            n_estimators=400,
            num_leaves=63,
            learning_rate=0.05,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=neg / pos,     # counter class imbalance
            n_jobs=-1,
            random_state=self.cfg.random_seed,
            verbosity=-1,
        )
        self.model.fit(X, y, sample_weight=sample_weight,
                       feature_name=self.feature_names)
        return self

    # ---- inference ------------------------------------------------------ #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        if self.model is not None:
            Xs = self._select(X)
            # predict on the raw feature matrix; silence the benign
            # "X does not have valid feature names" sklearn/LightGBM warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.model.predict_proba(Xs)[:, 1].astype(np.float32)
        return self._heuristic(X)

    def _heuristic(self, X: np.ndarray) -> np.ndarray:
        """Fallback score from raw coherence features (no learned weights).

        Linearity (col 2) and flow consistency (col 5) are the H1 signals;
        density (col 0) guards against sparse noise.
        """
        lin = X[:, 2]
        flow = X[:, 5]
        dens = X[:, 0]
        s = (np.tanh(lin / 6.0) * 0.5 + flow * 0.4
             + np.clip(dens / 30.0, 0, 1) * 0.1)
        return np.clip(s, 0, 1).astype(np.float32)

    # ---- persistence ---------------------------------------------------- #
    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"model": self.model,
                     "feature_names": self.feature_names,
                     "config": self.cfg.to_dict()}, path)
        # also dump a human-readable structure file (PRD: "model structure file")
        meta = {"feature_names": self.feature_names,
                "type": "LGBMClassifier" if self.model is not None else "heuristic"}
        if self.model is not None:
            meta["n_estimators"] = self.model.n_estimators_
            meta["importances"] = dict(zip(
                self.feature_names,
                [int(v) for v in self.model.booster_.feature_importance()]))
        with open(os.path.splitext(path)[0] + "_structure.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str, cfg: Config) -> "CoherenceClassifier":
        obj = cls(cfg)
        if path and os.path.exists(path):
            blob = joblib.load(path)
            obj.model = blob["model"]
            obj.feature_names = blob.get("feature_names", obj.feature_names)
        return obj
