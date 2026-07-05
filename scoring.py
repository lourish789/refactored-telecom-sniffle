"""Model serving layer: loads trained artifacts and scores subscribers.

Combined risk score = 0.5 * isolation-forest anomaly + 0.5 * xgboost fraud
probability, both in [0, 1]. Risk bands come from config thresholds.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

import config


class FraudScorer:
    def __init__(self):
        self.scaler = joblib.load(config.SCALER_PATH)
        self.iso = joblib.load(config.ISO_FOREST_PATH)
        self.xgb = joblib.load(config.XGB_PATH)
        # Reference range so streamed batches get the same anomaly normalization
        ref = pd.read_parquet(config.FEATURES_PATH)
        raw = self.iso.decision_function(self.scaler.transform(ref[config.FEATURE_COLUMNS].values))
        self._raw_min, self._raw_max = float(raw.min()), float(raw.max())

    def score(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Score feature rows; returns the frame with score & risk columns added."""
        X = self.scaler.transform(feats[config.FEATURE_COLUMNS].values)
        raw = self.iso.decision_function(X)
        iso_score = np.clip(
            (self._raw_max - raw) / (self._raw_max - self._raw_min), 0, 1
        )
        xgb_score = self.xgb.predict_proba(X)[:, 1]
        out = feats.copy()
        out["iso_score"] = np.round(iso_score, 4)
        out["xgb_score"] = np.round(xgb_score, 4)
        out["risk_score"] = np.round(0.5 * iso_score + 0.5 * xgb_score, 4)
        out["risk_band"] = pd.cut(
            out["risk_score"],
            bins=[-0.01, config.RISK_MEDIUM, config.RISK_HIGH, 1.01],
            labels=["low", "medium", "high"],
        ).astype(str)
        return out
