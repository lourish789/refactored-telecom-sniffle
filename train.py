"""Train the fraud detection models.

Two complementary detectors:
1. IsolationForest (unsupervised) — realistic for production where fraud labels
   are scarce; scores behavioral outliers.
2. XGBoost classifier (supervised) — trained on the injected labels; gives
   calibrated fraud probability and feature importances.

The API combines both into a single risk score.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config


def train():
    feats = pd.read_parquet(config.FEATURES_PATH)
    X = feats[config.FEATURE_COLUMNS].values
    y = feats["is_fraud"].values

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # --- Unsupervised: Isolation Forest ---
    iso = IsolationForest(
        n_estimators=300,
        contamination=config.FRAUD_RATE,
        random_state=config.RANDOM_SEED,
    ).fit(Xs)
    # decision_function: lower = more anomalous. Convert to [0,1] anomaly score.
    raw = iso.decision_function(Xs)
    iso_score = (raw.max() - raw) / (raw.max() - raw.min())
    iso_auc = roc_auc_score(y, iso_score)

    # --- Supervised: XGBoost ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=0.25, stratify=y, random_state=config.RANDOM_SEED
    )
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        eval_metric="aucpr",
        random_state=config.RANDOM_SEED,
    ).fit(X_tr, y_tr)
    proba = xgb.predict_proba(X_te)[:, 1]
    xgb_auc = roc_auc_score(y_te, proba)
    report = classification_report(y_te, (proba > 0.5).astype(int), output_dict=True)

    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(iso, config.ISO_FOREST_PATH)
    joblib.dump(xgb, config.XGB_PATH)

    importances = dict(zip(
        config.FEATURE_COLUMNS,
        [round(float(v), 4) for v in xgb.feature_importances_],
    ))
    metrics = {
        "isolation_forest_auc": round(float(iso_auc), 4),
        "xgboost_auc": round(float(xgb_auc), 4),
        "xgboost_precision_fraud": round(report["1"]["precision"], 4),
        "xgboost_recall_fraud": round(report["1"]["recall"], 4),
        "xgboost_f1_fraud": round(report["1"]["f1-score"], 4),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "fraud_rate": round(float(y.mean()), 4),
        "feature_importances": dict(
            sorted(importances.items(), key=lambda kv: -kv[1])
        ),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
