"""Feature engineering: aggregate raw CDRs into per-subscriber behavioral features.

Each feature maps to a known fraud signal:
- max_velocity_kmh        -> SIM cloning / impossible travel
- short_call_ratio + distinct_callees -> SIM box bypass
- premium_dest_ratio + intl_ratio     -> IRSF
- account_age_days + avg_daily_cost   -> subscription fraud
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from data_generator import haversine_km

NIGHT_HOURS = set(range(0, 6)) | {22, 23}
SHORT_CALL_SEC = 60


def _max_velocity(group: pd.DataFrame) -> float:
    """Max implied travel speed (km/h) between consecutive events of a subscriber."""
    g = group.sort_values("timestamp")
    if len(g) < 2:
        return 0.0
    lat, lon = g["tower_lat"].to_numpy(), g["tower_lon"].to_numpy()
    dist = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt_hours = np.diff(g["timestamp"].astype("int64")) / 3.6e12
    dt_hours = np.clip(dt_hours, 1 / 60, None)  # floor at 1 minute to avoid div-by-zero
    with np.errstate(divide="ignore", invalid="ignore"):
        v = dist / dt_hours
    return float(np.nanmax(v)) if len(v) else 0.0


def build_features(cdrs: pd.DataFrame) -> pd.DataFrame:
    df = cdrs.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour"].isin(NIGHT_HOURS)
    df["is_voice"] = df["call_type"] == "voice"
    df["is_sms"] = df["call_type"] == "sms"
    df["is_out"] = df["direction"] == "out"
    df["is_short"] = df["is_voice"] & (df["duration_sec"] < SHORT_CALL_SEC)
    n_days = max((df["timestamp"].max() - df["timestamp"].min()).days, 1)

    g = df.groupby("subscriber_id")
    feats = pd.DataFrame({
        "total_calls": g.size(),
        "avg_duration": g["duration_sec"].mean(),
        "std_duration": g["duration_sec"].std().fillna(0),
        "intl_ratio": g["is_international"].mean(),
        "night_ratio": g["is_night"].mean(),
        "outgoing_ratio": g["is_out"].mean(),
        "distinct_callees": g["callee_id"].nunique(),
        "distinct_towers": g["tower_id"].nunique(),
        "sms_ratio": g["is_sms"].mean(),
        "avg_daily_cost": g["cost"].sum() / n_days,
        "account_age_days": g["account_age_days"].first(),
        "premium_dest_ratio": g["is_premium_dest"].mean(),
        "short_call_ratio": g["is_short"].mean(),
    })
    feats["callee_reuse_ratio"] = 1 - feats["distinct_callees"] / feats["total_calls"].clip(lower=1)
    feats["max_velocity_kmh"] = g.apply(_max_velocity, include_groups=False)

    # Carry labels + OSS/BSS metadata through for training and the API layer
    meta = g.agg(
        is_fraud=("is_fraud", "first"),
        fraud_type=("fraud_type", "first"),
        plan_type=("plan_type", "first"),
        billing_status=("billing_status", "first"),
    )
    feats = feats.join(meta).reset_index()
    return feats


if __name__ == "__main__":
    cdrs = pd.read_parquet(config.CDR_PATH)
    feats = build_features(cdrs)
    feats.to_parquet(config.FEATURES_PATH, index=False)
    print(f"Built {len(feats)} subscriber feature rows -> {config.FEATURES_PATH}")
    print(feats.groupby("fraud_type")[["total_calls", "intl_ratio", "max_velocity_kmh"]].mean().round(2))
