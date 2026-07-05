"""Synthetic Call Detail Record (CDR) generator with injected fraud patterns.

Generates realistic telecom CDRs for a population of subscribers, then injects
four industry-standard fraud archetypes:

- simbox:             SIM box / bypass fraud — high-volume short outgoing calls,
                      stationary tower, hundreds of distinct callees, night-heavy.
- irsf:               International Revenue Share Fraud — bursts of long calls to
                      premium international destinations.
- sim_cloning:        Two devices on one identity — impossible travel between
                      distant towers within minutes.
- subscription_fraud: Brand-new account with immediate abnormally heavy usage
                      and delinquent billing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

RNG = np.random.default_rng(config.RANDOM_SEED)

# Cell towers on a rough grid around Lagos, Nigeria (lat, lon)
N_TOWERS = 60
TOWER_LAT = RNG.uniform(6.35, 6.70, N_TOWERS)
TOWER_LON = RNG.uniform(3.10, 3.60, N_TOWERS)

PREMIUM_DESTS = ["+882", "+979", "+371", "+673", "+252"]  # known IRSF ranges
INTL_DESTS = ["+44", "+1", "+91", "+27", "+233"]
PLANS = ["prepaid", "postpaid", "corporate"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _make_subscribers(n: int) -> pd.DataFrame:
    subs = pd.DataFrame({
        "subscriber_id": [f"SUB{100000 + i}" for i in range(n)],
        "plan_type": RNG.choice(PLANS, n, p=[0.7, 0.25, 0.05]),
        "account_age_days": RNG.integers(30, 2000, n),
        "billing_status": RNG.choice(["current", "overdue"], n, p=[0.93, 0.07]),
        "home_tower": RNG.integers(0, N_TOWERS, n),
        "is_fraud": 0,
        "fraud_type": "none",
    })
    n_fraud = int(n * config.FRAUD_RATE)
    fraud_idx = RNG.choice(n, n_fraud, replace=False)
    subs.loc[fraud_idx, "is_fraud"] = 1
    subs.loc[fraud_idx, "fraud_type"] = RNG.choice(config.FRAUD_TYPES, n_fraud)
    # Subscription fraudsters are brand-new accounts, usually delinquent
    sub_fraud = subs["fraud_type"] == "subscription_fraud"
    subs.loc[sub_fraud, "account_age_days"] = RNG.integers(1, 14, sub_fraud.sum())
    subs.loc[sub_fraud, "billing_status"] = "overdue"
    return subs


def _normal_cdrs(sub: pd.Series, start: pd.Timestamp) -> list[dict]:
    """Typical usage: 5-25 events/day, business-hours weighted, local mobility."""
    records = []
    n_events = int(RNG.poisson(12) * config.N_DAYS)
    nearby = [sub.home_tower, (sub.home_tower + 1) % N_TOWERS, (sub.home_tower + 2) % N_TOWERS]
    for _ in range(n_events):
        day = RNG.integers(0, config.N_DAYS)
        hour = int(np.clip(RNG.normal(14, 4.5), 0, 23))
        ts = start + pd.Timedelta(days=int(day), hours=hour, minutes=int(RNG.integers(0, 60)))
        call_type = RNG.choice(["voice", "sms", "data"], p=[0.55, 0.3, 0.15])
        is_intl = RNG.random() < 0.03
        dest = RNG.choice(INTL_DESTS) if is_intl else "+234"
        duration = float(RNG.exponential(180)) if call_type == "voice" else 0.0
        records.append(_row(sub, ts, call_type, dest, duration, int(RNG.choice(nearby)),
                            direction=RNG.choice(["out", "in"], p=[0.55, 0.45]),
                            callee=f"NUM{RNG.integers(0, 400)}"))
    return records


def _simbox_cdrs(sub: pd.Series, start: pd.Timestamp) -> list[dict]:
    """SIM box: 150-400 short outgoing local calls/day, one tower, unique callees."""
    records = []
    for day in range(config.N_DAYS):
        for _ in range(int(RNG.integers(150, 400))):
            hour = int(RNG.choice(24, p=_night_weighted()))
            ts = start + pd.Timedelta(days=day, hours=hour, minutes=int(RNG.integers(0, 60)),
                                      seconds=int(RNG.integers(0, 60)))
            records.append(_row(sub, ts, "voice", "+234", float(RNG.exponential(45)),
                                int(sub.home_tower), direction="out",
                                callee=f"NUM{RNG.integers(0, 100000)}"))
    return records


def _irsf_cdrs(sub: pd.Series, start: pd.Timestamp) -> list[dict]:
    """IRSF: normal use plus night bursts of long premium international calls."""
    records = _normal_cdrs(sub, start)
    for day in RNG.choice(config.N_DAYS, size=3, replace=False):
        burst_start = start + pd.Timedelta(days=int(day), hours=int(RNG.integers(0, 4)))
        for i in range(int(RNG.integers(20, 60))):
            ts = burst_start + pd.Timedelta(minutes=int(i * RNG.integers(8, 15)))
            records.append(_row(sub, ts, "voice", str(RNG.choice(PREMIUM_DESTS)),
                                float(RNG.uniform(300, 1800)), int(sub.home_tower),
                                direction="out", callee=f"PREM{RNG.integers(0, 40)}"))
    return records


def _cloning_cdrs(sub: pd.Series, start: pd.Timestamp) -> list[dict]:
    """Cloned SIM: two usage streams from towers ~40km apart, minutes between hits."""
    records = _normal_cdrs(sub, start)
    far_tower = int((sub.home_tower + N_TOWERS // 2) % N_TOWERS)
    for r in RNG.choice(len(records), size=max(4, len(records) // 3), replace=False):
        base_ts = records[int(r)]["timestamp"]
        ts = base_ts + pd.Timedelta(minutes=int(RNG.integers(2, 10)))
        records.append(_row(sub, ts, "voice", "+234", float(RNG.exponential(150)),
                            far_tower, direction="out", callee=f"NUM{RNG.integers(0, 400)}"))
    return records


def _subscription_fraud_cdrs(sub: pd.Series, start: pd.Timestamp) -> list[dict]:
    """New account, instant heavy international usage, never pays."""
    records = []
    for day in range(config.N_DAYS):
        for _ in range(int(RNG.integers(40, 90))):
            hour = int(RNG.integers(0, 24))
            ts = start + pd.Timedelta(days=day, hours=hour, minutes=int(RNG.integers(0, 60)))
            is_intl = RNG.random() < 0.6
            dest = str(RNG.choice(INTL_DESTS + PREMIUM_DESTS)) if is_intl else "+234"
            records.append(_row(sub, ts, "voice", dest, float(RNG.exponential(400)),
                                int(RNG.integers(0, N_TOWERS)), direction="out",
                                callee=f"NUM{RNG.integers(0, 5000)}"))
    return records


def _night_weighted() -> np.ndarray:
    w = np.ones(24)
    w[0:6] = 3.0
    w[22:24] = 2.5
    return w / w.sum()


def _row(sub, ts, call_type, dest_prefix, duration, tower, direction, callee) -> dict:
    intl = dest_prefix != "+234"
    premium = dest_prefix in PREMIUM_DESTS
    rate = 0.5 if not intl else (4.0 if premium else 1.5)  # cost units per minute
    return {
        "subscriber_id": sub.subscriber_id,
        "timestamp": ts,
        "call_type": call_type,
        "direction": direction,
        "callee_id": callee,
        "dest_prefix": dest_prefix,
        "is_international": intl,
        "is_premium_dest": premium,
        "duration_sec": round(duration, 1),
        "tower_id": int(tower),
        "tower_lat": float(TOWER_LAT[int(tower)]),
        "tower_lon": float(TOWER_LON[int(tower)]),
        "cost": round(duration / 60 * rate, 3),
        "plan_type": sub.plan_type,
        "billing_status": sub.billing_status,
        "account_age_days": int(sub.account_age_days),
        "is_fraud": int(sub.is_fraud),
        "fraud_type": sub.fraud_type,
    }


GENERATORS = {
    "none": _normal_cdrs,
    "simbox": _simbox_cdrs,
    "irsf": _irsf_cdrs,
    "sim_cloning": _cloning_cdrs,
    "subscription_fraud": _subscription_fraud_cdrs,
}


def generate(n_subscribers: int = config.N_SUBSCRIBERS) -> pd.DataFrame:
    start = pd.Timestamp("2026-06-22")
    subs = _make_subscribers(n_subscribers)
    all_records: list[dict] = []
    for sub in subs.itertuples(index=False):
        all_records.extend(GENERATORS[sub.fraud_type](sub, start))
    cdrs = pd.DataFrame(all_records).sort_values("timestamp").reset_index(drop=True)
    return cdrs


if __name__ == "__main__":
    cdrs = generate()
    cdrs.to_parquet(config.CDR_PATH, index=False)
    n_fraud_subs = cdrs.loc[cdrs.is_fraud == 1, "subscriber_id"].nunique()
    print(f"Generated {len(cdrs):,} CDRs for {cdrs.subscriber_id.nunique():,} subscribers "
          f"({n_fraud_subs} fraudulent) -> {config.CDR_PATH}")
    print(cdrs.fraud_type.value_counts())
