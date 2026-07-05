"""FastAPI backend for the telecom fraud detection system.

Endpoints
---------
GET  /api/health                      liveness + model/LLM status
GET  /api/metrics                     training metrics & feature importances
GET  /api/analytics/summary           dashboard aggregates
GET  /api/subscribers                 scored subscribers (filter by risk band)
GET  /api/subscribers/{id}            one subscriber: features, scores, recent CDRs
GET  /api/alerts                      high/medium-risk subscribers as alerts
POST /api/predict                     score a batch of raw CDRs (JSON)
POST /api/agent/investigate/{id}      LangGraph + Groq investigation report
POST /api/stream/start | /stop        control the live CDR stream simulator
GET  /api/stream/status               stream state + recently flagged events
GET  /                                dashboard frontend
"""
from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config
from agents.fraud_investigator import investigate
from features import build_features
from scoring import FraudScorer

# ---------------------------------------------------------------- state

class AppState:
    scorer: FraudScorer
    cdrs: pd.DataFrame
    scored: pd.DataFrame           # per-subscriber features + scores
    metrics: dict
    stream_task: Optional[asyncio.Task] = None
    stream_events: list = []       # recent flagged stream events (newest first)
    stream_ticks: int = 0
    stream_cdrs_processed: int = 0


S = AppState()


def _rescore_all():
    feats = build_features(S.cdrs)
    S.scored = S.scorer.score(feats)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import json
    S.scorer = FraudScorer()
    S.cdrs = pd.read_parquet(config.CDR_PATH)
    S.metrics = json.loads(config.METRICS_PATH.read_text())
    _rescore_all()
    yield
    if S.stream_task and not S.stream_task.done():
        S.stream_task.cancel()


app = FastAPI(
    title="Telecom Fraud & Anomaly Detection API",
    description="CDR-based fraud detection: IsolationForest + XGBoost + LangGraph/Groq agent",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------------------------------------------------------------- schemas

class CDRIn(BaseModel):
    subscriber_id: str
    timestamp: str
    call_type: str = Field(pattern="^(voice|sms|data)$")
    direction: str = Field(pattern="^(in|out)$")
    callee_id: str
    dest_prefix: str = "+234"
    duration_sec: float = 0.0
    tower_id: int = 0
    tower_lat: float = 6.5
    tower_lon: float = 3.35
    cost: float = 0.0
    plan_type: str = "prepaid"
    billing_status: str = "current"
    account_age_days: int = 365


class PredictRequest(BaseModel):
    cdrs: list[CDRIn]


# ---------------------------------------------------------------- helpers

SUB_COLS = ["subscriber_id", "plan_type", "billing_status", "risk_score",
            "iso_score", "xgb_score", "risk_band", "total_calls", "intl_ratio",
            "premium_dest_ratio", "short_call_ratio", "max_velocity_kmh",
            "distinct_callees", "night_ratio", "account_age_days", "avg_daily_cost"]


def _sub_records(df: pd.DataFrame) -> list[dict]:
    return df[SUB_COLS].replace([np.inf, -np.inf], 0).fillna(0).to_dict(orient="records")


def _prepare_cdrs(cdrs: list[CDRIn]) -> pd.DataFrame:
    df = pd.DataFrame([c.model_dump() for c in cdrs])
    df["is_international"] = df["dest_prefix"] != "+234"
    df["is_premium_dest"] = df["dest_prefix"].isin(["+882", "+979", "+371", "+673", "+252"])
    df["is_fraud"] = 0
    df["fraud_type"] = "unscored"
    return df


# ---------------------------------------------------------------- endpoints

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_loaded": S.scorer is not None,
        "subscribers_scored": int(len(S.scored)),
        "cdrs_loaded": int(len(S.cdrs)),
        "llm_configured": bool(config.GROQ_API_KEY),
        "llm_model": config.GROQ_MODEL,
        "stream_running": S.stream_task is not None and not S.stream_task.done(),
    }


@app.get("/api/metrics")
def metrics():
    return S.metrics


@app.get("/api/analytics/summary")
def analytics_summary():
    df = S.scored
    band_counts = df["risk_band"].value_counts().to_dict()
    hourly = (
        pd.to_datetime(S.cdrs["timestamp"]).dt.hour.value_counts().sort_index()
    )
    high = df[df["risk_band"] == "high"]
    return {
        "total_subscribers": int(len(df)),
        "total_cdrs": int(len(S.cdrs)),
        "risk_bands": {b: int(band_counts.get(b, 0)) for b in ["low", "medium", "high"]},
        "avg_risk_score": round(float(df["risk_score"].mean()), 4),
        "estimated_fraud_exposure_daily_cost": round(float(high["avg_daily_cost"].sum()), 2),
        "calls_by_hour": {int(h): int(c) for h, c in hourly.items()},
        "top_risk_subscribers": _sub_records(df.nlargest(10, "risk_score")),
        "stream": {
            "running": S.stream_task is not None and not S.stream_task.done(),
            "ticks": S.stream_ticks,
            "cdrs_processed": S.stream_cdrs_processed,
            "recent_flags": S.stream_events[:8],
        },
    }


@app.get("/api/subscribers")
def subscribers(risk: Optional[str] = None, limit: int = 100, offset: int = 0):
    df = S.scored.sort_values("risk_score", ascending=False)
    if risk:
        if risk not in ("low", "medium", "high"):
            raise HTTPException(422, "risk must be low|medium|high")
        df = df[df["risk_band"] == risk]
    total = len(df)
    return {"total": int(total), "items": _sub_records(df.iloc[offset:offset + limit])}


@app.get("/api/subscribers/{subscriber_id}")
def subscriber_detail(subscriber_id: str):
    row = S.scored[S.scored["subscriber_id"] == subscriber_id]
    if row.empty:
        raise HTTPException(404, f"unknown subscriber {subscriber_id}")
    recent = (
        S.cdrs[S.cdrs["subscriber_id"] == subscriber_id]
        .sort_values("timestamp", ascending=False)
        .head(25)
        .assign(timestamp=lambda d: d["timestamp"].astype(str))
        [["timestamp", "call_type", "direction", "callee_id", "dest_prefix",
          "duration_sec", "tower_id", "cost"]]
    )
    return {
        "subscriber": _sub_records(row)[0],
        "recent_cdrs": recent.to_dict(orient="records"),
    }


@app.get("/api/alerts")
def alerts(limit: int = 50):
    df = S.scored[S.scored["risk_band"].isin(["high", "medium"])]
    df = df.sort_values("risk_score", ascending=False).head(limit)
    return {"total": int(len(df)), "items": _sub_records(df)}


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Score a raw CDR batch: builds per-subscriber features, returns risk scores."""
    if not req.cdrs:
        raise HTTPException(422, "cdrs list is empty")
    df = _prepare_cdrs(req.cdrs)
    feats = build_features(df)
    scored = S.scorer.score(feats)
    return {"scored_subscribers": _sub_records(scored)}


@app.post("/api/agent/investigate/{subscriber_id}")
async def agent_investigate(subscriber_id: str):
    """Run the LangGraph + Groq investigation agent for one subscriber."""
    row = S.scored[S.scored["subscriber_id"] == subscriber_id]
    if row.empty:
        raise HTTPException(404, f"unknown subscriber {subscriber_id}")
    r = row.iloc[0]
    features = {k: float(r[k]) for k in config.FEATURE_COLUMNS}
    features["plan_type"] = r["plan_type"]
    features["billing_status"] = r["billing_status"]
    started = time.time()
    result = await asyncio.to_thread(
        investigate, subscriber_id, features,
        float(r["risk_score"]), float(r["iso_score"]), float(r["xgb_score"]),
    )
    result["latency_sec"] = round(time.time() - started, 2)
    return result


# ---------------------------------------------------------------- stream simulator

async def _stream_loop():
    """Replays sampled CDRs as a live feed and rescores affected subscribers."""
    rng = np.random.default_rng()
    while True:
        batch = S.cdrs.sample(config.STREAM_BATCH_SIZE, random_state=int(rng.integers(1e9)))
        subs = batch["subscriber_id"].unique().tolist()
        window = S.cdrs[S.cdrs["subscriber_id"].isin(subs)]
        scored = S.scorer.score(build_features(window))
        flagged = scored[scored["risk_band"] != "low"]
        now = pd.Timestamp.utcnow().isoformat()
        for rec in _sub_records(flagged):
            S.stream_events.insert(0, {
                "flagged_at": now,
                "subscriber_id": rec["subscriber_id"],
                "risk_score": rec["risk_score"],
                "risk_band": rec["risk_band"],
            })
        S.stream_events = S.stream_events[:100]
        S.stream_ticks += 1
        S.stream_cdrs_processed += len(batch)
        await asyncio.sleep(config.STREAM_INTERVAL_SECONDS)


@app.post("/api/stream/start")
async def stream_start():
    if S.stream_task and not S.stream_task.done():
        return {"running": True, "message": "stream already running"}
    S.stream_task = asyncio.create_task(_stream_loop())
    return {"running": True, "message": "stream started"}


@app.post("/api/stream/stop")
async def stream_stop():
    if S.stream_task and not S.stream_task.done():
        S.stream_task.cancel()
    return {"running": False, "message": "stream stopped"}


@app.get("/api/stream/status")
def stream_status():
    return {
        "running": S.stream_task is not None and not S.stream_task.done(),
        "ticks": S.stream_ticks,
        "cdrs_processed": S.stream_cdrs_processed,
        "recent_flags": S.stream_events[:20],
    }


# ---------------------------------------------------------------- frontend

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")
