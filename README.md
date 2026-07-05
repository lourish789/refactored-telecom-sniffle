# Telecom Fraud & Anomaly Detection Pipeline

End-to-end ML system that detects telecom fraud from **Call Detail Records (CDRs)**:
synthetic data generation → feature engineering → dual-model detection
(unsupervised + supervised) → FastAPI serving → live stream scoring →
**LangGraph + Groq AI investigation agent** → web dashboard.

## Fraud archetypes detected

| Archetype | Behavioral signature | Key features |
|---|---|---|
| **SIM box / bypass** | Hundreds of short outgoing local calls/day from one tower, unique callees, night-heavy | `short_call_ratio`, `distinct_callees`, `total_calls` |
| **IRSF** (Intl. Revenue Share Fraud) | Night bursts of long calls to premium international ranges | `premium_dest_ratio`, `intl_ratio`, `avg_daily_cost` |
| **SIM cloning** | Impossible travel — same identity on towers ~40 km apart within minutes | `max_velocity_kmh` |
| **Subscription fraud** | Brand-new account, instant heavy international usage, delinquent billing | `account_age_days`, `intl_ratio`, billing status |

## Architecture

```
data_generator.py     synthetic CDRs (2,000 subscribers, 7 days, ~250K records,
                      6% fraud) with OSS/BSS metadata (plan, billing status)
        │
features.py           15 per-subscriber behavioral features (velocity, night
                      ratio, callee reuse, premium destination ratio, ...)
        │
train.py              IsolationForest (unsupervised, AUC 0.998)
                      + XGBoost classifier (supervised) → models/*.joblib
        │
scoring.py            combined risk score = 0.5·anomaly + 0.5·fraud probability
        │
api/main.py           FastAPI: batch scoring, alerts, live stream simulator,
                      agent endpoint, dashboard hosting
        │
agents/fraud_investigator.py
                      LangGraph 3-node graph (gather_evidence → classify_fraud
                      → write_report) on Groq llama-3.1-8b-instant, with a
                      rule-based fallback when no API key is set
        │
frontend/index.html   dashboard: KPI tiles, hourly volume, risk bands,
                      alert table with one-click AI investigation, live feed
```

**Why LangGraph over CrewAI here:** the investigation is a fixed pipeline with
typed state, not open-ended multi-role collaboration — a deterministic graph is
cheaper per call, faster, and auditable.

## Quickstart

```bash
pip install -r requirements.txt

python data_generator.py    # generate data/cdrs.parquet
python features.py          # build data/subscriber_features.parquet
python train.py             # train + save models, print metrics

# optional: enable LLM reports
set GROQ_API_KEY=gsk_...    # (PowerShell: $env:GROQ_API_KEY="gsk_...")

python -m uvicorn api.main:app --port 8000
```

Open **http://127.0.0.1:8000** for the dashboard, **/docs** for Swagger.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness, model + LLM + stream status |
| GET | `/api/metrics` | training metrics & feature importances |
| GET | `/api/analytics/summary` | dashboard aggregates (risk bands, hourly volume, top risks) |
| GET | `/api/subscribers?risk=high&limit=&offset=` | scored subscribers, filterable by band |
| GET | `/api/subscribers/{id}` | one subscriber's features, scores, recent CDRs |
| GET | `/api/alerts?limit=` | medium/high-risk subscribers ranked by score |
| POST | `/api/predict` | score a raw CDR batch (JSON) — builds features, returns risk per subscriber |
| POST | `/api/agent/investigate/{id}` | run the LangGraph + Groq investigation, returns analyst report |
| POST | `/api/stream/start` / `/api/stream/stop` | control the live CDR replay simulator |
| GET | `/api/stream/status` | ticks, CDRs processed, recently flagged subscribers |
| GET | `/` | dashboard frontend |

### Example: score a CDR batch

```bash
curl -X POST http://127.0.0.1:8000/api/predict -H "Content-Type: application/json" -d '{
  "cdrs": [{
    "subscriber_id": "TEST1", "timestamp": "2026-07-01T02:00:00",
    "call_type": "voice", "direction": "out", "callee_id": "N1",
    "dest_prefix": "+882", "duration_sec": 900, "tower_id": 3, "cost": 60
  }]
}'
# → {"scored_subscribers":[{"subscriber_id":"TEST1","risk_score":0.90,"risk_band":"high",...}]}
```

## Model performance (held-out test set)

| Metric | Value |
|---|---|
| Isolation Forest ROC-AUC (fully unsupervised) | 0.998 |
| XGBoost ROC-AUC / precision / recall (fraud class) | 1.00 / 1.00 / 1.00 |

Synthetic fraud patterns are strongly separable by design; on real CDRs expect
lower numbers — the pipeline structure (dual detector + risk bands + agent
triage) is what transfers.

## Notes

- **Real-time**: the stream simulator replays CDRs through the same scoring
  path a Kafka/Redis consumer would use — swap `_stream_loop` for a consumer
  to go production.
- **MLOps hooks**: `models/training_metrics.json` is versioned per train run;
  `/api/metrics` exposes it for drift dashboards.
- **OSS/BSS awareness**: CDR schema carries plan type and billing status, used
  both as model features and in agent evidence.
