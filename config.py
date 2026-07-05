"""Central configuration for the telecom fraud detection pipeline."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# --- Data generation ---
N_SUBSCRIBERS = 2000
N_DAYS = 7
FRAUD_RATE = 0.06  # ~6% of subscribers are fraudulent
RANDOM_SEED = 42

# Fraud archetypes injected into synthetic CDRs
FRAUD_TYPES = ["simbox", "irsf", "sim_cloning", "subscription_fraud"]

# --- Model ---
CDR_PATH = DATA_DIR / "cdrs.parquet"
FEATURES_PATH = DATA_DIR / "subscriber_features.parquet"
ISO_FOREST_PATH = MODEL_DIR / "isolation_forest.joblib"
XGB_PATH = MODEL_DIR / "xgb_classifier.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

# Risk thresholds (combined score in [0, 1])
RISK_HIGH = 0.75
RISK_MEDIUM = 0.45

# --- LLM / Agents ---
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- Streaming ---
STREAM_BATCH_SIZE = 25          # CDRs per simulated stream tick
STREAM_INTERVAL_SECONDS = 2.0   # tick interval

FEATURE_COLUMNS = [
    "total_calls",
    "avg_duration",
    "std_duration",
    "intl_ratio",
    "night_ratio",
    "outgoing_ratio",
    "distinct_callees",
    "callee_reuse_ratio",
    "distinct_towers",
    "max_velocity_kmh",
    "sms_ratio",
    "avg_daily_cost",
    "account_age_days",
    "premium_dest_ratio",
    "short_call_ratio",
]
