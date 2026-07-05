"""LangGraph fraud-investigation agent powered by Groq (llama-3.1-8b-instant).

A three-node graph runs for each flagged subscriber:

    gather_evidence -> classify_fraud -> write_report

- gather_evidence: deterministic — assembles the subscriber's behavioral
  features, model scores, and anomalous signals into an evidence dossier.
- classify_fraud:  LLM — labels the most likely fraud archetype and confidence.
- write_report:    LLM — produces an analyst-ready investigation summary with
  recommended actions.

If GROQ_API_KEY is not set the graph still runs: LLM nodes fall back to a
rule-based classifier so the API and dashboard never break.

LangGraph was chosen over CrewAI here deliberately: the investigation is a
fixed pipeline with typed state, not an open-ended multi-role collaboration —
a deterministic graph is cheaper, faster, and easier to audit.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

import config

_llm = None


def _get_llm():
    """Lazy-init the Groq chat model; return None when no API key is configured."""
    global _llm
    if _llm is None and config.GROQ_API_KEY:
        from langchain_groq import ChatGroq
        _llm = ChatGroq(model=config.GROQ_MODEL, temperature=0.2, max_tokens=1024)
    return _llm


class InvestigationState(TypedDict):
    subscriber_id: str
    features: dict
    risk_score: float
    iso_score: float
    xgb_score: float
    evidence: str
    fraud_type: str
    confidence: str
    report: str
    llm_used: bool


# Signals checked against thresholds when building the evidence dossier
SIGNAL_RULES = [
    ("total_calls", 500, "abnormally high event volume ({v:.0f} events in the window)"),
    ("short_call_ratio", 0.5, "{v:.0%} of voice calls under 60s (SIM box signature)"),
    ("distinct_callees", 1000, "{v:.0f} distinct called numbers (SIM box signature)"),
    ("premium_dest_ratio", 0.05, "{v:.0%} of traffic to premium-rate international ranges (IRSF signature)"),
    ("intl_ratio", 0.3, "{v:.0%} international traffic vs ~3% population norm"),
    ("max_velocity_kmh", 2000, "implied travel speed of {v:.0f} km/h between towers (SIM cloning signature)"),
    ("night_ratio", 0.45, "{v:.0%} of activity at night (00:00-06:00, 22:00-24:00)"),
    ("avg_daily_cost", 200, "average daily charge of {v:.0f} cost units"),
]


def gather_evidence(state: InvestigationState) -> dict:
    f = state["features"]
    signals = [tmpl.format(v=f[k]) for k, thresh, tmpl in SIGNAL_RULES if f.get(k, 0) > thresh]
    if f.get("account_age_days", 999) < 21:
        signals.append(f"account is only {f['account_age_days']:.0f} days old")
    if f.get("billing_status") == "overdue":
        signals.append("billing status is overdue")
    evidence = (
        f"Subscriber {state['subscriber_id']} | plan={f.get('plan_type')} | "
        f"combined risk={state['risk_score']:.2f} "
        f"(isolation-forest anomaly={state['iso_score']:.2f}, xgboost fraud prob={state['xgb_score']:.2f})\n"
        f"Anomalous signals:\n- " + ("\n- ".join(signals) if signals else "none above threshold") + "\n"
        f"Full features: " + ", ".join(f"{k}={f[k]:.2f}" for k in config.FEATURE_COLUMNS if k in f)
    )
    return {"evidence": evidence}


def _rule_based_classification(f: dict) -> tuple[str, str]:
    if f.get("short_call_ratio", 0) > 0.5 and f.get("total_calls", 0) > 500:
        return "simbox", "high"
    if f.get("premium_dest_ratio", 0) > 0.05:
        return "irsf", "high"
    if f.get("account_age_days", 999) < 21 and f.get("intl_ratio", 0) > 0.3:
        return "subscription_fraud", "high"
    if f.get("max_velocity_kmh", 0) > 2000:
        return "sim_cloning", "medium"
    return "unknown", "low"


def classify_fraud(state: InvestigationState) -> dict:
    llm = _get_llm()
    if llm is None:
        ftype, conf = _rule_based_classification(state["features"])
        return {"fraud_type": ftype, "confidence": conf, "llm_used": False}
    prompt = (
        "You are a telecom fraud analyst. Based on the evidence, classify the most likely "
        "fraud type as exactly one of: simbox, irsf, sim_cloning, subscription_fraud, unknown. "
        "Also give a confidence of high, medium, or low.\n"
        "Answer in exactly this format: <fraud_type>|<confidence>\n\n" + state["evidence"]
    )
    try:
        answer = llm.invoke(prompt).content.strip().lower()
        parts = [p.strip() for p in answer.split("|")]
        ftype = parts[0] if parts[0] in config.FRAUD_TYPES + ["unknown"] else "unknown"
        conf = parts[1] if len(parts) > 1 and parts[1] in ("high", "medium", "low") else "medium"
        return {"fraud_type": ftype, "confidence": conf, "llm_used": True}
    except Exception:
        ftype, conf = _rule_based_classification(state["features"])
        return {"fraud_type": ftype, "confidence": conf, "llm_used": False}


FALLBACK_ACTIONS = {
    "simbox": "Suspend outbound service pending KYC re-verification; report SIM box location "
              "(single-tower operation) to the revenue-assurance team.",
    "irsf": "Block premium-rate international destinations for this MSISDN immediately; "
            "review interconnect partner for the destination ranges involved.",
    "sim_cloning": "Force re-authentication / SIM swap for the subscriber; invalidate the current "
                   "IMSI pair and notify the customer through a verified channel.",
    "subscription_fraud": "Suspend the account and hold provisioning of further services; "
                          "flag the identity documents used at onboarding for review.",
    "unknown": "Place the account under enhanced monitoring and route to a human analyst.",
}


def write_report(state: InvestigationState) -> dict:
    llm = _get_llm()
    if llm is None or not state.get("llm_used", False):
        report = (
            f"AUTOMATED INVESTIGATION (rule-based fallback — set GROQ_API_KEY for LLM reports)\n\n"
            f"{state['evidence']}\n\n"
            f"Classification: {state['fraud_type']} (confidence: {state['confidence']})\n"
            f"Recommended action: {FALLBACK_ACTIONS[state['fraud_type']]}"
        )
        return {"report": report}
    prompt = (
        "You are a senior telecom fraud analyst writing a concise investigation report for the "
        "revenue-assurance team. Use the evidence below. Structure: SUMMARY (2 sentences), "
        "KEY EVIDENCE (3-5 bullets), FRAUD CLASSIFICATION (one line), RECOMMENDED ACTIONS "
        "(2-3 numbered items). Be specific and factual; do not invent numbers.\n\n"
        f"Classified fraud type: {state['fraud_type']} (confidence {state['confidence']})\n\n"
        + state["evidence"]
    )
    try:
        report = llm.invoke(prompt).content.strip()
    except Exception:
        report = (f"{state['evidence']}\n\nClassification: {state['fraud_type']}\n"
                  f"Recommended action: {FALLBACK_ACTIONS[state['fraud_type']]}")
    return {"report": report}


def build_graph():
    g = StateGraph(InvestigationState)
    g.add_node("gather_evidence", gather_evidence)
    g.add_node("classify_fraud", classify_fraud)
    g.add_node("write_report", write_report)
    g.set_entry_point("gather_evidence")
    g.add_edge("gather_evidence", "classify_fraud")
    g.add_edge("classify_fraud", "write_report")
    g.add_edge("write_report", END)
    return g.compile()


_graph = None


def investigate(subscriber_id: str, features: dict, risk_score: float,
                iso_score: float, xgb_score: float) -> dict:
    """Run the investigation graph for one flagged subscriber."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    result = _graph.invoke({
        "subscriber_id": subscriber_id,
        "features": features,
        "risk_score": risk_score,
        "iso_score": iso_score,
        "xgb_score": xgb_score,
        "evidence": "",
        "fraud_type": "unknown",
        "confidence": "low",
        "report": "",
        "llm_used": False,
    })
    return {
        "subscriber_id": subscriber_id,
        "fraud_type": result["fraud_type"],
        "confidence": result["confidence"],
        "report": result["report"],
        "llm_used": result["llm_used"],
        "model": config.GROQ_MODEL if result["llm_used"] else "rule-based-fallback",
    }
