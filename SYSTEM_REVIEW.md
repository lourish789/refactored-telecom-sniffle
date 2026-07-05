# System Review — Telecom Fraud & Anomaly Detection Pipeline

*Multi-agent code review (8 finder angles, per-candidate verification) + comparison
against production fraud-management systems. July 2026.*

---

## Part 1 — Verified code-review findings

### Correctness (all independently verified)

| # | Location | Finding | Verdict |
|---|---|---|---|
| 1 | `api/main.py:94` | `CDRIn` defaults `tower_lat`/`tower_lon` to fixed constants; callers omitting coordinates get `max_velocity_kmh = 0`, silently disabling SIM-cloning detection | CONFIRMED |
| 2 | `features.py:44` | `n_days` computed once per batch, not per subscriber → `avg_daily_cost` train/serve skew (a single-day burst scores on a ~7× larger scale than training) | CONFIRMED |
| 3 | `api/main.py:248` | Stream loop runs pandas + model inference synchronously on the asyncio event loop; all API requests stall during every tick | CONFIRMED |
| 4 | `api/main.py:241` | Stream task has no try/except; any tick exception kills it silently with zero diagnostics | CONFIRMED |
| 5 | `agents/fraud_investigator.py:36` | `ChatGroq` has no timeout (langchain-groq passes `timeout=None` → unbounded); a hung Groq call parks a worker thread forever, eventually exhausting the thread pool | CONFIRMED |
| 6 | `agents/fraud_investigator.py:148` | When the report-writing LLM call fails, the fallback report is still labeled `llm_used=True` / `model=llama-3.1-8b-instant` — wrong provenance | CONFIRMED |
| 7 | `scoring.py:37` | `pd.cut` right-inclusive bins put a score of exactly 0.75 in "medium" while the dashboard says "high ≥ 0.75" — boundary accounts under-escalated | CONFIRMED |
| 8 | `api/main.py:209` | `/api/predict` accepts any string as timestamp; unparseable values produce an opaque 500 instead of a 422 | CONFIRMED |
| 9 | `train.py:43`, `scoring.py:30` | Isolation-forest normalization divides by `(max − min)` with no zero-range guard → NaN cascade if scores are ever degenerate | CONFIRMED |
| 10 | `data_generator.py:112` | `_cloning_cdrs` can sample 4 items from an empty list when a Poisson(12) draw is 0 (~1.8e-4 per run) → `ValueError` aborts generation | PLAUSIBLE |

### Maintainability (confirmed, below the severity cap)

| Location | Finding |
|---|---|
| `api/main.py:121` | Premium-destination prefixes hardcoded a second time instead of importing `PREMIUM_DESTS` — editing one list silently skews `premium_dest_ratio` between training and serving |
| `scoring.py:17-23` | `FraudScorer.__init__` reloads the full training parquet and re-scores every row just to derive two normalization floats; persist `raw_min`/`raw_max` at train time instead — serving currently hard-depends on the training data file existing |
| `agents/fraud_investigator.py:83-90` | `_rule_based_classification` re-encodes the same threshold literals as `SIGNAL_RULES`; tuning one copy makes evidence text and classification disagree |
| `api/main.py` (4 sites) | "Is the stream running" boolean recomputed inline in four endpoints instead of one `AppState` property |
| `frontend/index.html:160` | Risk thresholds (0.75 / 0.45) hardcoded in tile copy instead of read from the API — retuning `config.py` silently makes UI text wrong |
| `api/main.py:108` | `SUB_COLS` hand-lists features already in `config.FEATURE_COLUMNS`; new features won't surface in API responses unless both lists are edited |

---

## Part 2 — How production systems do it (and where this project stands)

Commercial fraud-management platforms in live carrier deployments:

| Capability | Subex HyperSense / Mobileum RAID / Amdocs (production) | This project |
|---|---|---|
| **Ingestion** | Kafka/stream processors (Flink) at 10K+ events/sec, <100ms latency | In-process replay loop (deliberate simulation; same scoring path) |
| **Detection models** | Hybrid: rules + supervised (RF/XGB) + unsupervised + **graph analytics for fraud rings** (GAT-COBO-style GNNs); VAE-GAN systems report ~96% accuracy, ~92% precision, ~89% recall on mixed SIM box/IRSF | IsolationForest + XGBoost per-subscriber (no graph layer yet) |
| **Fraud-ring detection** | Graph features over the call network — fraudsters collaborate, so subscriber-level scoring misses rings | Absent — single-subscriber features only |
| **Adaptation** | Continuous retraining, federated learning across operators, shared fraud-pattern databases updated by global client networks | Static one-shot training |
| **Investigation** | Automated "Investigative Agents" (Mobileum RAID) tracing fraud rings across borders in minutes | LangGraph + Groq agent — genuinely the same pattern, at demo scale |
| **Case management** | Analyst workflow: case queues, disposition feedback loops feeding labels back into training | Dashboard alerts only, no feedback capture |
| **Coverage** | Voice + data + digital services + roaming + interconnect settlement | Voice/SMS/data CDRs only |

**Honest positioning:** the architecture (dual detector → risk bands → agent triage →
dashboard) mirrors the production pattern correctly. What separates it from production
is scale plumbing (Kafka/Flink), the graph layer, and the retraining/feedback loop —
not the shape of the pipeline.

---

## Part 3 — Errors to avoid (from production experience)

1. **Trusting ROC-AUC on imbalanced data.** Under severe imbalance nearly any
   classifier hits 0.90–0.99 ROC-AUC; a model can show ROC-AUC 0.957 with PR-AUC
   0.708. Our 1.00/0.998 numbers are a synthetic-data artifact. **Report PR-AUC and
   precision@k for the alert budget instead.**
2. **Optimizing recall while ignoring alert fatigue.** A high false-positive rate
   overwhelms analysts and destroys trust; production teams tune the decision
   threshold to analyst capacity (e.g. flag only at p > 0.8), not to F1.
3. **Ignoring concept drift.** Fraudsters adapt; drift is an in-built property of
   fraud models. Without input/label-drift monitoring, silent decay is guaranteed.
   Our `/api/metrics` is the hook — nothing consumes it yet.
4. **Label leakage & delayed labels.** Fraud labels arrive weeks late and are biased
   (only investigated cases get labels). Never evaluate on same-window labels;
   respect temporal splits when retraining.
5. **Train/serve skew.** Any feature computed differently at serving time than at
   training time (see findings #1, #2 and the premium-prefix duplication) is a
   silent accuracy killer — the #1 preventable production failure mode.
6. **Per-subscriber-only view.** Fraud rings are collaborative; subscriber-level
   anomaly scoring misses coordinated SIM farms that individually look mild.
7. **Unbounded LLM calls in the serving path.** Timeouts, retries with caps, and
   provenance labeling (finding #6) are mandatory before an agent output reaches an
   analyst.

---

## Part 4 — Upgrade roadmap

**Phase 0 — fix the verified findings** (small, high value): thread-off the stream
loop, add try/except + logging around it, timeout on ChatGroq, timestamp/range
validation on `/api/predict`, per-subscriber `n_days`, persist iso min/max at train
time, single source of truth for premium prefixes and thresholds, left-inclusive
risk bins.

**Phase 1 — evaluation credibility:** switch headline metrics to PR-AUC +
precision@alert-budget; temporal train/test split; calibrate XGBoost probabilities
(isotonic); add a `simulate drift` mode to the generator to demo drift detection.

**Phase 2 — streaming for real:** replace the replay loop with Redis Streams or
Kafka (single-broker docker-compose is enough to demo); consumer computes windowed
features incrementally (feature-store pattern) instead of recomputing groupbys.

**Phase 3 — graph layer:** build a caller→callee graph per window; start with cheap
graph features (degree, PageRank, community membership via Louvain) added to the
feature vector — this is most of the GNN benefit at 5% of the complexity; a
GAT-style GNN is the stretch goal.

**Phase 4 — feedback loop / MLOps:** analyst disposition buttons (confirm/dismiss)
on the dashboard writing labels back; scheduled retrain job comparing metrics
against the previous version; Evidently-style drift dashboard on feature
distributions; model registry (MLflow) instead of bare joblib files.

**Phase 5 — agent depth:** give the LangGraph agent tools (query the CDR store,
pull the subscriber's graph neighborhood) instead of a fixed dossier; add a
human-approval gate before any recommended action is "executed".

### Sources
- [Subex — Telecom fraud in 2026: types & AI-first prevention](https://www.subex.com/article/telecom-fraud-in-2026-types-emerging-risks-how-ai-first-prevention-stops-revenue-leakage/)
- [Subex — Fraud Management platform](https://www.subex.com/fraud-management/)
- [Mobileum — RAID platform](https://www.mobileum.com/)
- [Telco Magazine — Top 10 fraud detection tools](https://telcomagazine.com/top10/top-10-fraud-detection-tools)
- [LATRO — Strategic ML for fraud detection in telecom](https://latro.com/blog/strategic-machine-learning-for-fraud-detection-in-telecom/)
- [GAT-COBO: cost-sensitive GNN for telecom fraud detection (arXiv)](https://arxiv.org/pdf/2303.17334)
- [Real-time fraud detection with Kafka/Flink/Redis (reference implementation)](https://github.com/AjayAlluri/realtime-fraud-detection)
- [Why ROC-AUC is misleading for highly imbalanced data (MDPI)](https://www.mdpi.com/2227-7080/14/1/54)
- [ROC-AUC vs precision-recall for imbalanced data](https://machinelearningmastery.com/roc-auc-vs-precision-recall-for-imbalanced-data/)
- [Evidently — Concept drift in production ML](https://www.evidentlyai.com/ml-in-production/concept-drift)
- [Monitoring fraud models in production](https://medium.com/@valeria.verzi1/monitoring-fraud-models-in-production-challenges-and-solutions-477ac99760f8)
