# ML Observability Platform

> Real-time drift detection, pipeline graph anomaly propagation, and LLM-powered root cause analysis for deployed ML models.

Most deployed ML systems are flying blind. The model ships, someone glances at aggregate accuracy occasionally, and degradation gets caught by an angry user rather than an alert. This platform builds the monitoring infrastructure that makes that situation impossible.

The differentiating contribution is a **GNN-based pipeline DAG anomaly detector**: your ML pipeline is modelled as a directed acyclic graph, and a GraphSAGE network learns what "normal" looks like across the entire pipeline jointly. When an upstream node's distribution shifts, the GNN propagates that signal forward to predict which downstream components are at risk - *before their accuracy metrics have visibly moved*. That is the thing production teams actually want and almost no monitoring tool gives them.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Component Deep Dives](#component-deep-dives)
  - [Instrumentation SDK](#1-instrumentation-sdk)
  - [Ingestion API](#2-ingestion-api)
  - [Drift Detection Engine](#3-drift-detection-engine)
  - [Pipeline DAG Anomaly Detector](#4-pipeline-dag-anomaly-detector)
  - [Alert Manager](#5-alert-manager)
  - [LLM Explainer](#6-llm-explainer)
  - [Dashboard](#7-dashboard)
- [Database Schema](#database-schema)
- [Setup and Running](#setup-and-running)
- [Instrumenting Your Own Model](#instrumenting-your-own-model)
- [Configuring the Pipeline DAG](#configuring-the-pipeline-dag)
- [Drift Thresholds](#drift-thresholds)
- [GNN Training](#gnn-training)
- [Demo Models](#demo-models)
- [The Demo Sequence](#the-demo-sequence)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Deployment Options](#deployment-options)
- [Known Limitations](#known-limitations)

---

## What It Does

You instrument a deployed ML model with a lightweight SDK (three lines of code). From that point:

1. Every inference is logged asynchronously - input features, prediction, confidence, latency
2. Ground truth labels are collected asynchronously when they arrive, days or weeks later
3. A drift detection engine runs on a configurable schedule, comparing incoming distributions against a 14-day reference window using KS tests, PSI, chi-squared, and KL divergence
4. The pipeline DAG anomaly detector receives drift signals and propagates risk scores forward through the pipeline graph, predicting which downstream components are at risk before their own metrics move
5. When risk scores breach thresholds, the alert manager fires with deduplication (one alert per issue per hour), severity scoring, and routing to Slack, email, or webhook
6. The LLM explainer calls Llama 3.1 70B via Groq, passing all active drift statistics, DAG risk scores, and alert context; it returns plain-English root cause analysis and numbered remediation steps
7. Everything is visible in a real-time Streamlit dashboard: live throughput counters, per-feature drift gauges, an interactive pipeline DAG coloured by risk score, alert feed with LLM explanations, and model version timeline

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INSTRUMENTATION SDK                          │
│                                                                     │
│   from sdk import Monitor                                           │
│   monitor = Monitor(model_id="churn_v2", api_endpoint="...", ...)   │
│   monitor.log(features=X, prediction=y_hat)                        │
│                                                                     │
│   • Daemon flush thread - batches writes every 5 seconds            │
│   • Batch size 50 records per POST                                  │
│   • Local buffer retains records if API is unreachable              │
│   • Returns inference_id UUID for ground-truth linkage              │
└────────────────────────────┬────────────────────────────────────────┘
                             │  POST /inferences
┌────────────────────────────▼────────────────────────────────────────┐
│                         INGESTION API                               │
│                                                                     │
│   FastAPI + uvicorn                                                 │
│   POST /inferences       - batch inference records                  │
│   POST /ground-truth     - delayed label collection                 │
│   POST /pipeline-event   - upstream data pipeline events            │
│   GET  /health           - liveness probe                           │
│                                                                     │
│   PostgreSQL - inference_logs, drift_results, alerts (persistent)   │
│   Redis      - live throughput counters, rolling accuracy (ephemeral)│
│                                                                     │
│   APScheduler background task runs DriftEngine every 15 minutes    │
└─────────────┬──────────────────────────────────────────────────────┘
              │
┌─────────────▼──────────────────────────────────────────────────────┐
│                     DRIFT DETECTION ENGINE                         │
│                                                                     │
│   Reference window: 14 days of historical inferences               │
│   Current window:   last 15 minutes of inferences                  │
│   Minimum sample size: 30 per feature before any test fires        │
│                                                                     │
│   Numeric features (int/float):                                    │
│   • Kolmogorov-Smirnov two-sample test (p < 0.05 → WARNING)        │
│   • Population Stability Index with Laplace smoothing              │
│     PSI ≥ 0.10 → WARNING   PSI ≥ 0.25 → CRITICAL                  │
│     Combined: KS p < 0.001 AND PSI ≥ 0.10 → CRITICAL              │
│                                                                     │
│   Categorical features (str):                                      │
│   • Chi-squared goodness-of-fit test (p < 0.05 → WARNING)         │
│   • New category detection (unseen value → immediate WARNING)      │
│                                                                     │
│   Prediction distribution:                                         │
│   • KL divergence vs. reference output distribution                │
│     KL ≥ 0.10 → WARNING   KL ≥ 0.30 → CRITICAL                   │
│                                                                     │
│   Results written to drift_results table                           │
└──────────┬────────────────────────────┬───────────────────────────┘
           │                            │
┌──────────▼─────────────┐  ┌──────────▼──────────────────────────┐
│  PIPELINE DAG          │  │         ALERT MANAGER               │
│  ANOMALY DETECTOR      │  │                                     │
│                        │  │  Deduplication: 1 alert per         │
│  GraphSAGE GNN         │  │  model_id + alert_type per hour     │
│  3 SAGEConv layers     │  │                                     │
│  in → 64 → 64 → 32     │  │  Severity scoring (0.0–1.0):        │
│  Reconstruction head   │  │  CRITICAL if score ≥ 0.8 or any    │
│  32 → in_channels      │  │    critical-severity drift          │
│                        │  │  WARNING  if score ≥ 0.4 or any    │
│  Trained on daily      │  │    warning-severity drift           │
│  node snapshots:       │  │  +0.1 bonus if >3 simultaneous     │
│  8 features per node   │  │    signals fire together            │
│  (mean_pred, std_pred, │  │                                     │
│  p25, p75, pos_rate,   │  │  Routes to: Slack webhook,          │
│  n_norm, mean_conf,    │  │  email (SMTP), REST callback        │
│  std_conf)             │  └──────────┬──────────────────────────┘
│                        │             │
│  Anomaly score =       │  ┌──────────▼──────────────────────────┐
│  per-node MSE          │  │        LLM EXPLAINER                │
│  reconstruction error  │  │                                     │
│                        │  │  Llama 3.1 70B via Groq API         │
│  Fallback (no model):  │  │  max_tokens=800, temperature=0.2    │
│  simple DAG traversal  │  │  diskcache TTL: 1 hour              │
│  with 0.7 decay factor │  │                                     │
│  per hop               │  │  Input: drift stats + DAG risk      │
└──────────┬─────────────┘  │  scores + alert context            │
           └────────────────►                                     │
                            │  Output: root cause + pipeline      │
                            │  impact + remediation steps         │
                            └──────────┬──────────────────────────┘
                                       │
                            ┌──────────▼──────────────────────────┐
                            │           DASHBOARD                 │
                            │                                     │
                            │  Streamlit - http://localhost:8501  │
                            │                                     │
                            │  Page 1: Live overview              │
                            │    Redis counters, throughput       │
                            │  Page 2: Feature drift              │
                            │    Per-feature gauges, time series  │
                            │  Page 3: Pipeline DAG               │
                            │    Interactive graph, risk colours  │
                            │  Page 4: Alerts                     │
                            │    Feed with LLM explanations       │
                            │  Page 5: Model versions             │
                            │    Version timeline, metrics        │
                            └─────────────────────────────────────┘
```

---

## Component Deep Dives

### 1. Instrumentation SDK

**File:** `sdk/monitor.py`, `sdk/buffer.py`

The SDK is designed for zero-friction adoption. The entire integration is three lines:

```python
from sdk import Monitor

monitor = Monitor(model_id="churn_v2", api_endpoint="http://localhost:8000", api_key="your_key")
monitor.log(features=features_dict, prediction=0.83)
```

**How it works internally:**

`Monitor.log()` is non-blocking. It pushes the record into an in-process `LocalBuffer` (a thread-safe deque) and returns the `inference_id` immediately. A daemon thread wakes every `flush_interval` seconds (default: 5s) and drains up to `batch_size` records (default: 50) into a single `POST /inferences` request.

If the API is unreachable, the `_flush()` method catches the `requests.RequestException` and pushes all records back into the buffer - no data is dropped during transient outages.

```python
def _flush(self) -> None:
    batch = self._buffer.drain(self._batch_size)
    if not batch:
        return
    try:
        requests.post(f"{self._endpoint}/inferences", json={"records": batch}, ...)
    except requests.RequestException:
        for record in batch:
            self._buffer.push(record)  # put back on failure
```

**What gets logged per inference:**

| Field | Type | Notes |
|---|---|---|
| `inference_id` | UUID | Auto-generated, returned to caller |
| `model_id` | str | Set at Monitor init |
| `timestamp` | float | `time.time()` at log() call |
| `features` | dict | Raw input feature values |
| `prediction` | float | Model output |
| `confidence` | float | Optional, e.g. `predict_proba` max |
| `latency_ms` | float | Optional, caller-measured |
| `metadata` | dict | Any extra context (user_id, request_id, etc.) |

**Ground truth linkage:**

```python
# Called when the true label arrives - can be hours or days later
monitor.log_ground_truth(inference_id="abc-123", ground_truth=1.0)
```

This fires a direct `POST /ground-truth` (not buffered) since ground truth events are low-frequency and latency-tolerant.

---

### 2. Ingestion API

**File:** `ingestion/api.py`, `ingestion/writer.py`, `ingestion/models.py`

FastAPI application with three write endpoints and a health probe. Runs with uvicorn. The database schema is created via `SQLAlchemy metadata.create_all()` on startup - no Alembic migrations needed for development.

**Endpoints:**

```
POST /inferences        - accepts {"records": [...]} batch
POST /ground-truth      - accepts {"inference_id": "...", "model_id": "...", "ground_truth": 0}
POST /pipeline-event    - accepts upstream data pipeline events for DAG tracking
GET  /health            - returns {"status": "ok"}
```

All write endpoints are authenticated via `X-API-Key` header. The key is set in `.env`.

**Async batch writer:** `writer.py` accumulates incoming records and flushes them to PostgreSQL in configurable batch sizes, preventing write amplification under high throughput.

**Redis counters:** On each `/inferences` write, the API increments Redis keys for live metrics:
- `throughput:{model_id}` - rolling inference count (expires after 60s)
- `accuracy:{model_id}` - rolling accuracy when ground truth is available

The dashboard reads these directly from Redis for the live overview page, avoiding PostgreSQL reads on a hot path.

**Built-in drift scheduler:** The API's lifespan context manager starts an `asyncio` background task that runs `DriftEngine.run()` every `DRIFT_INTERVAL_SECONDS` (default: 900 = 15 minutes). No external scheduler required.

---

### 3. Drift Detection Engine

**File:** `drift/engine.py`, `drift/numeric.py`, `drift/categorical.py`, `drift/prediction.py`, `drift/reference.py`

The engine compares a **current window** (last 15 minutes of inferences) against a **reference window** (last 14 days) for a given `model_id`. Both windows are configurable.

**Minimum sample gate:** Each test requires at least 30 samples in both windows. If either window is smaller, the test returns `drift_detected=False` with severity `INFO` - this prevents false alarms during cold start.

#### Numeric feature drift

Two tests run in parallel per numeric feature:

**Kolmogorov-Smirnov two-sample test** - a non-parametric test that measures the maximum absolute difference between the empirical CDFs of two samples. Sensitive to any kind of distributional shift (mean, variance, shape). Uses `scipy.stats.ks_2samp`.

**Population Stability Index (PSI)** - a stability metric widely used in credit risk modelling. Bins both distributions into 10 equal-width buckets (edges defined by the reference distribution's min/max), applies Laplace smoothing (0.5 pseudocount) to avoid `log(0)`, then computes:

```
PSI = Σ (p_current - p_reference) × ln(p_current / p_reference)
```

PSI interpretation:
- `< 0.10` - no significant shift
- `0.10 – 0.25` - moderate shift, WARNING
- `> 0.25` - major shift, CRITICAL

**Combined severity logic:**

```python
if psi >= psi_critical or (ks_p < 0.001 and psi >= psi_warning):
    severity = "CRITICAL"
elif psi >= psi_warning or ks_p < ks_threshold:
    severity = "WARNING"
else:
    severity = "INFO"
```

The conjunction `KS p < 0.001 AND PSI >= 0.10` catches cases where the KS test is extremely confident but the PSI hasn't yet crossed its own threshold - a belt-and-suspenders approach.

#### Categorical feature drift

**Chi-squared goodness-of-fit** - tests whether the observed category frequencies in the current window could have been drawn from the reference distribution. Uses `scipy.stats.chisquare` after aligning category counts between the two windows.

**New category detection** - any value present in the current window but absent from the reference window immediately triggers a WARNING regardless of frequency. This catches schema changes and novel enum values that the chi-squared test might miss at low frequency.

#### Prediction distribution drift

**KL divergence** - measures the information-theoretic distance between the reference and current prediction score distributions. Both distributions are estimated by histogramming into 20 equal-width bins with Laplace smoothing, then:

```
KL(P || Q) = Σ P(x) × ln(P(x) / Q(x))
```

This catches shifts in the model's output distribution (e.g. the model suddenly predicting more high-confidence positives) that might not yet be visible in feature-level tests.

**Results persistence:** All results - whether drift was detected or not - are written to the `drift_results` table. This gives a complete audit trail and allows the dashboard to plot drift score trends over time even for features that never breached thresholds.

---

### 4. Pipeline DAG Anomaly Detector

**Files:** `graph/dag.py`, `graph/gnn.py`, `graph/train.py`, `graph/propagator.py`

This is the novel component. The idea: an ML pipeline is not a collection of independent components - it is a graph where upstream failures propagate downstream in predictable ways. Treating it as a graph and learning the joint distribution of "normal" behaviour lets us predict cascade risk before it materialises.

#### DAG structure

The pipeline is defined in `configs/dag_definition.yaml` as a set of typed nodes and directed edges. The demo pipeline:

```
raw_data → feature_engineering → feature_store → churn_model_v2 → churn_scores
```

Node types: `data_source`, `transform`, `model`, `output`. Edges represent data flow direction.

#### GNN architecture

`graph/gnn.py` implements a **GraphSAGE** (Hamilton et al., 2017) model using PyTorch Geometric's `SAGEConv`. GraphSAGE was chosen over spectral GCN methods because it uses neighbourhood aggregation via sampling rather than requiring the full graph Laplacian - making it naturally inductive and able to handle DAGs of different sizes without retraining.

Architecture:
```
Layer 0: SAGEConv(in_channels → 64)  + ReLU + Dropout(0.2)
Layer 1: SAGEConv(64 → 64)           + ReLU + Dropout(0.2)
Layer 2: SAGEConv(64 → 32)           [embedding layer]
Score head: Linear(32 → in_channels) [reconstruction head]
```

**Training objective:** The model is trained as an **autoencoder on the graph**: given node features as input, encode to a 32-dimensional embedding via the SAGEConv layers, then reconstruct the original features via the linear head. Training loss is MSE between input and reconstruction.

```python
_, reconstructed = model(data.x, data.edge_index)
loss = F.mse_loss(reconstructed, data.x)
```

**At inference time**, anomaly score per node = per-node reconstruction MSE:

```python
def anomaly_score(self, x, edge_index) -> Tensor:
    _, reconstructed = self(x, edge_index)
    return F.mse_loss(reconstructed, x, reduction="none").mean(dim=1)
```

A high reconstruction error on a node means its current feature vector is unusual relative to what the GNN learned to expect from historical "normal" runs - i.e. that node is anomalous.

#### Node features

Each node is represented by 8 daily aggregate statistics computed from the `inference_logs` table:

| Index | Feature | Description |
|---|---|---|
| 0 | `mean_pred` | Mean prediction score for the day |
| 1 | `std_pred` | Standard deviation of predictions |
| 2 | `p25` | 25th percentile of predictions |
| 3 | `p75` | 75th percentile of predictions |
| 4 | `pos_rate` | Fraction of predictions > 0.5 |
| 5 | `n_norm` | Inference count normalised by maximum daily count |
| 6 | `mean_conf` | Mean confidence score (falls back to mean_pred) |
| 7 | `std_conf` | Std dev of confidence (falls back to 0.5 × std_pred) |

#### Fallback propagation

Before `models/pipeline_gnn.pt` exists (i.e. before the GNN has been trained), `RiskPropagator` automatically falls back to a simple threshold-based propagation:

```python
def _simple_propagate(self, node_features):
    # Each node's score propagates to downstream nodes with 0.7 decay per hop
    for d_id in self._dag.downstream(node_id):
        scores[d_id] = max(scores[d_id], scores[node_id] * 0.7)
```

This means the demo works correctly even without a trained GNN - risk still propagates, just linearly rather than via learned graph representations.

---

### 5. Alert Manager

**Files:** `alerts/manager.py`, `alerts/severity.py`, `alerts/router.py`

#### Deduplication

The `AlertManager._should_fire()` method queries the `alerts` table for any unresolved alert with the same `model_id` and `alert_type` created within the last hour. If one exists, the new alert is suppressed. This prevents alert storms during sustained drift events.

```python
def _should_fire(self, model_id, alert_type) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    recent = session.query(Alert).filter(
        Alert.model_id == model_id,
        Alert.alert_type == alert_type,
        Alert.created_at >= cutoff,
        Alert.resolved == False,
    ).first()
    return recent is None
```

#### Severity scoring

`compute_severity()` produces a float score in `[0, 1]` and a level:

| Condition | Score floor |
|---|---|
| Any CRITICAL drift result | 0.9 |
| Any WARNING drift result | 0.5 |
| Any DAG node with risk > 0.7 | 0.7 |
| >3 simultaneous signals | +0.1 bonus |

Level assignment: `CRITICAL` if score ≥ 0.8 or any critical drifts; `WARNING` if score ≥ 0.4 or any warning drifts; `INFO` otherwise.

#### Routing

`AlertRouter` reads `configs/alert_routing.yaml` and dispatches to configured channels:
- **Slack**: HTTP POST to `SLACK_WEBHOOK_URL`
- **Email**: SMTP via `SMTP_HOST` / `SMTP_USER` / `SMTP_PASSWORD`
- **Webhook**: HTTP POST to `ALERT_WEBHOOK_URL` with the full alert JSON

---

### 6. LLM Explainer

**File:** `alerts/explainer.py`

When an alert fires, `explain_alert()` constructs a structured prompt containing:
- All active alerts with severity and message
- All drift results where `drift_detected=True`, with test type, statistic, and p-value
- All DAG node risk scores, sorted descending

This context is sent to **Llama 3.3 70B** (`llama-3.3-70b-versatile`) via the Groq API with `temperature=0.2` (low for deterministic, factual output) and `max_tokens=800`.

The model is prompted to respond in a fixed three-section format:

```
ROOT CAUSE ANALYSIS
[2-3 sentences]

PIPELINE IMPACT
[bullet per at-risk component]

RECOMMENDED ACTIONS
[numbered remediation steps]
```

**Caching:** Responses are stored in a `diskcache.Cache` with a 1-hour TTL. The cache key is a deterministic JSON hash of the alert context (rounded to 1 decimal place for risk scores, 2 decimal places for statistics). This means repeated alerts for the same underlying issue return immediately from cache rather than burning Groq API quota.

**Graceful degradation:** If `GROQ_API_KEY` is missing or the API call fails, the function returns a plain error string rather than raising - the alert is still created and routed, just without an LLM explanation.

---

### 7. Dashboard

**File:** `dashboard/app.py`, `dashboard/pages/`

Five-page Streamlit application:

| Page | Content |
|---|---|
| Overview | Live inference throughput (Redis), rolling accuracy, model version selector |
| Drift | Per-feature drift gauges with KS/PSI/chi-squared scores, time-series of statistic over last 7 days |
| Pipeline | Interactive DAG via `streamlit-agraph`; nodes coloured green→yellow→red by current risk score |
| Alerts | Chronological alert feed; each alert expandable to show full LLM explanation |
| Models | Model version timeline; accuracy/F1/AUC metrics per version per time window |

---

## Database Schema

Three tables, all managed by SQLAlchemy ORM and created via `metadata.create_all()` on API startup.

### `inference_logs`

```sql
id              UUID          PRIMARY KEY
inference_id    VARCHAR       UNIQUE NOT NULL  -- caller-facing ID
model_id        VARCHAR       NOT NULL INDEX
timestamp       TIMESTAMPTZ   NOT NULL INDEX
features        JSONB         NOT NULL         -- raw feature dict
prediction      FLOAT         NOT NULL
confidence      FLOAT
ground_truth    FLOAT                          -- populated via /ground-truth
latency_ms      FLOAT
metadata        JSONB
```

### `drift_results`

```sql
id              UUID          PRIMARY KEY
model_id        VARCHAR       NOT NULL INDEX
computed_at     TIMESTAMPTZ   NOT NULL INDEX   -- server default now()
feature_name    VARCHAR       NOT NULL         -- "__prediction__" for output drift
test_type       VARCHAR       NOT NULL         -- ks | psi | chi2 | kl
statistic       FLOAT         NOT NULL
p_value         FLOAT                          -- NULL for PSI and KL (no p-value)
drift_detected  BOOLEAN       NOT NULL
severity        VARCHAR       NOT NULL         -- INFO | WARNING | CRITICAL
```

### `alerts`

```sql
id              UUID          PRIMARY KEY
model_id        VARCHAR       NOT NULL INDEX
created_at      TIMESTAMPTZ   NOT NULL INDEX   -- server default now()
severity        VARCHAR       NOT NULL         -- INFO | WARNING | CRITICAL
alert_type      VARCHAR       NOT NULL         -- "drift"
title           VARCHAR       NOT NULL
message         TEXT          NOT NULL
details         JSONB                          -- drift_count, dag_risk_scores, reasons
explanation     TEXT                          -- LLM output
resolved        BOOLEAN       DEFAULT false
resolved_at     TIMESTAMPTZ
```

---

## Setup and Running

### Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- A Groq API key - [free at console.groq.com](https://console.groq.com)

### Quick Start

```bash
git clone https://github.com/yourusername/ml-observability-platform
cd ml-observability-platform

# Copy and fill in environment variables
cp .env.example .env
# Set GROQ_API_KEY in .env

# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Activate venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

# Start the ingestion API (run from project root)
set PYTHONPATH=.                # Windows cmd
$env:PYTHONPATH = "."           # Windows PowerShell
export PYTHONPATH=.             # Linux/macOS

python -m ingestion.api

# In a second terminal - seed demo data
python demo/setup_demo.py

# In a third terminal - start the dashboard
streamlit run dashboard/app.py
```

Open `http://localhost:8501`.

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | LLM explainer. Free at console.groq.com |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `API_KEY` | Yes | Authentication key for SDK and HTTP clients |
| `API_HOST` | No | API bind address (default: `0.0.0.0`) |
| `API_PORT` | No | API port (default: `8000`) |
| `DRIFT_INTERVAL_SECONDS` | No | Drift engine schedule (default: `900` = 15min) |
| `SLACK_WEBHOOK_URL` | No | Slack alert routing |
| `SMTP_HOST` | No | Email alert routing |
| `ALERT_WEBHOOK_URL` | No | Webhook alert routing |

---

## Instrumenting Your Own Model

```python
from sdk import Monitor

# Initialise once at application startup
monitor = Monitor(
    model_id="my_churn_model_v2",
    api_endpoint="http://localhost:8000",
    api_key="your_key",
    flush_interval=5.0,   # seconds between batch flushes
    batch_size=50,        # max records per POST
)

# Log every inference - non-blocking, returns immediately
def predict(features: dict) -> float:
    prediction = model.predict_proba([list(features.values())])[0][1]

    inference_id = monitor.log(
        features=features,
        prediction=prediction,
        confidence=prediction,              # optional
        latency_ms=elapsed_ms,             # optional
        metadata={"user_id": "u_123"},     # optional
    )

    return prediction

# Log ground truth when the true label arrives (can be days later)
monitor.log_ground_truth(
    inference_id=inference_id,
    ground_truth=1,   # customer did churn
)
```

The `Monitor` instance holds a daemon thread that flushes in the background. It is safe to share across threads. If the process exits cleanly, the daemon thread exits too - any unflushed records in the local buffer are lost. For production use, call `monitor._flush()` explicitly in a shutdown hook.

---

## Configuring the Pipeline DAG

Edit `configs/dag_definition.yaml` to match your actual pipeline:

```yaml
nodes:
  - id: raw_data
    type: data_source
    description: "Raw customer transaction data"

  - id: feature_engineering
    type: transform
    description: "Feature computation (recency, frequency, monetary)"

  - id: feature_store
    type: data_source
    description: "Pre-computed features from feature store"

  - id: churn_model_v2
    type: model
    description: "XGBoost churn prediction model"

  - id: churn_scores
    type: output
    description: "Churn probability scores served to downstream systems"

edges:
  - from: raw_data
    to: feature_engineering
  - from: feature_engineering
    to: feature_store
  - from: feature_store
    to: churn_model_v2
  - from: churn_model_v2
    to: churn_scores
```

**Node types and their meaning:**

| Type | Description |
|---|---|
| `data_source` | Raw data input or feature store |
| `transform` | Feature engineering, preprocessing |
| `model` | ML model producing predictions |
| `output` | Downstream consumer of model scores |

The GNN is retrained every time the DAG structure changes. After editing the YAML, delete `models/pipeline_gnn.pt` and run `python graph/train.py` with new demo data. Until retraining completes, the fallback propagation handles risk scoring automatically.

---

## Drift Thresholds

Default thresholds apply to all features unless overridden per-feature in `configs/drift_thresholds.yaml`:

```yaml
defaults:
  ks_p_value: 0.05          # KS test significance threshold
  psi_warning: 0.1          # PSI moderate shift
  psi_critical: 0.25        # PSI major shift
  chi2_p_value: 0.05        # Chi-squared significance threshold
  kl_divergence_warning: 0.1
  kl_divergence_critical: 0.3

features:
  customer_tenure_days:     # tighter critical threshold - this feature matters more
    psi_warning: 0.1
    psi_critical: 0.2
  transaction_frequency:    # looser thresholds - naturally higher variance
    ks_p_value: 0.01
    psi_warning: 0.15
    psi_critical: 0.3
```

Per-feature overrides merge with defaults - only the specified keys are overridden.

---

## GNN Training

After seeding demo data, train the GraphSAGE model on the 30-day history:

```bash
python -m graph.train churn_model_v2
```

This queries `inference_logs` for daily aggregated statistics, builds one graph snapshot per day, and trains the autoencoder for 200 epochs with Adam (`lr=1e-3`). Training output:

```
Fetching daily snapshots for 'churn_model_v2'...
Training GNN on 30 daily snapshots...
Epoch   0: loss=0.3421
Epoch  50: loss=0.0187
Epoch 100: loss=0.0043
Epoch 150: loss=0.0021
Epoch 200: loss=0.0018     (not printed - loop ends at 199)
Model saved to models/pipeline_gnn.pt
```

Restart the ingestion API after training so `RiskPropagator` picks up the new model file.

**Training data requirements:** The GNN needs enough "normal" days to learn the baseline distribution. 14 days is the practical minimum; 30 days produces reliable anomaly scores. The model is sensitive to the feature scale - if your model's prediction score distribution is very different from the demo churn model, the 8 node features may need re-normalisation.

---

## Demo Models

The platform ships with six instrumented demo models, each demonstrating a distinct drift scenario. All six are seeded with 30 days of baseline history plus a 2-day drift window by `demo/setup_demo.py`.

| Model ID | Task | Drift Scenario | Key Signal |
|---|---|---|---|
| `churn_model_v2` | Binary classification | Customer cohort changes - tenure collapses from 180 days avg to 14 days | `customer_tenure_days` KS/PSI critical |
| `credit_model_v1` | Binary classification | Economic downturn - income drops 55%, credit scores fall 130 points, DTI doubles | All 5 features drift simultaneously |
| `fraud_detection_v2` | Binary classification | Class imbalance shift - fraud rate jumps from 1% to 15% | `__prediction__` KL divergence + feature drift |
| `demand_forecast_v1` | Regression | Competitor undercuts market - their price drops 45%, breaking price-demand relationship | `competitor_price` PSI critical, concept drift |
| `sentiment_v1` | Binary classification | Data source changes to aggressive short-form platform - caps ratio 6x higher, text 4x shorter | All 6 NLP features drift |
| `recommender_v3` | Binary classification | New low-engagement user cohort onboards - CTR collapses from 12% to 1%, sessions from 7min to 28s | Engagement collapse across all features |

Each model in `demo/models/` integrates the SDK and implements `sample_features(drift=True/False)` so it can be used both for bulk seeding and for live injection.

---

## The Demo Sequence

Run these steps in order for the full end-to-end demonstration:

```bash
# Step 1 - seed 30 days of healthy baseline
python demo/setup_demo.py
# Dashboard: all green, clean drift scores

# Step 2 - inject upstream feature distribution drift
python demo/inject_drift.py --type feature_distribution --node raw_data
# Simulates a schema change in the raw data source

# Step 3 - run the drift engine
python -m drift.engine churn_model_v2
# Or wait 15 minutes for the scheduler to fire

# Step 4 - check the dashboard
# Drift page: raw_data features show elevated KS/PSI scores
# Pipeline page: risk propagated forward - churn_model_v2 flagged BEFORE accuracy moves
# Alerts page: WARNING alert with LLM root cause analysis

# Step 5 - inject label drift (concept drift)
python demo/inject_drift.py --type label_drift --severity high
# Simulates the real world changing - customers churning for different reasons

# Step 6 - run drift engine again
# Alerts page: CRITICAL alert as accuracy metrics start to degrade
```

**The key moment in the demo** is between Steps 3 and 4: the pipeline DAG shows `churn_model_v2` as high-risk *before* its accuracy metric has visibly moved. That gap - catching the signal at `raw_data` and predicting the downstream impact before it hits the model - is the core value proposition.

---

## Repository Structure

```
ml-observability-platform/
│
├── sdk/
│   ├── monitor.py            # Client SDK - Monitor.log() and Monitor.log_ground_truth()
│   ├── buffer.py             # Thread-safe local buffer for offline resilience
│   └── __init__.py
│
├── ingestion/
│   ├── api.py                # FastAPI app - 3 write endpoints + health + drift scheduler
│   ├── models.py             # SQLAlchemy models: InferenceLog, DriftResult, Alert
│   └── writer.py             # Async batch writer to PostgreSQL
│
├── drift/
│   ├── engine.py             # Orchestrator - fetches windows, runs tests, saves results
│   ├── numeric.py            # KS test + PSI with Laplace smoothing
│   ├── categorical.py        # Chi-squared + new category detection
│   ├── prediction.py         # KL divergence on output distribution
│   └── reference.py          # Reference window management (14-day lookback)
│
├── graph/
│   ├── dag.py                # PipelineDAG - loads YAML, manages nodes/edges/risk scores
│   ├── gnn.py                # PipelineGNN - GraphSAGE autoencoder
│   ├── train.py              # Training loop + __main__ entry point
│   └── propagator.py         # RiskPropagator - GNN inference + fallback propagation
│
├── alerts/
│   ├── manager.py            # AlertManager - dedup, severity, persistence, routing
│   ├── severity.py           # SeverityScore computation (0.0–1.0 float + level)
│   ├── router.py             # AlertRouter - Slack, email, webhook dispatch
│   └── explainer.py          # LLM root cause analysis via Groq + diskcache
│
├── dashboard/
│   ├── app.py                # Streamlit app entrypoint + navigation
│   └── pages/
│       ├── overview.py       # Live metrics from Redis
│       ├── drift.py          # Per-feature drift visualisations (Plotly)
│       ├── pipeline.py       # Interactive DAG (streamlit-agraph)
│       ├── alerts.py         # Alert feed with expandable LLM explanations
│       └── models.py         # Model version management and metrics
│
├── demo/
│   ├── setup_demo.py         # Seeds 30 days of synthetic inference history (all 6 models)
│   ├── inject_drift.py       # Injects feature/label drift for demonstration
│   ├── inject_live_drift.py  # Writes current-timestamp drifted records directly to DB
│   └── models/
│       ├── churn_model.py    # Churn prediction - tenure/frequency collapse
│       ├── credit_model.py   # Credit scoring - economic downturn scenario
│       ├── fraud_model.py    # Fraud detection - class imbalance shift (1% to 15%)
│       ├── demand_model.py   # Demand forecasting - competitor undercuts market
│       ├── sentiment_model.py # Sentiment classification - platform migration
│       └── recommender_model.py # Recommendation CTR - new low-engagement cohort
│
├── tests/
│   ├── test_drift.py         # Unit tests for all drift detection methods
│   ├── test_gnn.py           # GNN forward pass and anomaly score tests
│   ├── test_alerts.py        # Alert dedup and severity scoring tests
│   └── test_sdk.py           # SDK buffering and flush tests
│
├── configs/
│   ├── drift_thresholds.yaml # Per-feature KS/PSI/chi2/KL thresholds
│   ├── dag_definition.yaml   # Pipeline nodes and edges
│   └── alert_routing.yaml    # Slack/email/webhook routing config
│
├── models/                   # Saved GNN weights (created by graph/train.py)
│   └── pipeline_gnn.pt
│
├── docker-compose.yml        # postgres:16 + redis:7-alpine + api + dashboard
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI + uvicorn | Async, fast, automatic OpenAPI docs |
| ORM | SQLAlchemy 2.0 | Async session support, JSONB column types |
| Database | PostgreSQL 16 | JSONB for flexible feature storage, strong indexing |
| Cache | Redis 7 | Sub-millisecond live counter reads for dashboard |
| Drift stats | scipy | `ks_2samp`, `chisquare` - battle-tested implementations |
| Numerics | numpy + pandas | Array ops for PSI/KL binning |
| Graph ML | PyTorch Geometric | `SAGEConv` - inductive, works on DAGs without recomputing Laplacian |
| Deep learning | PyTorch 2.1 (CUDA 12.8) | GPU-accelerated GNN training |
| LLM | Groq (Llama 3.3 70B) | Fast inference, free tier, strong instruction following |
| LLM cache | diskcache | Persistent disk cache, 1-hour TTL |
| Dashboard | Streamlit | Rapid development, good Plotly integration |
| Visualisation | Plotly | Interactive drift time-series and gauges |
| DAG viz | streamlit-agraph | Interactive graph with node colouring |
| Infrastructure | Docker + Docker Compose | Single command to start the full stack |
| Testing | pytest | Standard, integrates with CI |

---

## Deployment Options

### Option 1 - Local + ngrok (Best for Interviews)

Run the stack locally and tunnel it to a public URL with one command. No account or credit card needed beyond a free ngrok signup.

```bash
# Start the stack
docker-compose up -d postgres redis
python -m ingestion.api &
python demo/setup_demo.py
streamlit run dashboard/app.py &

# Expose dashboard publicly
ngrok http 8501
```

ngrok prints a URL like `https://abc123.ngrok.io`. Share it during the interview. The tunnel is live as long as your machine is running.

### Option 2 - Render (Always-On, No Credit Card)

Deploy the full stack from GitHub to Render's free tier. Includes managed PostgreSQL and Redis.

1. Push this repo to GitHub
2. Create a new Web Service on `render.com` pointing at the repo
3. Add a PostgreSQL instance and a Redis instance from the Render dashboard
4. Set environment variables (`GROQ_API_KEY`, `DATABASE_URL`, `REDIS_URL`, `API_KEY`)
5. Set the start command to `python -m ingestion.api`
6. Deploy a second service for the dashboard with start command `streamlit run dashboard/app.py`

Free tier services sleep after 15 minutes of inactivity and take ~30 seconds to wake. Sufficient for a portfolio link.

### Option 3 - Azure for Students (Always-On, No Credit Card)

If you have a university email, Azure for Students gives $100 free credit with no credit card required. Enough to run a VM with Docker Compose for months.

```bash
# On the Azure VM after provisioning
git clone https://github.com/yourusername/ml-observability-platform
cd ml-observability-platform
cp .env.example .env   # fill in keys
docker-compose up -d
python demo/setup_demo.py
```

Open ports 8000 and 8501 in the VM's network security group. Best option for a permanent always-on URL.

---

## Known Limitations

**GNN requires historical training data.** The graph anomaly detector needs at least 14 days of "normal" pipeline runs before its anomaly scores are meaningful. For a new pipeline, fall back to threshold-based alerting while history accumulates. The fallback activates automatically when `models/pipeline_gnn.pt` does not exist.

**LLM explanation quality scales with context richness.** With a single drifted feature and a small shift, the Llama explanation will be generic. It becomes genuinely useful when multiple signals fire simultaneously - the more context passed in (features, risk scores, alert history), the more specific and actionable the output.

**PostgreSQL JSONB for features is flexible but not typed.** Feature schemas can change between model versions without a migration, which is convenient for development but means no compile-time guarantees on feature shape. Production use would benefit from a feature registry.

**Groq rate limits under alert storms.** If many alerts fire simultaneously and each triggers a fresh LLM call, you can exhaust Groq's daily free-tier limits quickly. The 1-hour alert dedup and 1-hour diskcache TTL mitigate this substantially, but during active demo drift injection sessions, monitor your Groq API usage dashboard.

**No Alembic migrations.** The schema is created via `metadata.create_all()` on API startup. This is fine for development but means any schema change requires manual intervention in production. Adding Alembic is the first production hardening step.

**Single-process drift engine.** The drift engine runs in a single thread per `model_id`. For deployments with many models, this becomes a bottleneck. The natural fix is a task queue (Celery + Redis) with one worker per model - the `DriftEngine` class is already self-contained and stateless, so parallelisation is straightforward.
