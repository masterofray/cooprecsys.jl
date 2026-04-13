# HybridRecommender — Julia Two-Stage Recommender System

- **Author**: Aryanto
- **Email**: aryanto.dandan@gmail.com
- **Homepage**: https://masterofray.github.io

A production-grade, memory-efficient hybrid recommender system built in Julia, combining **Collaborative Filtering** (Truncated SVD) with **Association Rule Mining** (FP-Growth) for real-time, session-aware recommendations.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Training the Model](#training-the-model)
6. [Running Predictions](#running-predictions)
7. [HTTP API Server](#http-api-server)
8. [Module Reference](#module-reference)
9. [Analytics & Diagnostics](#analytics--diagnostics)
10. [Configuration Reference](#configuration-reference)
11. [Cron / Production Deployment](#cron--production-deployment)
12. [Design Decisions](#design-decisions)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   HybridRecommender Pipeline                    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 1 — Candidate Generation (Collaborative Filter)   │   │
│  │                                                          │   │
│  │  train.csv ──► Sparse Matrix ──► Truncated SVD           │   │
│  │                (user × item)      (d-dim latent space)   │   │
│  │                                                          │   │
│  │  Output: Top-100 candidates per user (CF score)          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 2 — Precision Reranking (FP-Growth)               │   │
│  │                                                          │   │
│  │  Baskets ──► FP-Growth ──► Association Rules             │   │
│  │  (orders)    mining        (antecedent → consequent)     │   │
│  │                                                          │   │
│  │  Score_final = Score_CF × (1 + Lift_FP)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│              Final Ranked Recommendations                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
HybridRecommender/
├── src/
│   ├── HybridRecommender.jl    # Main module + train_pipeline()
│   ├── Logger.jl               # Structured lifecycle logging
│   ├── DataLoader.jl           # CSV ingestion + sparse matrix builder
│   ├── MatrixFactorization.jl  # Stage 1: Truncated SVD
│   ├── FPGrowth.jl             # Stage 2: FP-Growth + rules
│   ├── Reranker.jl             # Hybrid score fusion
│   ├── Analytics.jl            # Diagnostic visualizations
│   └── PredictAPI.jl           # HTTP prediction API
├── scripts/
│   ├── train.jl                # CLI training entrypoint
│   └── predict.jl              # CLI prediction entrypoint
├── data/
│   └── train.csv               # Training data
├── models/                     # Saved model artifacts (.jls)
├── plots/                      # Diagnostic plots (.png)
├── logs/                       # Timestamped run logs
├── docs/
│   ├── README.md               # ← You are here
│   ├── API.md                  # HTTP API reference
│   ├── TRAINING.md             # Training guide
│   └── MODULES.md              # Module internals
├── Project.toml                # Julia dependencies
├── run_train.sh                # Bash training launcher
└── run_predict.sh              # Bash prediction/server launcher
```

---

## Quick Start

```bash
# 1. Clone and enter project
cd HybridRecommender

# 2. Place your data
cp /path/to/train.csv data/train.csv

# 3. Train (uses all CPU threads)
chmod +x run_train.sh run_predict.sh
./run_train.sh

# 4a. Single prediction
./run_predict.sh single --user-id 12345 --cart "789,456,123"

# 4b. Start HTTP server
./run_predict.sh server --port 8080

# 4c. Curl test
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": 12345, "cart": [789, 456]}'
```

---

## Installation

### Requirements

- **Julia** ≥ 1.9 ([julialang.org](https://julialang.org/downloads/))
- 4 GB RAM minimum (8 GB recommended for large datasets)
- Multi-core CPU (parallelism scales with available threads)

### Dependency Installation

```bash
# Auto-installed by run_train.sh, or manually:
julia --project=. -e "import Pkg; Pkg.instantiate()"
```

Key dependencies:
| Package | Purpose |
|---------|---------|
| `CSV`, `DataFrames` | Data ingestion |
| `SparseArrays`, `LinearAlgebra` | Sparse SVD |
| `Plots` | Diagnostic visualizations |
| `HTTP`, `JSON` | REST API server |
| `ProgressMeter` | Training progress bar |
| `Serialization` | Model persistence |
| `ArgParse` | CLI argument parsing |

---

## Training the Model

See [docs/TRAINING.md](TRAINING.md) for the full guide.

### Bash (recommended)

```bash
./run_train.sh \
  --data         data/train.csv \
  --model-dir    models         \
  --components   50             \
  --min-support  0.01           \
  --min-conf     0.10           \
  --top-k        100            \
  --chunk-size   256            \
  --threads      auto
```

### Julia REPL

```julia
include("src/HybridRecommender.jl")
using .HybridRecommender

svd_model, fp_model = train_pipeline(
    "data/train.csv";
    model_dir      = "models",
    n_components   = 50,
    min_support    = 0.01,
    min_confidence = 0.10,
    top_k_cf       = 100,
    chunk_size     = 256
)
```

Training outputs:
- `models/svd_model.jls` — Truncated SVD latent factors
- `models/fp_model.jls`  — FP-Growth rules + index
- `plots/explained_variance.png`
- `plots/rule_quality_scatter.png`
- `plots/reranking_impact.png`

---

## Running Predictions

See [docs/API.md](API.md) for the full API reference.

### Single User (CLI)

```bash
./run_predict.sh single \
  --user-id 12345 \
  --cart "789,456,123" \
  --top-k 20 \
  --output recommendations.json
```

### Batch (CLI)

```bash
# Prepare input file
cat > requests.json << 'EOF'
{
  "requests": [
    {"user_id": 12345, "cart": [789, 456]},
    {"user_id": 67890, "cart": []}
  ]
}
EOF

./run_predict.sh batch \
  --input  requests.json \
  --output results.json \
  --top-k  20
```

### Julia API (programmatic)

```julia
include("src/HybridRecommender.jl")
using .HybridRecommender.PredictAPI, .HybridRecommender.Reranker

# Load models once
PredictAPI.load_models("models/svd_model.jls", "models/fp_model.jls")

# Single prediction
result = PredictAPI.predict_single(12345, [789, 456, 123])
println(result["recommendations"])

# Batch
results = PredictAPI.predict_batch([
    Dict("user_id" => 12345, "cart" => [789]),
    Dict("user_id" => 67890, "cart" => [])
])
```

---

## HTTP API Server

See [docs/API.md](API.md) for all endpoint specifications.

```bash
# Start server
./run_predict.sh server --port 8080

# Health check
curl http://localhost:8080/health

# Model info
curl http://localhost:8080/model/info

# Predict
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": 12345, "cart": [789, 456], "top_k_final": 10}'

# Batch predict
curl -X POST http://localhost:8080/batch \
  -H "Content-Type: application/json" \
  -d '{"requests": [{"user_id": 12345, "cart": [789]}]}'
```

---

## Module Reference

See [docs/MODULES.md](MODULES.md) for detailed API docs for each `.jl` file.

| Module | Key Exports |
|--------|-------------|
| `Logger` | `log_info`, `log_warn`, `log_error`, `log_stage`, `log_metric` |
| `DataLoader` | `load_data`, `build_sparse_matrix`, `extract_baskets` |
| `MatrixFactorization` | `SVDModel`, `fit_svd!`, `get_top_candidates`, `save_model`, `load_model` |
| `FPGrowth` | `FPModel`, `fit_fpgrowth!`, `get_lift_for_item`, `save_fp_model` |
| `Reranker` | `rerank_candidates`, `batch_rerank`, `RerankerConfig` |
| `Analytics` | `plot_explained_variance`, `plot_rule_quality`, `plot_reranking_impact` |
| `PredictAPI` | `load_models`, `predict_single`, `predict_batch`, `start_server` |

---

## Analytics & Diagnostics

Three diagnostic plots are generated automatically during training:

### 1. Cumulative Explained Variance
`plots/explained_variance.png`
Shows how much variance each SVD component captures. The red dashed line marks 80%; the orange dotted line marks the elbow. Use this to tune `--components`.

### 2. Rule Quality Scatter
`plots/rule_quality_scatter.png`
Plots Support vs Confidence for all mined rules, colour-coded by Lift. High-lift rules in the top-right corner are the most valuable for reranking.

### 3. Reranking Impact Histogram
`plots/reranking_impact.png`
Compares the distribution of item ranks before (CF only) vs after (Hybrid) FP-Growth. A left shift in the "after" distribution means FP-Growth is successfully promoting relevant items.

---

## Configuration Reference

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Data path | `--data` | `data/train.csv` | Input CSV |
| Model dir | `--model-dir` | `models` | Where to save `.jls` files |
| SVD components | `--components` | `50` | Latent dimensions (tune with explained variance plot) |
| Min support | `--min-support` | `0.01` | FP-Growth: fraction of baskets containing itemset |
| Min confidence | `--min-conf` | `0.10` | FP-Growth: rule confidence threshold |
| CF candidates | `--top-k` | `100` | Candidates from Stage 1 |
| Chunk size | `--chunk-size` | `256` | Users per parallel batch |
| Threads | `--threads` | `auto` | Julia thread count |

---

## Cron / Production Deployment

### Nightly Batch Job

```cron
# Run batch recommendations every night at 02:00
0 2 * * * /opt/HybridRecommender/run_predict.sh batch \
  --input  /data/nightly_users.json \
  --output /data/recommendations_$(date +\%Y\%m\%d).json \
  >> /var/log/recommender_cron.log 2>&1
```

### Weekly Retraining

```cron
# Retrain model every Sunday at 00:00
0 0 * * 0 /opt/HybridRecommender/run_train.sh \
  --data /data/train.csv \
  >> /var/log/recommender_train.log 2>&1
```

### Server Watchdog (restart if crashed)

```cron
# Keep API server alive — restart every 5 min if not running
*/5 * * * * pgrep -f "predict.jl.*server" > /dev/null || \
  nohup /opt/HybridRecommender/run_predict.sh server --port 8080 \
  >> /var/log/recommender_server.log 2>&1 &
```

---

## Design Decisions

### Memory Safety
- All data loading uses `CSV.Chunks` streaming — no full materialisation.
- Sparse matrix construction uses triplet accumulation in chunks.
- SVD uses randomised projection (Halko et al. 2011) — never forms A·Aᵀ explicitly.
- `GC.gc()` is called explicitly after every major allocation phase.

### Scalability
- Candidate scoring is chunked over items (configurable `chunk_size`).
- Batch reranking uses `@threads` for parallelism within each user chunk.
- FP-Growth pair counting is parallelised across threads.

### Observability
- Every lifecycle stage emits timestamped, coloured log lines.
- Green progress bar with throughput (users/sec) tracks batch processing.
- `log_metric` emits named scalar values for monitoring integration.

### Production Readiness
- Models serialised with Julia's built-in `Serialization` (zero dependencies).
- HTTP API is stateless — models loaded once at startup, shared across requests.
- All endpoints return structured JSON with timestamps and model version.
