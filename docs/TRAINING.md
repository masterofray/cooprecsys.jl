# Training Guide — HybridRecommender

This document covers the full training pipeline: data preparation, hyperparameter tuning, interpreting diagnostics, and retraining strategies.

---

## Data Format

The system expects a CSV with at minimum these columns:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | Unique user identifier |
| `product_id` | int | Unique product identifier |
| `order_id` | int | Unique order identifier (for basket extraction) |
| `reordered` | 0/1 | Whether the product was reordered (used as implicit rating weight) |

Optional columns (used for richer feature engineering in future versions):
`aisle_id`, `department_id`, `order_number`, `order_dow`, `order_hour_of_day`, `days_since_prior_order`, `add_to_cart_order`

**Implicit feedback model:** The system treats all purchases as positive signals. Rating weight = `1 + reordered` (reorders counted twice to reflect stronger preference).

---

## Training Pipeline Stages

```
train.csv
    │
    ▼  DataLoader.load_data()
    │  • CSV streaming in chunks (CSV.Chunks)
    │  • Filter: users with ≥ min_user_orders, products with ≥ min_prod_orders
    │
    ▼  DataLoader.build_sparse_matrix()
    │  • Builds sparse COO triplets in chunks of 5,000 rows
    │  • Deduplicates via SparseArrays (sums duplicates → order count)
    │
    ▼  MatrixFactorization.fit_svd!()
    │  • Randomised Truncated SVD (Halko 2011)
    │  • Power iterations for accuracy: A ≈ U·Σ·Vᵀ
    │  • Computes cumulative explained variance
    │  • GC.gc() after decomposition
    │
    ▼  DataLoader.extract_baskets()
    │  • Groups by order_id → list of product_id strings
    │
    ▼  FPGrowth.fit_fpgrowth!()
    │  • Counts frequent 1-itemsets and 2-itemsets (parallel @threads)
    │  • Generates association rules with support, confidence, lift
    │  • Builds consequent → rules lookup index
    │  • GC.gc() after pair counting
    │
    ▼  Saves models + generates diagnostic plots
```

---

## Hyperparameter Guide

### `--components` (SVD latent dimensions)

Controls the expressiveness of the latent space. More components = more expressive but slower and more memory.

**How to tune:**
1. Run training with default `--components 50`.
2. Inspect `plots/explained_variance.png`.
3. Find the knee/elbow — the component count where cumulative variance flattens.
4. The orange dotted line marks where variance ≥ 80%.

| Dataset size | Recommended range |
|-------------|------------------|
| < 10K rows  | 20–50            |
| 10K–100K    | 50–100           |
| > 100K      | 100–200          |

### `--min-support`

Minimum fraction of orders that must contain an itemset for it to be "frequent".

- **Too high (e.g. 0.05):** Very few rules, only trivially popular items boosted.
- **Too low (e.g. 0.001):** Millions of rules, slower inference, noisy lifts.
- **Recommended starting point:** `0.01` (1% of baskets).

Check `plots/rule_quality_scatter.png` — if the scatter is sparse, lower the threshold.

### `--min-conf`

Minimum confidence for a rule `A → B`:  P(B | A).

- **Too high (e.g. 0.5):** Only very strong rules kept; may miss many relevant associations.
- **Too low (e.g. 0.01):** Many spurious rules.
- **Recommended:** `0.10` to start; tune based on rule count from `[METRIC] total_rules`.

### `--top-k`

How many candidates Stage 1 returns before Stage 2 reranking.

- Larger top-K = better recall but slower reranking.
- For production with latency constraints, use 50–100.
- For offline batch jobs, 200–500 is feasible.

### `--chunk-size`

Users processed per parallel batch. Adjust based on available RAM:

| RAM | Recommended chunk-size |
|-----|----------------------|
| 4 GB  | 128 |
| 8 GB  | 256 |
| 16 GB | 512 |
| 32 GB | 1024 |

---

## Reading Training Logs

Each stage emits structured log lines:

```
[2024-01-15 10:30:00] [INFO] [DataLoader] Raw dataset: 10000 rows × 20 cols
[2024-01-15 10:30:01] [METRIC] [DataLoader] filtered_rows = 9823
[2024-01-15 10:30:01] [METRIC] [DataLoader] unique_users = 9467
[2024-01-15 10:30:01] [METRIC] [DataLoader] unique_products = 4659
...
[2024-01-15 10:30:05] [METRIC] [SVD-Fit] explained_var_topK = 0.8312
...
[2024-01-15 10:30:12] [METRIC] [FP-Fit] total_rules = 12450
```

**Key metrics to check:**
- `nnz_entries` — non-zero elements; if very low (<0.1% sparsity), data may be too sparse for good CF.
- `explained_var_topK` — total variance explained by your chosen components. Aim for ≥ 0.75.
- `total_rules` — total association rules mined. Healthy range: 1K–100K.

---

## Retraining Strategies

### Full Retraining (recommended for weekly jobs)

```bash
./run_train.sh --data /data/latest_train.csv
```

This rebuilds both models from scratch. Old models are overwritten.

### Versioned Retraining

To keep historical models:

```bash
DATE=$(date +%Y%m%d)
./run_train.sh \
  --data      /data/train_${DATE}.csv \
  --model-dir models/${DATE}

# Point the server to the new models
./run_predict.sh server \
  --svd-model models/${DATE}/svd_model.jls \
  --fp-model  models/${DATE}/fp_model.jls
```

### Incremental Update (future work)

The current architecture does not support online/incremental learning. For production systems with rapidly evolving catalogues, schedule full retraining nightly via cron.

---

## Memory Footprint

Approximate peak memory during training:

| Stage | Memory |
|-------|--------|
| CSV loading (10K rows) | ~50 MB |
| Sparse matrix (9K users × 4.6K items, ~0.2% dense) | ~10 MB |
| SVD U, S, Vt matrices (50 components) | ~20 MB |
| FP-Growth pair counting | ~30 MB |
| **Total peak** | **~150 MB** |

For datasets 100× larger, expect ~5–10 GB peak. Use `--chunk-size` to tune.

---

## Troubleshooting

### "Out of memory" during SVD

Lower `--components`. The randomised SVD overshoots by 10 components internally (Halko oversampling), so requesting 50 components allocates for 60.

### "No rules mined" / `total_rules = 0`

Lower `--min-support` (e.g. `0.005`) and/or `--min-conf` (e.g. `0.05`). This happens when the dataset is very sparse or items rarely co-occur.

### "Unknown user_id" warnings at inference

The user was not present in training data. These users fall back to popular-item recommendations (future enhancement — currently returns empty list).

### Slow training

- Ensure Julia is using multiple threads: `JULIA_NUM_THREADS=auto julia ...`
- Verify with: `julia -e "using Base.Threads; println(nthreads())"`
- Reduce `--chunk-size` if hitting memory limits (which causes GC pressure).
