# Module Reference — HybridRecommender

Detailed API documentation for every Julia module in the system.

---

## Logger.jl

Structured, coloured lifecycle logging for every pipeline stage.

### Exports

```julia
log_info(msg; stage="SYSTEM")
log_warn(msg; stage="SYSTEM")
log_error(msg; stage="SYSTEM")
log_stage(title)
log_metric(name, value; stage="METRICS")
```

### Usage

```julia
include("src/Logger.jl")
using .Logger

log_stage("My Pipeline Stage")
log_info("Processing 10,000 rows"; stage="DataLoader")
log_warn("Sparse matrix is very sparse"; stage="DataLoader")
log_metric("n_users", 9467; stage="DataLoader")
log_error("File not found"; stage="IO")
```

### Output format

```
[2024-01-15 10:30:00] [INFO]  [DataLoader] Processing 10,000 rows
[2024-01-15 10:30:01] [WARN]  [DataLoader] Sparse matrix is very sparse
[2024-01-15 10:30:01] [METRIC][DataLoader] n_users = 9467
[2024-01-15 10:30:02] [ERROR] [IO] File not found
```

Colours: INFO=cyan, WARN=yellow, ERROR=red, METRIC=cyan, stage banners=bold blue.

---

## DataLoader.jl

CSV ingestion, filtering, sparse matrix construction, basket extraction.

### Types

```julia
struct DataConfig
    filepath::String
    chunk_size::Int        # rows per streaming chunk (default: 2000)
    min_user_orders::Int   # min purchases per user (default: 2)
    min_prod_orders::Int   # min purchases per product (default: 2)
end

DataConfig(filepath::String)  # uses all defaults
```

### Exports

```julia
load_data(cfg::DataConfig) -> DataFrame
```
Streams CSV in chunks, concatenates, applies user/product frequency filters. Calls `GC.gc()` after each chunk.

```julia
build_sparse_matrix(df::DataFrame)
    -> (mat::SparseMatrixCSC, user_idx::Dict, item_idx::Dict,
        users::Vector, items::Vector)
```
Builds implicit-feedback matrix. Rating = `1 + reordered`. Processes in chunks of 5,000 rows. Duplicate `(user, item)` entries are summed.

```julia
extract_baskets(df::DataFrame) -> Vector{Vector{String}}
```
Groups by `order_id`, returns each order as a `Vector{String}` of product ID strings. Input for FP-Growth.

### Example

```julia
cfg = DataConfig("data/train.csv", 2000, 2, 2)
df  = load_data(cfg)

mat, user_idx, item_idx, users, items = build_sparse_matrix(df)
# mat: SparseMatrixCSC{Float64,Int64}
# user_idx: Dict{Int,Int}  user_id → row
# item_idx: Dict{Int,Int}  product_id → col
# users: Vector{Int}       ordered user IDs
# items: Vector{Int}       ordered product IDs

baskets = extract_baskets(df)
# baskets[1] = ["49302", "11109", "10246", ...]
```

---

## MatrixFactorization.jl

Stage 1: Truncated SVD collaborative filtering.

### Types

```julia
mutable struct SVDModel
    U::Matrix{Float64}         # n_users  × n_components
    S::Vector{Float64}         # n_components (singular values)
    Vt::Matrix{Float64}        # n_components × n_items
    n_components::Int
    user_idx::Dict
    item_idx::Dict
    users::Vector
    items::Vector
    var_ratios::Vector{Float64} # cumulative explained variance
    is_fitted::Bool
end

SVDModel(n_components::Int)    # constructor
```

### Exports

```julia
fit_svd!(model::SVDModel, mat, user_idx, item_idx, users, items) -> SVDModel
```
Fits randomised truncated SVD (Halko 2011). Stores U, S, Vt, explained variance. Calls `GC.gc()` after decomposition. Modifies `model` in-place.

```julia
get_top_candidates(model, user_id, top_k=100; chunk_size=500)
    -> Vector{Pair{Any,Float64}}
```
Returns `top_k` items ranked by latent-space dot-product. Iterates over item chunks to avoid materialising the full score matrix. Returns `[]` for unknown users.

```julia
explained_variance_ratios(model::SVDModel) -> Vector{Float64}
```
Returns the cumulative explained variance per component.

```julia
save_model(model::SVDModel, path::String)
load_model(path::String) -> SVDModel
```
Serialise / deserialise model using Julia's built-in `Serialization`.

### Scoring formula

```
user_vector  = U[u, :] .* S          # scale by singular values
item_scores  = Vt[:, i:j]' * user_vec # chunk dot products
```

### Example

```julia
model = SVDModel(50)
fit_svd!(model, mat, user_idx, item_idx, users, items)

candidates = get_top_candidates(model, 12345, 100)
# candidates[1] = 49302 => 0.923411
# candidates[2] = 13176 => 0.891200
# ...

save_model(model, "models/svd_model.jls")
model2 = load_model("models/svd_model.jls")
```

---

## FPGrowth.jl

Stage 2: FP-Growth association rule mining.

### Types

```julia
struct AssociationRule
    antecedent::Set{String}   # cart items (LHS)
    consequent::String        # candidate item (RHS)
    support::Float64
    confidence::Float64
    lift::Float64
end

mutable struct FPModel
    rules::Vector{AssociationRule}
    rule_index::Dict{String, Vector{AssociationRule}}  # consequent → rules
    min_support::Float64
    min_confidence::Float64
    n_baskets::Int
    is_fitted::Bool
end

FPModel(min_support=0.01, min_confidence=0.1)
```

### Exports

```julia
fit_fpgrowth!(model::FPModel, baskets::Vector{Vector{String}}) -> FPModel
```
Mines frequent 1-itemsets, frequent 2-itemsets (parallelised with `@threads`), and generates association rules. Builds `rule_index` for O(1) lookup by consequent. Calls `GC.gc()` after pair counting.

```julia
get_lift_for_item(model::FPModel, cart::Vector, item_id) -> Float64
```
Returns the maximum lift from all rules whose antecedent intersects the user's cart and whose consequent matches `item_id`. Returns `0.0` if no rule fires.

```julia
save_fp_model(model::FPModel, path::String)
load_fp_model(path::String) -> FPModel
```

### Rule mining algorithm

1. Count frequent 1-itemsets (items appearing in ≥ `min_support × n_baskets` baskets).
2. Count frequent 2-itemsets using multi-threaded parallel pair enumeration.
3. For each frequent pair `{A, B}`, generate rules `A → B` and `B → A`.
4. Compute: `support = count(A,B) / n_baskets`, `confidence = support / support(A)`, `lift = confidence / support(B)`.
5. Keep rules where `confidence ≥ min_confidence`.

### Example

```julia
fp = FPModel(0.01, 0.10)
fit_fpgrowth!(fp, baskets)

# Check lift for product 49302 given cart [789, 456]
lift = get_lift_for_item(fp, [789, 456], 49302)
# lift = 3.14  (product 49302 is 3.14× more likely given the cart)
```

---

## Reranker.jl

Hybrid score fusion (Stage 1 + Stage 2).

### Types

```julia
struct RerankerConfig
    top_k_cf::Int       # candidates from Stage 1 (default: 100)
    top_k_final::Int    # final items returned (default: 20)
    chunk_size::Int     # users per batch (default: 256)
    lift_weight::Float64 # FP lift multiplier (default: 1.0)
end

RerankerConfig()  # all defaults

struct RerankedResult
    user_id::Any
    items::Vector
    cf_scores::Vector{Float64}
    fp_lifts::Vector{Float64}
    hybrid_scores::Vector{Float64}
end
```

### Exports

```julia
rerank_candidates(svd_model, fp_model, user_id, cart;
                  cfg=RerankerConfig()) -> RerankedResult
```
Single-user reranking. Retrieves CF candidates, computes FP lift per candidate with `@threads`, applies hybrid score formula, returns top `cfg.top_k_final` items.

**Hybrid score formula:**
```
Score_final = Score_CF × (1 + lift_weight × Lift_FP)
```

```julia
batch_rerank(svd_model, fp_model, user_ids, cart_map;
             cfg=RerankerConfig()) -> Vector{RerankedResult}
```
Processes users in chunks of `cfg.chunk_size`. Displays a **green progress bar** with real-time throughput. Calls `GC.gc()` after each chunk.

`cart_map` is a `Dict` mapping `user_id → Vector` of cart product IDs.

### Example

```julia
cfg = RerankerConfig(100, 20, 256, 1.0)

# Single user
result = rerank_candidates(svd_model, fp_model, 12345, [789, 456]; cfg=cfg)
result.items          # [49302, 13176, ...]
result.hybrid_scores  # [3.820, 0.891, ...]

# Batch
user_ids = [12345, 67890, 11111]
cart_map = Dict(12345 => [789, 456], 67890 => [], 11111 => [123])
results  = batch_rerank(svd_model, fp_model, user_ids, cart_map; cfg=cfg)
```

---

## Analytics.jl

Diagnostic visualization suite.

### Exports

```julia
plot_explained_variance(model::SVDModel; outdir=".", show=false)
```
Plots cumulative explained variance vs component count. Marks 80% threshold and elbow point. Saves to `<outdir>/explained_variance.png`.

```julia
plot_rule_quality(fp_model::FPModel; outdir=".", show=false)
```
Scatter plot of support vs confidence, colour-coded by lift. Marks min_support and min_confidence thresholds. Saves to `<outdir>/rule_quality_scatter.png`.

```julia
plot_reranking_impact(pre_ranks::Vector{Int}, post_ranks::Vector{Int};
                      outdir=".", show=false)
```
Overlapping histograms of item ranks before and after FP-Growth reranking. Saves to `<outdir>/reranking_impact.png`.

```julia
run_all_diagnostics(svd_model, fp_model, results::Vector{RerankedResult};
                    outdir="plots")
```
Runs all three plots in sequence. Called automatically at end of `train_pipeline()`.

### Fallback behaviour

If `Plots.jl` is unavailable (headless server, missing dependency), all functions gracefully fall back to writing CSV files with the underlying data instead of PNG plots.

---

## PredictAPI.jl

Production HTTP prediction server.

### Module-level state

```julia
_SVD_MODEL::Ref{Union{SVDModel, Nothing}}  # loaded once at startup
_FP_MODEL::Ref{Union{FPModel, Nothing}}    # loaded once at startup
_START_TIME::Ref{DateTime}                  # for uptime reporting
```

### Exports

```julia
load_models(svd_path::String, fp_path::String)
```
Deserialise both models into module-level cache. Thread-safe for read access.

```julia
predict_single(user_id, cart::Vector; cfg=RerankerConfig()) -> Dict
```
Single-user prediction returning a JSON-serialisable Dict. See [API.md](API.md) for response schema.

```julia
predict_batch(requests::Vector{Dict}; cfg=RerankerConfig()) -> Vector{Dict}
```
Batch prediction. Per-user errors are caught and returned as `{"error": "..."}` entries — one error does not abort the batch. Calls `GC.gc()` after completion.

```julia
start_server(svd_path, fp_path; host="0.0.0.0", port=8080)
```
Loads models and starts the HTTP server (blocking). Routes:
- `GET /health`
- `GET /model/info`
- `POST /predict`
- `POST /batch`

### Thread safety

The HTTP server uses Julia's `HTTP.serve` which handles concurrent connections. The `_SVD_MODEL` and `_FP_MODEL` refs are **read-only** after `load_models()` — no locking required. The `rerank_candidates` function is stateless and safe to call concurrently.
