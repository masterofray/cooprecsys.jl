"""
    HybridRecommender.jl — Main Module Entry Point

Two-stage hybrid recommender system:
  Stage 1: Truncated SVD Collaborative Filtering (Candidate Generation)
  Stage 2: FP-Growth Association Rule Mining (Precision Reranking)

Usage
-----
```julia
using HybridRecommender

# Train
train_pipeline("data/train.csv"; model_dir="models/")

# Predict (programmatic)
using .PredictAPI
PredictAPI.load_models("models/svd_model.jls", "models/fp_model.jls")
result = PredictAPI.predict_single(12345, [789, 456])

# Serve HTTP API
PredictAPI.start_server("models/svd_model.jls", "models/fp_model.jls"; port=8080)
```
"""
module HybridRecommender

# Sub-modules
include("Logger.jl")
include("DataLoader.jl")
include("MatrixFactorization.jl")
include("FPGrowth.jl")
include("Reranker.jl")
include("Analytics.jl")
include("PredictAPI.jl")

using .Logger, .DataLoader, .MatrixFactorization, .FPGrowth,
      .Reranker, .Analytics, .PredictAPI

export train_pipeline, Logger, DataLoader, MatrixFactorization,
       FPGrowth, Reranker, Analytics, PredictAPI

"""
    train_pipeline(data_path;
                   model_dir="models",
                   n_components=50,
                   min_support=0.01,
                   min_confidence=0.1,
                   top_k_cf=100,
                   chunk_size=256)

Full end-to-end training pipeline:
1. Load and preprocess data.
2. Build sparse interaction matrix.
3. Fit Truncated SVD (Stage 1).
4. Extract baskets and fit FP-Growth (Stage 2).
5. Save both models to disk.
6. Generate diagnostic plots.
"""
function train_pipeline(data_path::String;
                         model_dir::String="models",
                         n_components::Int=50,
                         min_support::Float64=0.01,
                         min_confidence::Float64=0.1,
                         top_k_cf::Int=100,
                         chunk_size::Int=256)

    mkpath(model_dir)
    mkpath("plots")
    mkpath("logs")

    log_stage("HybridRecommender Training Pipeline")
    log_info("data_path=$(data_path)"; stage="Pipeline")
    log_info("n_components=$(n_components) | min_support=$(min_support) | min_confidence=$(min_confidence)"; stage="Pipeline")

    # ── Step 1: Data Loading ───────────────────────────────────────────────────
    cfg = DataLoader.DataConfig(data_path)
    df  = DataLoader.load_data(cfg)

    # ── Step 2: Sparse Matrix ─────────────────────────────────────────────────
    mat, user_idx, item_idx, users, items = DataLoader.build_sparse_matrix(df)

    # ── Step 3: SVD (Stage 1) ─────────────────────────────────────────────────
    svd_model = MatrixFactorization.SVDModel(n_components)
    MatrixFactorization.fit_svd!(svd_model, mat, user_idx, item_idx, users, items)

    svd_path = joinpath(model_dir, "svd_model.jls")
    MatrixFactorization.save_model(svd_model, svd_path)

    # Release matrix — no longer needed
    mat = nothing; GC.gc()

    # ── Step 4: FP-Growth (Stage 2) ──────────────────────────────────────────
    baskets  = DataLoader.extract_baskets(df)
    fp_model = FPGrowth.FPModel(min_support, min_confidence)
    FPGrowth.fit_fpgrowth!(fp_model, baskets)
    baskets  = nothing; GC.gc()

    fp_path = joinpath(model_dir, "fp_model.jls")
    FPGrowth.save_fp_model(fp_model, fp_path)

    # ── Step 5: Diagnostics ───────────────────────────────────────────────────
    log_stage("Running Diagnostics")
    Analytics.plot_explained_variance(svd_model; outdir="plots")
    Analytics.plot_rule_quality(fp_model;        outdir="plots")

    # Quick sample rerank for the impact histogram
    sample_users = users[1:min(50, length(users))]
    cart_map = Dict(u => [] for u in sample_users)
    rcfg   = Reranker.RerankerConfig(top_k_cf, 20, chunk_size, 1.0)
    sample_results = Reranker.batch_rerank(svd_model, fp_model, sample_users, cart_map; cfg=rcfg)
    Analytics.run_all_diagnostics(svd_model, fp_model, sample_results; outdir="plots")

    log_stage("Training Complete")
    log_info("SVD model → $(svd_path)"; stage="Pipeline")
    log_info("FP model  → $(fp_path)";  stage="Pipeline")
    log_info("Plots     → plots/";       stage="Pipeline")

    return svd_model, fp_model
end

end # module HybridRecommender
