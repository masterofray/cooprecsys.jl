"""
    train.jl — CLI entrypoint for model training

Usage
-----
    julia --threads auto scripts/train.jl [options]

Options
-------
    --data         Path to train.csv  (default: data/train.csv)
    --model-dir    Output dir for models (default: models)
    --components   SVD latent dimensions (default: 50)
    --min-support  FP-Growth min support 0–1 (default: 0.01)
    --min-conf     FP-Growth min confidence 0–1 (default: 0.10)
    --top-k        CF candidate count (default: 100)
    --chunk-size   Users per batch (default: 256)
"""

using ArgParse

function parse_args_train()
    s = ArgParseSettings(description="Train HybridRecommender pipeline")
    @add_arg_table! s begin
        "--data"
            help    = "Path to train.csv"
            default = "data/train.csv"
        "--model-dir"
            help    = "Output directory for saved models"
            default = "models"
        "--components"
            help    = "Number of SVD latent components"
            arg_type = Int
            default  = 50
        "--min-support"
            help    = "FP-Growth minimum support (0–1)"
            arg_type = Float64
            default  = 0.01
        "--min-conf"
            help    = "FP-Growth minimum confidence (0–1)"
            arg_type = Float64
            default  = 0.10
        "--top-k"
            help    = "Number of CF candidates per user"
            arg_type = Int
            default  = 100
        "--chunk-size"
            help    = "Users per processing chunk"
            arg_type = Int
            default  = 256
    end
    return parse_args(s)
end

# ── Bootstrap ──────────────────────────────────────────────────────────────────
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

include(joinpath(@__DIR__, "..", "src", "HybridRecommender.jl"))
using .HybridRecommender

args = parse_args_train()

println("\n\e[1m\e[32m╔══════════════════════════════════════════╗\e[0m")
println(  "\e[1m\e[32m║   HybridRecommender — Training Pipeline  ║\e[0m")
println(  "\e[1m\e[32m╚══════════════════════════════════════════╝\e[0m\n")

train_pipeline(
    args["data"];
    model_dir      = args["model-dir"],
    n_components   = args["components"],
    min_support    = args["min-support"],
    min_confidence = args["min-conf"],
    top_k_cf       = args["top-k"],
    chunk_size     = args["chunk-size"]
)
