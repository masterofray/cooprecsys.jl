"""
    predict.jl — CLI entrypoint for single/batch predictions

Usage
-----
    julia --threads auto scripts/predict.jl [options]

Modes
-----
  --mode single   Predict for one user (requires --user-id)
  --mode batch    Predict from a JSON file of requests (requires --input)
  --mode server   Start HTTP server

Options
-------
    --mode        single | batch | server  (default: single)
    --svd-model   Path to SVD model .jls  (default: models/svd_model.jls)
    --fp-model    Path to FP model .jls   (default: models/fp_model.jls)
    --user-id     User ID for single prediction
    --cart        Comma-separated product IDs in cart (e.g. 123,456,789)
    --input       Path to JSON batch request file
    --output      Path to write JSON output  (default: stdout)
    --top-k       Number of final recommendations (default: 20)
    --host        Server host (default: 0.0.0.0)
    --port        Server port (default: 8080)
"""

using ArgParse, JSON

function parse_args_predict()
    s = ArgParseSettings(description="HybridRecommender Prediction CLI")
    @add_arg_table! s begin
        "--mode"
            help    = "Prediction mode: single | batch | server"
            default = "single"
        "--svd-model"
            help    = "Path to trained SVD model"
            default = "models/svd_model.jls"
        "--fp-model"
            help    = "Path to trained FP model"
            default = "models/fp_model.jls"
        "--user-id"
            help    = "User ID (single mode)"
            default = nothing
        "--cart"
            help    = "Comma-separated cart product IDs"
            default = ""
        "--input"
            help    = "JSON file with batch requests"
            default = nothing
        "--output"
            help    = "Output JSON file (default: stdout)"
            default = nothing
        "--top-k"
            help    = "Number of recommendations"
            arg_type = Int
            default  = 20
        "--host"
            help    = "Server host"
            default = "0.0.0.0"
        "--port"
            help    = "Server port"
            arg_type = Int
            default  = 8080
    end
    return parse_args(s)
end

# ── Bootstrap ──────────────────────────────────────────────────────────────────
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

include(joinpath(@__DIR__, "..", "src", "HybridRecommender.jl"))
using .HybridRecommender
using .HybridRecommender.PredictAPI
using .HybridRecommender.Reranker

args = parse_args_predict()

svd_path = args["svd-model"]
fp_path  = args["fp-model"]
top_k    = args["top-k"]
mode     = args["mode"]

println("\n\e[1m\e[32m╔══════════════════════════════════════════╗\e[0m")
println(  "\e[1m\e[32m║   HybridRecommender — Predict ($(uppercase(mode)))   ║\e[0m")
println(  "\e[1m\e[32m╚══════════════════════════════════════════╝\e[0m\n")

if mode == "server"
    # ── HTTP server mode ───────────────────────────────────────────────────────
    PredictAPI.start_server(svd_path, fp_path;
                             host=args["host"], port=args["port"])

elseif mode == "single"
    # ── Single user prediction ─────────────────────────────────────────────────
    PredictAPI.load_models(svd_path, fp_path)

    uid_str = args["user-id"]
    isnothing(uid_str) && error("--user-id is required in single mode")

    uid  = tryparse(Int, uid_str)
    uid  = isnothing(uid) ? uid_str : uid

    cart_str = args["cart"]
    cart = isempty(cart_str) ? [] :
           [tryparse(Int, x) !== nothing ? parse(Int,x) : x
            for x in split(cart_str, ",")]

    cfg    = RerankerConfig(100, top_k, 256, 1.0)
    result = PredictAPI.predict_single(uid, cart; cfg=cfg)
    json_out = JSON.json(result, 2)

    out_path = args["output"]
    if isnothing(out_path)
        println(json_out)
    else
        write(out_path, json_out)
        println("Results written to $(out_path)")
    end

elseif mode == "batch"
    # ── Batch prediction from JSON file ───────────────────────────────────────
    PredictAPI.load_models(svd_path, fp_path)

    in_path = args["input"]
    isnothing(in_path) && error("--input is required in batch mode")

    raw_reqs = JSON.parsefile(in_path)
    haskey(raw_reqs, "requests") || error("JSON must have 'requests' key")

    reqs    = raw_reqs["requests"]
    cfg     = RerankerConfig(100, top_k, 256, 1.0)
    results = PredictAPI.predict_batch(reqs; cfg=cfg)

    json_out = JSON.json(Dict("results" => results, "count" => length(results)), 2)
    out_path = args["output"]
    if isnothing(out_path)
        println(json_out)
    else
        write(out_path, json_out)
        println("Batch results ($(length(results))) written to $(out_path)")
    end

else
    error("Unknown mode '$(mode)'. Use: single | batch | server")
end
