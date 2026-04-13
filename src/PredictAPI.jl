"""
    PredictAPI.jl — Production Prediction API

Provides a lightweight HTTP API surface for the Hybrid Recommender.
Loads pre-trained SVD and FP models from disk once at startup,
then serves prediction requests in real time.

Endpoints
---------
  POST /predict     — single-user recommendation
  POST /batch       — batch recommendation for multiple users
  GET  /health      — liveness probe
  GET  /model/info  — model metadata
"""
module PredictAPI

using HTTP, JSON, Dates, Serialization
include("Logger.jl")
include("MatrixFactorization.jl")
include("FPGrowth.jl")
include("Reranker.jl")
using .Logger, .MatrixFactorization, .FPGrowth, .Reranker

export start_server, predict_single, predict_batch, load_models

# ── Model registry (module-level cache) ───────────────────────────────────────
const _SVD_MODEL = Ref{Union{SVDModel, Nothing}}(nothing)
const _FP_MODEL  = Ref{Union{FPModel,  Nothing}}(nothing)
const _START_TIME = Ref{DateTime}(now())


# ── Model loading ──────────────────────────────────────────────────────────────

"""
    load_models(svd_path, fp_path)

Load both models from disk into the module-level cache.
Call once at server startup.
"""
function load_models(svd_path::String, fp_path::String)
    log_stage("Model Loading")
    _SVD_MODEL[] = MatrixFactorization.load_model(svd_path)
    _FP_MODEL[]  = FPGrowth.load_fp_model(fp_path)
    _START_TIME[] = now()
    log_info("Models ready. Server up at $(_START_TIME[])"; stage="PredictAPI")
end


# ── Core prediction functions (usable independently of HTTP) ──────────────────

"""
    predict_single(user_id, cart::Vector; cfg=RerankerConfig())
        -> Dict

Return recommendations for a single user as a Dict (JSON-serialisable).

Input
-----
- `user_id` : any user ID present in the training data.
- `cart`    : current shopping cart as a Vector of product IDs.

Output
------
```json
{
  "user_id": 12345,
  "recommendations": [
    {"item_id": 789, "cf_score": 0.85, "fp_lift": 2.1, "hybrid_score": 2.635},
    ...
  ],
  "model_version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00"
}
```
"""
function predict_single(user_id, cart::Vector;
                         cfg::RerankerConfig=RerankerConfig())::Dict

    svd = _SVD_MODEL[]
    fp  = _FP_MODEL[]

    if isnothing(svd) || isnothing(fp)
        error("Models not loaded. Call load_models() first.")
    end

    log_info("predict_single: user=$(user_id) cart_size=$(length(cart))"; stage="PredictAPI")

    result = rerank_candidates(svd, fp, user_id, cart; cfg=cfg)

    recs = [Dict(
        "item_id"      => result.items[i],
        "cf_score"     => round(result.cf_scores[i];     digits=6),
        "fp_lift"      => round(result.fp_lifts[i];      digits=6),
        "hybrid_score" => round(result.hybrid_scores[i]; digits=6),
        "rank"         => i
    ) for i in 1:length(result.items)]

    return Dict(
        "user_id"         => user_id,
        "recommendations" => recs,
        "model_version"   => "1.0.0",
        "timestamp"       => string(now()),
        "cart_size"       => length(cart)
    )
end


"""
    predict_batch(requests::Vector{Dict}; cfg=RerankerConfig())
        -> Vector{Dict}

Batch prediction for multiple users.

Each element of `requests` must have:
- `"user_id"` : user ID
- `"cart"`    : list of product IDs (may be empty)

Returns a list of prediction dicts in the same order.
"""
function predict_batch(requests::Vector{Dict};
                        cfg::RerankerConfig=RerankerConfig())::Vector{Dict}
    log_info("predict_batch: $(length(requests)) requests"; stage="PredictAPI")

    results = Vector{Dict}(undef, length(requests))
    for (i, req) in enumerate(requests)
        uid  = req["user_id"]
        cart = get(req, "cart", [])
        try
            results[i] = predict_single(uid, cart; cfg=cfg)
        catch e
            log_error("Batch error for user=$(uid): $(e)"; stage="PredictAPI")
            results[i] = Dict(
                "user_id" => uid,
                "error"   => string(e),
                "timestamp" => string(now())
            )
        end
    end

    GC.gc()
    return results
end


# ── HTTP handlers ──────────────────────────────────────────────────────────────

function _json_response(data; status=200)
    HTTP.Response(status,
        ["Content-Type" => "application/json"],
        body=JSON.json(data))
end

function _parse_body(req::HTTP.Request)
    try
        JSON.parse(String(req.body))
    catch e
        nothing
    end
end


"""Health check endpoint."""
function _handle_health(req::HTTP.Request)
    uptime_sec = round(Int, (now() - _START_TIME[]).value / 1000)
    data = Dict(
        "status"    => "ok",
        "uptime_s"  => uptime_sec,
        "svd_ready" => !isnothing(_SVD_MODEL[]),
        "fp_ready"  => !isnothing(_FP_MODEL[]),
        "timestamp" => string(now())
    )
    return _json_response(data)
end


"""Model info endpoint."""
function _handle_model_info(req::HTTP.Request)
    svd = _SVD_MODEL[]
    fp  = _FP_MODEL[]
    data = Dict(
        "svd_components"  => isnothing(svd) ? 0 : svd.n_components,
        "svd_n_users"     => isnothing(svd) ? 0 : length(svd.users),
        "svd_n_items"     => isnothing(svd) ? 0 : length(svd.items),
        "fp_rules"        => isnothing(fp)  ? 0 : length(fp.rules),
        "fp_min_support"  => isnothing(fp)  ? 0 : fp.min_support,
        "fp_min_conf"     => isnothing(fp)  ? 0 : fp.min_confidence,
        "model_version"   => "1.0.0"
    )
    return _json_response(data)
end


"""Single prediction endpoint — POST /predict"""
function _handle_predict(req::HTTP.Request)
    body = _parse_body(req)
    if isnothing(body)
        return _json_response(Dict("error" => "Invalid JSON body"); status=400)
    end
    if !haskey(body, "user_id")
        return _json_response(Dict("error" => "Missing 'user_id'"); status=400)
    end

    uid  = body["user_id"]
    cart = get(body, "cart", [])
    top_k_cf    = get(body, "top_k_cf",    100)
    top_k_final = get(body, "top_k_final", 20)
    cfg = RerankerConfig(top_k_cf, top_k_final, 256, 1.0)

    try
        result = predict_single(uid, cart; cfg=cfg)
        return _json_response(result)
    catch e
        log_error("Predict error: $(e)"; stage="PredictAPI")
        return _json_response(Dict("error" => string(e)); status=500)
    end
end


"""Batch prediction endpoint — POST /batch"""
function _handle_batch(req::HTTP.Request)
    body = _parse_body(req)
    if isnothing(body) || !haskey(body, "requests")
        return _json_response(Dict("error" => "Expected {requests: [...]}"); status=400)
    end

    reqs = [Dict(String(k) => v for (k,v) in r) for r in body["requests"]]
    results = predict_batch(reqs)
    return _json_response(Dict("results" => results, "count" => length(results)))
end


"""Router."""
function _router(req::HTTP.Request)
    log_info("$(req.method) $(req.target)"; stage="HTTP")

    if req.target == "/health" && req.method == "GET"
        return _handle_health(req)
    elseif req.target == "/model/info" && req.method == "GET"
        return _handle_model_info(req)
    elseif req.target == "/predict" && req.method == "POST"
        return _handle_predict(req)
    elseif req.target == "/batch" && req.method == "POST"
        return _handle_batch(req)
    else
        return _json_response(Dict("error" => "Not Found"); status=404)
    end
end


# ── Server startup ─────────────────────────────────────────────────────────────

"""
    start_server(svd_path, fp_path; host="0.0.0.0", port=8080)

Load models and start the HTTP server (blocking call).

Example
-------
```julia
include("src/PredictAPI.jl")
using .PredictAPI
start_server("models/svd_model.jls", "models/fp_model.jls"; port=8080)
```
"""
function start_server(svd_path::String, fp_path::String;
                       host::String="0.0.0.0", port::Int=8080)
    load_models(svd_path, fp_path)
    log_stage("HTTP Server Starting")
    log_info("Listening on http://$(host):$(port)"; stage="PredictAPI")
    log_info("Endpoints: GET /health  GET /model/info  POST /predict  POST /batch"; stage="PredictAPI")

    HTTP.serve(_router, host, port)
end

end # module PredictAPI
