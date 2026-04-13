"""
    Reranker.jl — Hybrid Score Reranking (Stage 1 + Stage 2)

Merges Collaborative Filtering candidate scores with FP-Growth lift
to produce the final Hybrid Score:

    Score_final = Score_CF × (1 + Lift_FP)

Features
--------
- Chunk-based user batch processing with progress bar (green).
- Multi-threaded reranking via @threads.
- Explicit GC after each batch.
- Full observability logging at every step.
"""
module Reranker

using Base.Threads, Dates, Serialization
include("Logger.jl")
include("MatrixFactorization.jl")
include("FPGrowth.jl")
using .Logger, .MatrixFactorization, .FPGrowth

export rerank_candidates, batch_rerank, RerankerConfig, RerankedResult

# ── Types ──────────────────────────────────────────────────────────────────────

struct RerankerConfig
    top_k_cf::Int          # candidates from Stage 1
    top_k_final::Int       # final items to return
    chunk_size::Int        # users per batch
    lift_weight::Float64   # multiplier on FP lift (default 1.0)
end

RerankerConfig() = RerankerConfig(100, 20, 256, 1.0)

struct RerankedResult
    user_id::Any
    items::Vector           # item_ids, ranked
    cf_scores::Vector{Float64}
    fp_lifts::Vector{Float64}
    hybrid_scores::Vector{Float64}
end


# ── Core reranking ─────────────────────────────────────────────────────────────

"""
    rerank_candidates(svd_model, fp_model, user_id, cart;
                      cfg=RerankerConfig()) -> RerankedResult

Single-user hybrid reranking:
1. Retrieve Top-K CF candidates (Stage 1).
2. For each candidate, compute Lift from FP rules (Stage 2).
3. Compute Hybrid Score = CF_score × (1 + weight × Lift).
4. Return top_k_final items sorted by Hybrid Score.
"""
function rerank_candidates(svd_model::SVDModel,
                            fp_model::FPModel,
                            user_id,
                            cart::Vector;
                            cfg::RerankerConfig=RerankerConfig())::RerankedResult

    log_info("Reranking for user=$(user_id) | cart_size=$(length(cart))"; stage="Reranker")

    # Stage 1 — CF candidates
    candidates = get_top_candidates(svd_model, user_id, cfg.top_k_cf)

    if isempty(candidates)
        log_warn("No candidates for user=$(user_id)"; stage="Reranker")
        return RerankedResult(user_id, [], [], [], [])
    end

    n = length(candidates)
    items         = [p.first  for p in candidates]
    cf_scores     = [p.second for p in candidates]
    fp_lifts      = Vector{Float64}(undef, n)

    # Stage 2 — FP Lift (parallelised over candidates)
    @threads for i in 1:n
        fp_lifts[i] = get_lift_for_item(fp_model, cart, items[i])
    end

    # Hybrid Score
    hybrid = cf_scores .* (1.0 .+ cfg.lift_weight .* fp_lifts)

    # Sort descending by hybrid score
    order = sortperm(hybrid; rev=true)
    k     = min(cfg.top_k_final, n)

    return RerankedResult(
        user_id,
        items[order[1:k]],
        cf_scores[order[1:k]],
        fp_lifts[order[1:k]],
        hybrid[order[1:k]]
    )
end


# ── Batch processing with progress bar ────────────────────────────────────────

"""
    batch_rerank(svd_model, fp_model, user_ids, cart_map;
                 cfg=RerankerConfig()) -> Vector{RerankedResult}

Process a list of users in chunks of `cfg.chunk_size`.
Displays a green progress bar to track throughput.
Invokes GC after each chunk to manage memory in long-running jobs.

Arguments
---------
- `user_ids`  : Vector of user IDs to process.
- `cart_map`  : Dict mapping user_id → Vector of cart item IDs.
"""
function batch_rerank(svd_model::SVDModel,
                       fp_model::FPModel,
                       user_ids::Vector,
                       cart_map::Dict;
                       cfg::RerankerConfig=RerankerConfig())::Vector{RerankedResult}

    log_stage("Batch Reranking")
    n_users   = length(user_ids)
    n_chunks  = ceil(Int, n_users / cfg.chunk_size)
    results   = Vector{RerankedResult}(undef, n_users)

    log_info("Processing $(n_users) users in $(n_chunks) chunks of $(cfg.chunk_size)"; stage="BatchRerank")

    # ANSI green progress bar
    _green = "\e[32m"
    _reset = "\e[0m"
    _bold  = "\e[1m"

    bar_width = 40
    t_start   = time()

    for chunk_idx in 1:n_chunks
        start = (chunk_idx - 1) * cfg.chunk_size + 1
        stop  = min(chunk_idx * cfg.chunk_size, n_users)
        chunk_users = user_ids[start:stop]

        # ── Progress bar ────────────────────────────────────────────────────
        pct      = chunk_idx / n_chunks
        filled   = round(Int, pct * bar_width)
        bar      = "$(_green)$(_bold)" * "█"^filled * " "^(bar_width - filled) * "$(_reset)"
        elapsed  = round(time() - t_start; digits=1)
        speed    = round(stop / max(elapsed, 0.001); digits=1)
        print("\r  [$bar] $(lpad(round(Int, pct*100), 3))%  " *
              "chunk $(chunk_idx)/$(n_chunks)  " *
              "$(speed) users/s  elapsed=$(elapsed)s   ")
        flush(stdout)
        # ────────────────────────────────────────────────────────────────────

        # Parallel reranking within chunk
        @threads for idx in 1:length(chunk_users)
            uid  = chunk_users[idx]
            cart = get(cart_map, uid, [])
            results[start + idx - 1] = rerank_candidates(svd_model, fp_model, uid, cart; cfg=cfg)
        end

        log_info("\n  Chunk $(chunk_idx)/$(n_chunks) done (users $(start)–$(stop))"; stage="BatchRerank")
        GC.gc()   # release chunk allocations
    end

    println()  # newline after progress bar
    total_t = round(time() - t_start; digits=2)
    log_metric("batch_total_seconds", total_t; stage="BatchRerank")
    log_metric("throughput_users_per_sec", round(n_users/total_t; digits=1); stage="BatchRerank")

    return results
end

end # module Reranker
