"""
    MatrixFactorization.jl — Stage 1: Truncated SVD Collaborative Filtering

Implements memory-efficient truncated SVD on a sparse user×item matrix.
Projects users and items into a d-dimensional latent space.
Provides chunk-based Top-K candidate retrieval without materialising
the full dense score matrix.
"""
module MatrixFactorization

using SparseArrays, LinearAlgebra, Statistics, Serialization
include("Logger.jl")
using .Logger

export SVDModel, fit_svd!, get_top_candidates, save_model, load_model,
       explained_variance_ratios

# ── Model struct ───────────────────────────────────────────────────────────────
"""
    SVDModel

Stores the result of a truncated SVD decomposition.

Fields
------
- `U`          : n_users  × n_components  (user latent vectors)
- `S`          : n_components             (singular values)
- `Vt`         : n_components × n_items   (item latent vectors transposed)
- `n_components`: rank of the decomposition
- `user_idx`   : Dict{user_id → row_index}
- `item_idx`   : Dict{item_id → col_index}
- `users`      : ordered list of user IDs
- `items`      : ordered list of item IDs
- `var_ratios` : cumulative explained variance per component
"""
mutable struct SVDModel
    U::Matrix{Float64}
    S::Vector{Float64}
    Vt::Matrix{Float64}
    n_components::Int
    user_idx::Dict
    item_idx::Dict
    users::Vector
    items::Vector
    var_ratios::Vector{Float64}
    is_fitted::Bool
end

SVDModel(n_components::Int) = SVDModel(
    Matrix{Float64}(undef, 0, 0),
    Vector{Float64}(),
    Matrix{Float64}(undef, 0, 0),
    n_components,
    Dict(), Dict(), [], [],
    Vector{Float64}(),
    false
)

# ── Truncated SVD via power-iteration (Lanczos-style) ─────────────────────────
"""
    _truncated_svd(A::SparseMatrixCSC, k::Int; n_iter=5, seed=42)

Memory-efficient randomised truncated SVD (Halko et al. 2011).
Never forms A*Aᵀ as a dense matrix; uses sparse matrix-vector products.

Returns (U, S, Vt) where A ≈ U * Diagonal(S) * Vt.
"""
function _truncated_svd(A::SparseMatrixCSC{Float64}, k::Int;
                         n_iter::Int=5, seed::Int=42)
    m, n = size(A)
    rng  = MersenneTwister(seed)

    log_info("  Randomised SVD: $(m)×$(n) → rank $(k), power_iter=$(n_iter)"; stage="SVD")

    # Step 1: Random projection
    Omega = randn(rng, n, k + 10)    # oversample by 10
    Y     = A * Omega                # m × (k+10) — sparse × dense

    # Step 2: Power iteration to improve accuracy
    for i in 1:n_iter
        Y  = A  * (A' * Y)           # still uses sparse ops
        if i % 2 == 0
            Y, _ = qr(Y)             # re-orthogonalise periodically
            Y = Matrix(Y)
        end
        log_info("  Power iteration $(i)/$(n_iter)"; stage="SVD")
        GC.gc(false)
    end

    # Step 3: QR of Y
    Q, _  = qr(Y)
    Q     = Matrix(Q)[:, 1:min(k+10, size(Y,2))]

    # Step 4: Project A into low-dim subspace (still sparse × dense)
    B     = Q' * A                    # (k+10) × n

    # Step 5: Full SVD of tiny B
    Ub, S, Vt = svd(B)

    # Step 6: Lift back
    U = Q * Ub

    # Truncate to requested rank
    U  = U[:, 1:k]
    S  = S[1:k]
    Vt = Vt'[1:k, :]                 # (k × n)

    return U, S, Vt
end


# ── Public API ─────────────────────────────────────────────────────────────────

"""
    fit_svd!(model::SVDModel, mat::SparseMatrixCSC,
             user_idx, item_idx, users, items)

Fit the Truncated SVD model on the sparse interaction matrix.
Computes and stores explained variance ratios.
Explicitly GC's after decomposition.
"""
function fit_svd!(model::SVDModel,
                  mat::SparseMatrixCSC{Float64},
                  user_idx::Dict, item_idx::Dict,
                  users::Vector, items::Vector)

    log_stage("Stage 1: Matrix Factorization (Truncated SVD)")
    log_info("Fitting SVD with $(model.n_components) components"; stage="SVD-Fit")
    log_metric("matrix_nnz", nnz(mat); stage="SVD-Fit")

    # Convert to Float64 if needed
    A = Float64.(mat)

    # Run randomised SVD
    U, S, Vt = _truncated_svd(A, model.n_components)

    model.U        = U
    model.S        = S
    model.Vt       = Vt
    model.user_idx = user_idx
    model.item_idx = item_idx
    model.users    = users
    model.items    = items
    model.is_fitted = true

    # Compute explained variance ratios
    total_var = sum(sv -> sv^2, diag(A' * A); init=0.0)
    if total_var > 0
        ev_ratios = cumsum((S .^ 2) ./ total_var)
        model.var_ratios = ev_ratios
    else
        model.var_ratios = cumsum(S .^ 2) ./ sum(S .^ 2)
    end

    log_metric("explained_var_top1",  round(model.var_ratios[1];    digits=4); stage="SVD-Fit")
    log_metric("explained_var_topK",  round(model.var_ratios[end];  digits=4); stage="SVD-Fit")
    log_info("SVD fit complete. Invoking GC…"; stage="SVD-Fit")

    # Release the working matrix before returning
    A = nothing
    GC.gc()

    return model
end


"""
    explained_variance_ratios(model::SVDModel) -> Vector{Float64}

Return cumulative explained variance per component (for diagnostics).
"""
function explained_variance_ratios(model::SVDModel)::Vector{Float64}
    @assert model.is_fitted "Model must be fitted first."
    return model.var_ratios
end


"""
    get_top_candidates(model::SVDModel, user_id,
                       top_k::Int=100;
                       chunk_size::Int=500) -> Vector{Pair{Any,Float64}}

Retrieve Top-K items for a given user via latent-factor dot-product scores.
Operates chunk-by-chunk over the item space to avoid dense-matrix
materialisation.

Returns a sorted vector of (item_id => score) pairs, descending by score.
"""
function get_top_candidates(model::SVDModel, user_id,
                             top_k::Int=100;
                             chunk_size::Int=500)::Vector

    @assert model.is_fitted "Model must be fitted before candidate generation."

    if !haskey(model.user_idx, user_id)
        log_warn("Unknown user_id=$(user_id); returning empty candidates."; stage="CandidateGen")
        return []
    end

    u_row = model.user_idx[user_id]

    # User latent vector scaled by singular values: û = U[u,:] .* S
    u_vec = model.U[u_row, :] .* model.S   # length d

    n_items  = length(model.items)
    scores   = Vector{Float64}(undef, n_items)

    # Chunk over items
    for start in 1:chunk_size:n_items
        stop = min(start + chunk_size - 1, n_items)
        # Vt[:, start:stop] is (d × chunk) — dot with u_vec gives chunk scores
        scores[start:stop] = model.Vt[:, start:stop]' * u_vec
        GC.gc(false)
    end

    # Top-K indices (partial sort is faster than full sort)
    top_idx = partialsortperm(scores, 1:min(top_k, n_items); rev=true)

    return [model.items[i] => scores[i] for i in top_idx]
end


"""
    save_model(model::SVDModel, path::String)

Serialise model to disk using Julia's built-in Serialization.
"""
function save_model(model::SVDModel, path::String)
    log_info("Saving SVD model → $(path)"; stage="IO")
    open(path, "w") do io
        serialize(io, model)
    end
    log_info("Model saved ($(round(stat(path).size/1024; digits=1)) KB)"; stage="IO")
end


"""
    load_model(path::String) -> SVDModel

Deserialise model from disk.
"""
function load_model(path::String)::SVDModel
    log_info("Loading SVD model ← $(path)"; stage="IO")
    model = open(deserialize, path)
    log_info("Model loaded (components=$(model.n_components))"; stage="IO")
    return model
end

end # module MatrixFactorization
