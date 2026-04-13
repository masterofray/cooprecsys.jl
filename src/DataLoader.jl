"""
    DataLoader.jl — CSV ingestion & sparse interaction-matrix builder

Responsibilities
----------------
- Stream train.csv in configurable chunk sizes (no full materialisation).
- Build a sparse user×item rating matrix (implicit: value = reorder count).
- Return index mappings: user_id → row, product_id → col.
- Provide basket extraction for FP-Growth input.
"""
module DataLoader

using CSV, DataFrames, SparseArrays, Statistics
include("Logger.jl")
using .Logger

export load_data, build_sparse_matrix, extract_baskets, DataConfig

# ── Configuration ──────────────────────────────────────────────────────────────
struct DataConfig
    filepath::String
    chunk_size::Int       # rows per streaming chunk
    min_user_orders::Int  # filter low-activity users
    min_prod_orders::Int  # filter low-frequency products
end

DataConfig(filepath::String) = DataConfig(filepath, 2000, 2, 2)

# ── Internal helpers ───────────────────────────────────────────────────────────
"""Build consecutive integer index from a vector of IDs."""
function _build_index(ids::AbstractVector)
    unique_ids = sort(unique(ids))
    Dict(id => i for (i, id) in enumerate(unique_ids))
end

# ── Public API ─────────────────────────────────────────────────────────────────

"""
    load_data(cfg::DataConfig) -> DataFrame

Load CSV in chunks, concatenate, and apply quality filters.
Logs row counts before/after filtering.
"""
function load_data(cfg::DataConfig)::DataFrame
    log_stage("Data Loading")
    log_info("Reading $(cfg.filepath)"; stage="DataLoader")

    # CSV.Chunks for memory-safe streaming
    chunks = CSV.Chunks(cfg.filepath; ntasks=1)
    frames = DataFrame[]
    total_rows = 0

    for chunk in chunks
        df_chunk = DataFrame(chunk)
        total_rows += nrow(df_chunk)
        push!(frames, df_chunk)
        log_info("  Loaded chunk: $(nrow(df_chunk)) rows (cumulative: $(total_rows))"; stage="DataLoader")
        GC.gc()   # release chunk buffer immediately
    end

    df = vcat(frames...)
    frames = nothing
    GC.gc()

    log_info("Raw dataset: $(nrow(df)) rows × $(ncol(df)) cols"; stage="DataLoader")

    # ── Filter ─────────────────────────────────────────────────────────────────
    user_counts   = combine(groupby(df, :user_id),    nrow => :cnt)
    prod_counts   = combine(groupby(df, :product_id), nrow => :cnt)

    valid_users   = user_counts[user_counts.cnt .>= cfg.min_user_orders,  :user_id]
    valid_prods   = prod_counts[prod_counts.cnt  .>= cfg.min_prod_orders,  :product_id]

    df = df[in.(df.user_id,   Ref(Set(valid_users))), :]
    df = df[in.(df.product_id, Ref(Set(valid_prods))), :]

    log_metric("filtered_rows",     nrow(df);     stage="DataLoader")
    log_metric("unique_users",      length(valid_users); stage="DataLoader")
    log_metric("unique_products",   length(valid_prods); stage="DataLoader")

    return df
end


"""
    build_sparse_matrix(df::DataFrame)
        -> (matrix::SparseMatrixCSC, user_idx, item_idx, users, items)

Construct a sparse implicit-feedback matrix where
  M[u, i] = number of times user u purchased product i.

Returns the matrix plus forward and reverse index lookups.
Uses chunk iteration to avoid large intermediate allocations.
"""
function build_sparse_matrix(df::DataFrame)
    log_stage("Sparse Matrix Construction")

    user_idx = _build_index(df.user_id)
    item_idx = _build_index(df.product_id)

    n_users = length(user_idx)
    n_items = length(item_idx)

    log_info("Matrix shape: $(n_users) users × $(n_items) items"; stage="DataLoader")

    # Accumulate triplets in chunks to bound peak memory
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    sizehint!(rows, nrow(df))
    sizehint!(cols, nrow(df))
    sizehint!(vals, nrow(df))

    CHUNK = 5_000
    for start in 1:CHUNK:nrow(df)
        stop = min(start + CHUNK - 1, nrow(df))
        sub  = df[start:stop, :]
        for row in eachrow(sub)
            push!(rows, user_idx[row.user_id])
            push!(cols, item_idx[row.product_id])
            # Implicit rating: 1 + reordered flag (reorders weighted higher)
            push!(vals, 1.0 + Float64(row.reordered))
        end
        GC.gc(false)   # minor GC only — cheap
    end

    # Build sparse matrix; duplicate (u,i) entries are summed by SparseArrays
    mat = sparse(rows, cols, vals, n_users, n_items)
    log_metric("nnz_entries", nnz(mat); stage="DataLoader")
    log_metric("sparsity_pct",
        round(100.0 * (1.0 - nnz(mat) / (n_users * n_items)); digits=4);
        stage="DataLoader")

    users = sort(collect(keys(user_idx)), by=k -> user_idx[k])
    items = sort(collect(keys(item_idx)), by=k -> item_idx[k])

    return mat, user_idx, item_idx, users, items
end


"""
    extract_baskets(df::DataFrame) -> Vector{Vector{String}}

Group purchases by order_id and return each order's product list
as a basket of String item-IDs, ready for FP-Growth.
"""
function extract_baskets(df::DataFrame)::Vector{Vector{String}}
    log_info("Extracting baskets for FP-Growth…"; stage="DataLoader")
    grouped  = groupby(df, :order_id)
    baskets  = [string.(sub.product_id) for sub in grouped]
    log_metric("num_baskets", length(baskets); stage="DataLoader")
    return baskets
end

end # module DataLoader
