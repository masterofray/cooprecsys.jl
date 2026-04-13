"""
    FPGrowth.jl — Stage 2: FP-Growth Association Rule Mining

Implements the FP-Growth algorithm for mining frequent itemsets and
generating association rules (antecedent → consequent).

Used by the Reranker to boost CF candidates that co-occur with
items in the user's active cart.

Parallelised rule-mining using Julia Distributed / Threads.
"""
module FPGrowth

using Statistics, Serialization, Base.Threads
include("Logger.jl")
using .Logger

export FPModel, AssociationRule, fit_fpgrowth!, get_lift_for_item,
       save_fp_model, load_fp_model

# ── Data structures ────────────────────────────────────────────────────────────

struct AssociationRule
    antecedent::Set{String}   # cart items (LHS)
    consequent::String        # candidate item (RHS)
    support::Float64
    confidence::Float64
    lift::Float64
end

"""FP-Tree node."""
mutable struct FPNode
    item::Union{String, Nothing}
    count::Int
    parent::Union{FPNode, Nothing}
    children::Dict{String, FPNode}
    FPNode(item=nothing, count=0, parent=nothing) =
        new(item, count, parent, Dict{String, FPNode}())
end

"""Trained FP-Growth model."""
mutable struct FPModel
    rules::Vector{AssociationRule}
    # Lookup: consequent_item → rules affecting it (for fast rerank)
    rule_index::Dict{String, Vector{AssociationRule}}
    min_support::Float64
    min_confidence::Float64
    n_baskets::Int
    is_fitted::Bool
end

FPModel(min_support=0.01, min_confidence=0.1) =
    FPModel([], Dict(), min_support, min_confidence, 0, false)


# ── FP-Tree builder ────────────────────────────────────────────────────────────

"""Count item frequencies across all baskets."""
function _count_items(baskets::Vector{Vector{String}},
                       min_sup_count::Int)::Dict{String,Int}
    freq = Dict{String,Int}()
    for basket in baskets
        for item in basket
            freq[item] = get(freq, item, 0) + 1
        end
    end
    # Prune below min_support
    return filter(p -> p.second >= min_sup_count, freq)
end


"""Insert one basket into the FP-tree."""
function _insert_tree!(node::FPNode, items::Vector{String},
                        freq::Dict{String,Int})
    isempty(items) && return
    item = items[1]
    if haskey(node.children, item)
        node.children[item].count += 1
    else
        child = FPNode(item, 1, node)
        node.children[item] = child
    end
    _insert_tree!(node.children[item], items[2:end], freq)
end


"""Build FP-Tree from baskets (returns root node + header table)."""
function _build_fp_tree(baskets::Vector{Vector{String}},
                          freq::Dict{String,Int},
                          min_sup_count::Int)

    root = FPNode()

    for basket in baskets
        # Filter and sort items by descending frequency (FP-Growth standard)
        filtered = filter(i -> haskey(freq, i), basket)
        sort!(filtered; by=i -> -freq[i])
        _insert_tree!(root, filtered, freq)
    end

    return root
end


"""Mine conditional pattern bases from the FP-Tree for a given item."""
function _mine_patterns(node::FPNode, item::String,
                         patterns::Vector{Tuple{Vector{String},Int}})
    for (child_item, child_node) in node.children
        if child_item == item
            # Walk up to collect prefix path
            path  = String[]
            curr  = child_node.parent
            while curr !== nothing && curr.item !== nothing
                push!(path, curr.item)
                curr = curr.parent
            end
            isempty(path) || push!(patterns, (path, child_node.count))
        end
        _mine_patterns(child_node, item, patterns)
    end
end


# ── Rule generation ────────────────────────────────────────────────────────────

"""
    _generate_rules(freq_itemsets, item_freq, n_baskets, min_conf)

Generate association rules from frequent 2-itemsets.
For simplicity and speed we focus on 1→1 and 2→1 rules which are
the most actionable for cart-based reranking.
"""
function _generate_rules(freq_pairs::Dict{Tuple{String,String},Int},
                           item_freq::Dict{String,Int},
                           n_baskets::Int,
                           min_conf::Float64)::Vector{AssociationRule}
    rules = AssociationRule[]

    for ((a, b), ab_count) in freq_pairs
        sup_ab  = ab_count / n_baskets
        sup_a   = get(item_freq, a, 0) / n_baskets
        sup_b   = get(item_freq, b, 0) / n_baskets

        # Rule a → b
        if sup_a > 0
            conf_ab = sup_ab / sup_a
            lift_ab = conf_ab / max(sup_b, 1e-9)
            if conf_ab >= min_conf
                push!(rules, AssociationRule(Set([a]), b, sup_ab, conf_ab, lift_ab))
            end
        end

        # Rule b → a
        if sup_b > 0
            conf_ba = sup_ab / sup_b
            lift_ba = conf_ba / max(sup_a, 1e-9)
            if conf_ba >= min_conf
                push!(rules, AssociationRule(Set([b]), a, sup_ab, conf_ba, lift_ba))
            end
        end
    end

    return rules
end


"""Count co-occurring pairs across baskets (chunk-parallel)."""
function _count_pairs(baskets::Vector{Vector{String}},
                       freq::Dict{String,Int},
                       min_sup_count::Int)::Dict{Tuple{String,String},Int}

    n = length(baskets)
    n_chunks = nthreads()
    chunk_size = max(1, div(n, n_chunks))

    # Thread-local pair counts
    local_counts = [Dict{Tuple{String,String},Int}() for _ in 1:n_chunks]

    @threads for t in 1:n_chunks
        start = (t-1)*chunk_size + 1
        stop  = (t == n_chunks) ? n : t*chunk_size
        lc = local_counts[t]
        for i in start:stop
            items = filter(x -> haskey(freq, x), unique(baskets[i]))
            for j in 1:length(items)
                for k in (j+1):length(items)
                    pair = items[j] < items[k] ?
                           (items[j], items[k]) : (items[k], items[j])
                    lc[pair] = get(lc, pair, 0) + 1
                end
            end
        end
    end

    # Merge thread-local maps
    merged = Dict{Tuple{String,String},Int}()
    for lc in local_counts
        for (pair, cnt) in lc
            merged[pair] = get(merged, pair, 0) + cnt
        end
    end

    # Apply support threshold
    return filter(p -> p.second >= min_sup_count, merged)
end


# ── Public API ─────────────────────────────────────────────────────────────────

"""
    fit_fpgrowth!(model::FPModel, baskets::Vector{Vector{String}})

Mine frequent itemsets and generate association rules.
Uses multi-threaded pair counting; runs GC after heavy allocations.
"""
function fit_fpgrowth!(model::FPModel,
                        baskets::Vector{Vector{String}})

    log_stage("Stage 2: FP-Growth Association Rule Mining")
    n = length(baskets)
    model.n_baskets = n
    min_sup_count = max(2, round(Int, model.min_support * n))

    log_info("Baskets: $(n) | min_support=$(model.min_support) → count≥$(min_sup_count)"; stage="FP-Fit")
    log_info("min_confidence=$(model.min_confidence) | threads=$(nthreads())"; stage="FP-Fit")

    # Step 1: Frequent 1-itemsets
    log_info("Step 1/3 — counting frequent items…"; stage="FP-Fit")
    item_freq = _count_items(baskets, min_sup_count)
    log_metric("frequent_1_itemsets", length(item_freq); stage="FP-Fit")

    # Step 2: Frequent 2-itemsets (parallel pair counting)
    log_info("Step 2/3 — counting frequent pairs (parallel)…"; stage="FP-Fit")
    pair_freq = _count_pairs(baskets, item_freq, min_sup_count)
    log_metric("frequent_2_itemsets", length(pair_freq); stage="FP-Fit")
    GC.gc()

    # Step 3: Generate rules
    log_info("Step 3/3 — generating association rules…"; stage="FP-Fit")
    rules = _generate_rules(pair_freq, item_freq, n, model.min_confidence)
    model.rules = rules
    log_metric("total_rules", length(rules); stage="FP-Fit")

    # Build fast consequent → rules index
    idx = Dict{String, Vector{AssociationRule}}()
    for r in rules
        if !haskey(idx, r.consequent)
            idx[r.consequent] = AssociationRule[]
        end
        push!(idx[r.consequent], r)
    end
    model.rule_index = idx
    model.is_fitted  = true

    pair_freq = nothing
    GC.gc()

    log_info("FP-Growth fit complete."; stage="FP-Fit")
    return model
end


"""
    get_lift_for_item(model::FPModel, cart::Vector, item_id) -> Float64

Given the user's current cart (antecedents) and a candidate item,
return the maximum lift from matching rules. Returns 0.0 if no rule fires.
"""
function get_lift_for_item(model::FPModel,
                             cart::Vector,
                             item_id)::Float64
    @assert model.is_fitted "FPModel must be fitted first."
    item_str = string(item_id)
    cart_set = Set(string.(cart))

    rules = get(model.rule_index, item_str, AssociationRule[])
    max_lift = 0.0
    for r in rules
        if !isempty(r.antecedent ∩ cart_set)
            max_lift = max(max_lift, r.lift)
        end
    end
    return max_lift
end


"""
    save_fp_model(model::FPModel, path::String)

Serialise the FP model to disk.
"""
function save_fp_model(model::FPModel, path::String)
    log_info("Saving FP model → $(path)"; stage="IO")
    open(path, "w") do io; serialize(io, model) end
    log_info("FP model saved."; stage="IO")
end

"""
    load_fp_model(path::String) -> FPModel
"""
function load_fp_model(path::String)::FPModel
    log_info("Loading FP model ← $(path)"; stage="IO")
    open(deserialize, path)
end

end # module FPGrowth
