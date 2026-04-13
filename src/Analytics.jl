"""
    Analytics.jl — Diagnostic Visualization Suite

Provides three diagnostic plots for evaluating model health:

1. `plot_explained_variance`  — Cumulative Explained Variance (Stage 1)
2. `plot_rule_quality`        — Support vs Confidence scatter (Stage 2)
3. `plot_reranking_impact`    — Rank distribution before/after FP-Growth
"""
module Analytics

using Statistics, Dates
include("Logger.jl")
include("MatrixFactorization.jl")
include("FPGrowth.jl")
include("Reranker.jl")
using .Logger, .MatrixFactorization, .FPGrowth, .Reranker

export plot_explained_variance, plot_rule_quality, plot_reranking_impact,
       run_all_diagnostics

# ── Try to load Plots lazily ───────────────────────────────────────────────────
function _ensure_plots()
    try
        @eval using Plots
        @eval gr()
        return true
    catch e
        log_warn("Plots.jl unavailable — will save CSV data instead. ($e)"; stage="Analytics")
        return false
    end
end


# ── 1. Cumulative Explained Variance ──────────────────────────────────────────

"""
    plot_explained_variance(model::SVDModel; outdir=".", show=false)

Plot the cumulative explained variance ratio vs. number of components.
Includes a reference line at 80% to guide component selection.
Saves to `<outdir>/explained_variance.png`.
"""
function plot_explained_variance(model::SVDModel;
                                  outdir::String=".",
                                  show::Bool=false)

    log_info("Generating Explained Variance diagnostic…"; stage="Analytics")
    @assert model.is_fitted "SVD model must be fitted."

    ev = model.var_ratios
    k  = length(ev)
    components = 1:k

    # Find the "elbow" — first component reaching 80% variance
    elbow_k = findfirst(x -> x >= 0.80, ev)

    has_plots = _ensure_plots()

    if has_plots
        @eval begin
            p = plot($(collect(components)), $(ev) .* 100;
                label="Cumulative Explained Variance",
                xlabel="Number of Components",
                ylabel="Variance Explained (%)",
                title="Stage 1 — Latent Factor Diagnostic\nOptimal Component Selection",
                linewidth=2,
                color=:steelblue,
                marker=:circle,
                markersize=3,
                legend=:bottomright,
                grid=true,
                gridalpha=0.3,
                size=(800, 480),
                dpi=150)

            hline!([80.0]; label="80% threshold", linestyle=:dash, color=:red, linewidth=1.5)

            if !isnothing($(elbow_k))
                vline!([$(elbow_k)]; label="Elbow k=$($(elbow_k))",
                       linestyle=:dot, color=:orange, linewidth=1.5)
            end

            out = joinpath($(outdir), "explained_variance.png")
            savefig(p, out)
            @info "Saved: $out"
            $(show) && display(p)
        end
    else
        # Fallback: write CSV
        out = joinpath(outdir, "explained_variance.csv")
        open(out, "w") do f
            println(f, "component,cumulative_var_pct")
            for (i, v) in zip(components, ev)
                println(f, "$(i),$(round(v*100; digits=4))")
            end
        end
        log_info("Saved CSV fallback → $(out)"; stage="Analytics")
    end

    log_metric("elbow_k", isnothing(elbow_k) ? k : elbow_k; stage="Analytics")
end


# ── 2. Rule Quality Scatter ───────────────────────────────────────────────────

"""
    plot_rule_quality(fp_model::FPModel; outdir=".", show=false)

Scatter plot of Support vs Confidence for all mined rules,
colour-coded by Lift value.
Saves to `<outdir>/rule_quality_scatter.png`.
"""
function plot_rule_quality(fp_model::FPModel;
                            outdir::String=".",
                            show::Bool=false)

    log_info("Generating Rule Quality scatter…"; stage="Analytics")
    @assert fp_model.is_fitted "FP model must be fitted."

    rules = fp_model.rules
    if isempty(rules)
        log_warn("No rules to plot."; stage="Analytics")
        return
    end

    sups  = [r.support    for r in rules]
    confs = [r.confidence for r in rules]
    lifts = [r.lift       for r in rules]

    log_metric("rule_count",    length(rules);        stage="Analytics")
    log_metric("avg_support",   round(mean(sups);  digits=4); stage="Analytics")
    log_metric("avg_confidence",round(mean(confs); digits=4); stage="Analytics")
    log_metric("avg_lift",      round(mean(lifts); digits=4); stage="Analytics")

    has_plots = _ensure_plots()

    if has_plots
        @eval begin
            p = scatter($(sups), $(confs);
                marker_z=$(lifts),
                color=:viridis,
                colorbar=true,
                colorbar_title="Lift",
                label="",
                xlabel="Support",
                ylabel="Confidence",
                title="Stage 2 — Rule Quality Scatter\nSupport vs Confidence (coloured by Lift)",
                alpha=0.65,
                markersize=4,
                markerstrokewidth=0,
                size=(800, 520),
                dpi=150,
                grid=true,
                gridalpha=0.3)

            vline!([$(fp_model.min_support)];   label="min_support",    linestyle=:dash, color=:red)
            hline!([$(fp_model.min_confidence)];label="min_confidence",  linestyle=:dash, color=:blue)

            out = joinpath($(outdir), "rule_quality_scatter.png")
            savefig(p, out)
            @info "Saved: $out"
            $(show) && display(p)
        end
    else
        out = joinpath(outdir, "rule_quality.csv")
        open(out, "w") do f
            println(f, "support,confidence,lift")
            for (s,c,l) in zip(sups,confs,lifts)
                println(f, "$(s),$(c),$(l)")
            end
        end
        log_info("Saved CSV fallback → $(out)"; stage="Analytics")
    end
end


# ── 3. Reranking Impact Histogram ─────────────────────────────────────────────

"""
    plot_reranking_impact(pre_ranks::Vector{Int}, post_ranks::Vector{Int};
                          outdir=".", show=false)

Compare item rank distributions before and after FP-Growth reranking.
Saves to `<outdir>/reranking_impact.png`.

Arguments
---------
- `pre_ranks`  : item ranks from pure CF output (Stage 1).
- `post_ranks` : item ranks after hybrid reranking (Stage 2).
"""
function plot_reranking_impact(pre_ranks::Vector{Int},
                                post_ranks::Vector{Int};
                                outdir::String=".",
                                show::Bool=false)

    log_info("Generating Reranking Impact histogram…"; stage="Analytics")

    if isempty(pre_ranks) || isempty(post_ranks)
        log_warn("Empty rank vectors — skipping plot."; stage="Analytics")
        return
    end

    rank_shift = mean(pre_ranks) - mean(post_ranks)
    log_metric("avg_rank_before", round(mean(pre_ranks);  digits=2); stage="Analytics")
    log_metric("avg_rank_after",  round(mean(post_ranks); digits=2); stage="Analytics")
    log_metric("avg_rank_shift",  round(rank_shift;       digits=2); stage="Analytics")

    has_plots = _ensure_plots()

    if has_plots
        @eval begin
            bins = range(1, max(maximum($(pre_ranks)), maximum($(post_ranks)))+1; step=1)

            p = histogram($(pre_ranks);
                bins=bins,
                label="Before FP-Growth (CF only)",
                alpha=0.55,
                color=:steelblue,
                xlabel="Item Rank",
                ylabel="Frequency",
                title="Stage 2 — Reranking Impact\nRank Distribution Before vs After FP-Growth",
                size=(800, 480),
                dpi=150,
                grid=true,
                gridalpha=0.3,
                legend=:topright)

            histogram!($(post_ranks);
                bins=bins,
                label="After FP-Growth (Hybrid)",
                alpha=0.55,
                color=:darkorange)

            vline!([mean($(pre_ranks))];  label="μ before", linestyle=:dash, color=:steelblue, linewidth=2)
            vline!([mean($(post_ranks))]; label="μ after",  linestyle=:dash, color=:darkorange,  linewidth=2)

            out = joinpath($(outdir), "reranking_impact.png")
            savefig(p, out)
            @info "Saved: $out"
            $(show) && display(p)
        end
    else
        out = joinpath(outdir, "reranking_impact.csv")
        open(out, "w") do f
            println(f, "pre_rank,post_rank")
            for (a,b) in zip(pre_ranks, post_ranks)
                println(f, "$(a),$(b)")
            end
        end
        log_info("Saved CSV fallback → $(out)"; stage="Analytics")
    end
end


# ── Convenience: run all diagnostics ──────────────────────────────────────────

"""
    run_all_diagnostics(svd_model, fp_model, results; outdir="plots")

Run all three diagnostic plots in sequence.
"""
function run_all_diagnostics(svd_model::SVDModel,
                               fp_model::FPModel,
                               results::Vector{RerankedResult};
                               outdir::String="plots")

    mkpath(outdir)
    log_stage("Analytics & Diagnostics")

    # 1. Explained variance
    plot_explained_variance(svd_model; outdir=outdir)

    # 2. Rule quality
    plot_rule_quality(fp_model; outdir=outdir)

    # 3. Reranking impact — derive ranks from results
    pre_ranks  = Int[]
    post_ranks = Int[]
    for res in results
        for (rank, item) in enumerate(res.items)
            push!(post_ranks, rank)
            # Pre-rank: sort by CF score alone
            cf_order = sortperm(res.cf_scores; rev=true)
            pre_rank = findfirst(==(rank), cf_order)
            isnothing(pre_rank) || push!(pre_ranks, pre_rank)
        end
    end

    length(pre_ranks) == length(post_ranks) || resize!(pre_ranks, length(post_ranks))
    plot_reranking_impact(pre_ranks, post_ranks; outdir=outdir)

    log_info("All diagnostics saved to '$(outdir)/'"; stage="Analytics")
end

end # module Analytics
