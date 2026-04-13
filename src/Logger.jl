"""
    Logger.jl — Structured lifecycle logging for HybridRecommender

Provides timestamped, leveled log output for every pipeline stage:
  Fit → Candidate Generation → Reranking → Predict API
"""
module Logger

using Dates

export log_info, log_warn, log_error, log_stage, log_metric

# ANSI colour codes
const _RESET  = "\e[0m"
const _CYAN   = "\e[36m"
const _GREEN  = "\e[32m"
const _YELLOW = "\e[33m"
const _RED    = "\e[31m"
const _BOLD   = "\e[1m"
const _BLUE   = "\e[34m"

_ts() = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")

"""
    log_info(msg; stage="SYSTEM")

Print an INFO-level log line to stdout.
"""
function log_info(msg::AbstractString; stage::AbstractString="SYSTEM")
    println("$(_CYAN)[$(_ts())]$(_RESET) $(_BOLD)[INFO]$(_RESET) [$(_GREEN)$(stage)$(_RESET)] $(msg)")
    flush(stdout)
end

"""
    log_warn(msg; stage="SYSTEM")

Print a WARN-level log line to stdout.
"""
function log_warn(msg::AbstractString; stage::AbstractString="SYSTEM")
    println("$(_YELLOW)[$(_ts())]$(_RESET) $(_BOLD)[WARN]$(_RESET) [$(_YELLOW)$(stage)$(_RESET)] $(msg)")
    flush(stdout)
end

"""
    log_error(msg; stage="SYSTEM")

Print an ERROR-level log line to stderr.
"""
function log_error(msg::AbstractString; stage::AbstractString="SYSTEM")
    println(stderr, "$(_RED)[$(_ts())]$(_RESET) $(_BOLD)[ERROR]$(_RESET) [$(_RED)$(stage)$(_RESET)] $(msg)")
    flush(stderr)
end

"""
    log_stage(title)

Print a prominent stage-separator banner.
"""
function log_stage(title::AbstractString)
    banner = "=" ^ 60
    println("\n$(_BOLD)$(_BLUE)$(banner)$(_RESET)")
    println("$(_BOLD)$(_BLUE)  STAGE: $(uppercase(title))$(_RESET)")
    println("$(_BOLD)$(_BLUE)$(banner)$(_RESET)\n")
    flush(stdout)
end

"""
    log_metric(name, value; stage="METRICS")

Print a named scalar metric.
"""
function log_metric(name::AbstractString, value; stage::AbstractString="METRICS")
    println("$(_CYAN)[$(_ts())]$(_RESET) $(_BOLD)[METRIC]$(_RESET) [$(_GREEN)$(stage)$(_RESET)] $(name) = $(value)")
    flush(stdout)
end

end # module Logger
