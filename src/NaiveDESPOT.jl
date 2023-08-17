
module NaiveDESPOT

#=
Current constraints:
- action space discrete
- action space same for all states, histories
- no built-in support for history-dependent rollouts (this could be added though)
- initial n and initial v are 0
=#

using POMDPs
using Parameters
using ParticleFilters
using CPUTime
using Colors
using Random
using Printf
using POMDPLinter: @POMDP_require, @show_requirements
using POMDPTools
# using BasicPOMCP

import POMDPs: action, solve, updater
import POMDPLinter

using MCTS
import MCTS: convert_estimator, estimate_value, node_tag, tooltip_tag, default_action

using D3Trees
using BasicPOMCP

export
    NDESPOTSolver,
    NDESPOTPlanner,

    # action,
    solve,
    updater,

    # NoDecision,
    # AllSamplesTerminal,
    # ExceptionRethrow,
    # ReportWhenUsed,
    # default_action,

    # BeliefNode,
    # LeafNodeBelief,
    AbstractNDESPOTSolver,

    # PORollout,
    # FORollout,
    # RolloutEstimator,
    # FOValue,

    D3Tree,
    node_tag,
    tooltip_tag,

    # deprecated
    AOHistoryBelief

abstract type AbstractNDESPOTSolver <: Solver end

"""
    NDESPOTSolver(#=keyword arguments=#)

Partially Observable Monte Carlo Planning Solver.

## Keyword Arguments

- `max_depth::Int`
    Rollouts and tree expension will stop when this depth is reached.
    default: `20`

- `c::Float64`
    UCB exploration constant - specifies how much the solver should explore.
    default: `1.0`

- `tree_queries::Int`
    Number of iterations during each action() call.
    default: `1000`

- `max_time::Float64`
    Maximum time for planning in each action() call.
    default: `Inf`

- `tree_in_info::Bool`
    If `true`, returns the tree in the info dict when action_info is called.
    default: `false`

- `estimate_value::Any`
    Function, object, or number used to estimate the value at the leaf nodes.
    default: `RolloutEstimator(RandomSolver(rng))`
    - If this is a function `f`, `f(pomdp, s, h::BeliefNode, steps)` will be called to estimate the value.
    - If this is an object `o`, `estimate_value(o, pomdp, s, h::BeliefNode, steps)` will be called.
    - If this is a number, the value will be set to that number
    Note: In many cases, the simplest way to estimate the value is to do a rollout on the fully observable MDP with a policy that is a function of the state. To do this, use `FORollout(policy)`.

- `default_action::Any`
    Function, action, or Policy used to determine the action if POMCP fails with exception `ex`.
    default: `ExceptionRethrow()`
    - If this is a Function `f`, `f(pomdp, belief, ex)` will be called.
    - If this is a Policy `p`, `action(p, belief)` will be called.
    - If it is an object `a`, `default_action(a, pomdp, belief, ex)` will be called, and if this method is not implemented, `a` will be returned directly.

- `rng::AbstractRNG`
    Random number generator.
    default: `Random.GLOBAL_RNG`
"""
@with_kw mutable struct NDESPOTSolver <: AbstractNDESPOTSolver
    max_depth::Int          = 20
    scenarios::Int          = 20
    c::Float64              = 1.0
    tree_queries::Int       = 1000
    max_time::Float64       = Inf
    tree_in_info::Bool      = false
    default_action::Any     = ExceptionRethrow()
    rng::AbstractRNG        = Random.GLOBAL_RNG
    estimate_value::Any     = RolloutEstimator(RandomSolver(rng))
end

mutable struct NDESPOTPlanner{P, SE, RNG} <: Policy
    solver::NDESPOTSolver
    problem::P
    solved_estimator::SE
    rng::RNG
    _best_node_mem::Vector{Int}
    _tree::Union{Nothing, Any}
end

function NDESPOTPlanner(solver::NDESPOTSolver, pomdp::POMDP)
    se = convert_estimator(solver.estimate_value, solver, pomdp)
    return NDESPOTPlanner(solver, pomdp, se, solver.rng, Int[], nothing)
end

Random.seed!(p::NDESPOTPlanner, seed) = Random.seed!(p.rng, seed)

function updater(p::NDESPOTPlanner)
    P = typeof(p.problem)
    S = statetype(P)
    A = actiontype(P)
    O = obstype(P)
    return UnweightedParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
    # XXX It would be better to automatically use an SIRParticleFilter if possible
    # if !@implemented ParticleFilters.obs_weight(::P, ::S, ::A, ::S, ::O)
    #     return UnweightedParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
    # end
    # return SIRParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
end

include("solver.jl")

include("exceptions.jl")
include("rollout.jl")
include("visualization.jl")
include("requirements_info.jl")

end # module
