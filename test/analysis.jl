using POMDPs
using POMDPGym
import POMDPModels
using Test
using Crux
using Flux
using Random
using BSON

mdp = GymPOMDP(:CartPole)
as = actions(mdp)
S = state_space(mdp)

# Flux network: Map states to actions
A() = DiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, length(as))), as)
V() = ContinuousNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 1)))

solver_reinforce = REINFORCE(S=S, π=A())
policy_reinforce = solve(solver_reinforce, mdp)

solver_a2c = A2C(S=S, π=ActorCritic(A(), V()))
policy_a2c = solve(solver_a2c, mdp)

solver_ppo = PPO(S=S, π=ActorCritic(A(), V()))
policy_ppo = solve(solver_ppo, mdp)

p = plot_learning([solver_reinforce, solver_a2c, solver_ppo],
                  title="CartPole Training Curves",
                  labels=["REINFORCE", "A2C", "PPO"])

Crux.savefig(p, "test.pdf")

rm("test.pdf")

@testset "Learning utility coverage" begin

    # Test percentile interpolation helper
    @test percentile(0.5, 10, 20) == 15
    @test percentile(0.0, 10, 20) == 10
    @test percentile(1.0, 10, 20) == 20

    # Test threshold crossing detection
    @test find_crossing([1, 2, 3], [0.1, 0.7, 0.9], 0.6) == 2

    # Verify NaN is returned when threshold is never reached
    @test isnan(find_crossing([1, 2, 3], [0.1, 0.2, 0.3], 0.6))

    # Test exponential smoothing utility
    @test smooth([1.0, 3.0, 5.0], 0.5) == [1.0, 2.0, 3.5]

    # Test directory conversion helpers for common input types
    @test directories("logs/run1") == ["logs/run1"]
    @test directories(["a", "b"]) == ["a", "b"]

    # Test custom Dict concatenation utility
    d1 = Dict(:a => [1, 2], :b => [3])
    d2 = Dict(:a => [4], :b => [5, 6])

    dcat = vcat(d1, d2)

    @test dcat[:a] == [1, 2, 4]
    @test dcat[:b] == [3, 5, 6]
end

@testset "Learning curve plotting coverage" begin
    solvers = [solver_reinforce, solver_a2c, solver_ppo]

    # CartPole logs use :undiscounted_return, not :undiscounted_return/T1
    p1 = plot_jumpstart(solvers; key = i -> :undiscounted_return)
    p2 = plot_peak_performance(solvers; key = i -> :undiscounted_return)

    Crux.savefig(p1, "jumpstart.pdf")
    Crux.savefig(p2, "peak.pdf")

    @test isfile("jumpstart.pdf")
    @test isfile("peak.pdf")

    rm("jumpstart.pdf")
    rm("peak.pdf")
end

@testset "Continual learning plotting coverage" begin
    solvers = [solver_reinforce, solver_a2c, solver_ppo]

    x, y, breaks = cumulative_rewards(solvers; key = i -> :undiscounted_return)

    @test length(x) > 0
    @test length(y) > 0
    @test length(breaks) == length(solvers)

    res, breaks2 = single_task_performances(solvers; key = i -> :undiscounted_return)

    @test length(res) > 0
    @test length(breaks2) == length(solvers)

    p3 = plot_cumulative_rewards(
        solvers;
        key = i -> :undiscounted_return,
        show_lines = true
    )

    Crux.savefig(p3, "cumulative_rewards.pdf")
    @test isfile("cumulative_rewards.pdf")
    rm("cumulative_rewards.pdf")
end
