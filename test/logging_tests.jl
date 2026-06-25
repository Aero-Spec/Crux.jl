using Crux
using Test

@testset "logging callback construction coverage" begin
    avg_cb = log_episode_averages([:reward], 2)
    sum_cb = log_experience_sums([:reward], 2)

    @test avg_cb isa Function
    @test sum_cb isa Function
end

@testset "save_gif early return coverage" begin
    cb = save_gif(log_at_zero = false)

    s = (
        dir = tempdir(),
        mdp = nothing,
        agent = (; π = nothing)
    )

    @test cb(i = 0, s = s, dir = tempdir(), logger = nothing) === nothing
end
