using Crux
using Test

@testset "logging callback body coverage" begin
    buffer = ExperienceBuffer(
        Dict(
            :reward => Float32[1, 2, 3, 4],
            :cost => Float32[2, 4, 6, 8],
            :episode_end => Bool[true, true, true, true]
        )
    )

    s = (; buffer = buffer)

    avg_cb = log_episode_averages([:reward, :cost], 2)
    avg_result = avg_cb(𝒮 = s)

    @test haskey(avg_result, Symbol("avg_reward"))
    @test haskey(avg_result, Symbol("avg_cost"))
    @test isfinite(avg_result[Symbol("avg_reward")])
    @test isfinite(avg_result[Symbol("avg_cost")])

    sum_cb = log_experience_sums([:reward, :cost], 2)
    sum_result = sum_cb(𝒮 = s)

    @test haskey(sum_result, Symbol("avg_reward"))
    @test haskey(sum_result, Symbol("avg_cost"))
    @test isfinite(sum_result[Symbol("avg_reward")])
    @test isfinite(sum_result[Symbol("avg_cost")])
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
