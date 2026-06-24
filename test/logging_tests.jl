@testset "logging callback coverage" begin
    buffer = Dict(
        :reward => [1.0, 2.0, 3.0, 4.0],
        :episode_end => [1, 1, 1, 1]
    )

    s = (; buffer = buffer)

    avg_cb = log_episode_averages([:reward], 2)
    avg_result = avg_cb(s)

    @test haskey(avg_result, Symbol("avg_reward"))
    @test avg_result[Symbol("avg_reward")] ≈ 3.5

    sum_cb = log_experience_sums([:reward], 2)
    sum_result = sum_cb(s)

    @test haskey(sum_result, Symbol("avg_reward"))
    @test sum_result[Symbol("avg_reward")] ≈ 7.0
end

@testset "save_gif early return coverage" begin
    cb = save_gif(log_at_zero = false)

    s = (
        dir = tempdir(),
        mdp = nothing,
        agent = (; π = nothing),
        logger = nothing
    )

    @test cb(0, s) === nothing
end
