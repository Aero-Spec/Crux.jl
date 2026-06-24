@testset "logging helper coverage" begin
    buffer = ExperienceBuffer(10)

    push!(buffer; reward = 1.0f0, episode_end = true)
    push!(buffer; reward = 2.0f0, episode_end = true)
    push!(buffer; reward = 3.0f0, episode_end = true)
    push!(buffer; reward = 4.0f0, episode_end = true)

    s = (; buffer = buffer)

    avg_cb = log_episode_averages([:reward], 2)
    avg_result = avg_cb(s)

    @test haskey(avg_result, Symbol("avg_reward"))
    @test isfinite(avg_result[Symbol("avg_reward")])

    sum_cb = log_experience_sums([:reward], 2)
    sum_result = sum_cb(s)

    @test haskey(sum_result, Symbol("avg_reward"))
    @test isfinite(sum_result[Symbol("avg_reward")])
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

@testset "FirstExplorePolicy logging coverage" begin
    p = FirstExplorePolicy(5, nothing)

    cb = log_exploration(p)

    d1 = cb(1)
    d2 = cb(10)

    @test haskey(d1, "first_explore_on")
    @test haskey(d2, "first_explore_on")

    @test d1["first_explore_on"] == true
    @test d2["first_explore_on"] == false
end
