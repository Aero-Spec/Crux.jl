## Behavioral Cloning / IQ-Learn losses

@testset "Behavior Cloning Losses" begin
    π_cont = ContinuousNetwork(
        Chain(Dense(2, 16, relu), Dense(16, 1))
    )

    s = rand(Float32, 2, 8)
    a = rand(Float32, 1, 8)

    D_cont = Dict(
        :s => s,
        :a => a,
        :value => value(π_cont, s)
    )

    @test isfinite(mse_action_loss(π_cont, (;), D_cont))

    π_gauss = GaussianPolicy(
        ContinuousNetwork(Chain(Dense(2, 16, relu), Dense(16, 1))),
        zeros(Float32, 1)
    )

    D_value = Dict(
        :s => s,
        :value => action(π_gauss, s)
    )

    @test isfinite(mse_value_loss(π_gauss, (λe=1f-3,), D_value))

    π_disc = DiscreteNetwork(
        Chain(Dense(2, 16, relu), Dense(16, 2)),
        [1, 2]
    )

    a_disc = Flux.onehotbatch(rand([1, 2], 8), [1, 2])

    D_disc = Dict(
        :s => s,
        :a => a_disc
    )

    info = Dict()
    l = logpdf_bc_loss(π_disc, (λe=1f-3,), D_disc; info=info)

    @test isfinite(l)
    @test haskey(info, :entropy)
    @test haskey(info, :logpdf)
end

@testset "IQ Loss" begin
    π = DiscreteNetwork(
        Chain(Dense(2, 16, relu), Dense(16, 2)),
        [1, 2]
    )

    D = Dict(
        :s => rand(Float32, 2, 10),
        :sp => rand(Float32, 2, 10),
        :a => Flux.onehotbatch(rand([1, 2], 10), [1, 2]),
        :done => zeros(Float32, 1, 10),
        :expert => reshape(
            [true, false, true, false, true, false, true, false, true, false],
            1,
            :
        )
    )

    info = Dict()

    loss = iq_loss(gp=false)(
        π,
        (;),
        D,
        nothing;
        info=info
    )

    @test isfinite(loss)
    @test haskey(info, :softQloss)
    @test haskey(info, :valueloss)
    @test haskey(info, :avg_R_expert_IQ)
    @test haskey(info, :avg_R_demo_IQ)
    @test haskey(info, :reg_loss)
end

@testset "IQ Callback" begin
    D = Dict{Symbol, Any}(
        :done => zeros(Float32, 5)
    )

    Crux.iq_callback(D; 𝒮=nothing)

    @test haskey(D, :expert)
    @test size(D[:expert]) == (1, 5)
    @test all(.!D[:expert])
end
