using Crux
using Test
using Zygote
using Flux
using Random

Random.seed!(1)

@testset "gradient penalty" begin
    m = Dense(2, 1, init = ones, bias = false)
    x = ones(Float32, 2, 100)

    @test gradient_penalty(m, x) ≈ (sqrt(2) - 1)^2
end

@testset "gradient penalty with pullback" begin
    idim = 5
    batch_size = 8

    m = Chain(Dense(idim, 2 * idim, tanh), Dense(2 * idim, 1)) |> gpu
    x = rand(Float32, idim, batch_size) |> gpu
    y = rand(Float32, 1, batch_size) |> gpu

    function total_loss()
        Flux.mse(m(x), y) + gradient_penalty(m, x)
    end

    l, b = Flux.pullback(total_loss, Flux.params(m))
    grad = b(1f0)

    @test l isa Number
    @test isfinite(l)
    @test grad.grads isa IdDict
end

@testset "GAN loss coverage" begin
    x = rand(Float32, 2, 8)
    z = rand(Float32, 2, 8)

    G = Chain(Dense(2, 2))
    D = Chain(Dense(2, 1))

    losses = [
        GAN_BCELoss(),
        GAN_LSLoss(),
        GAN_HingeLoss(),
        GAN_WLoss(),
        GAN_WLossGP()
    ]

    for loss in losses
        d_loss = Crux.Lᴰ(loss, D, x, x)
        g_loss = Crux.Lᴳ(loss, G, D, z)

        @test d_loss isa Number
        @test g_loss isa Number
        @test isfinite(d_loss)
        @test isfinite(g_loss)
    end
end

@testset "Conditional GAN loss coverage" begin
    batch_size = 8

    x = rand(Float32, 2, batch_size)
    z = rand(Float32, 2, batch_size)

    yG = [rand(Float32, 1, batch_size)]
    yD = [rand(Float32, 1, batch_size)]

    G = Chain(Dense(3, 2))
    D = Chain(Dense(3, 1))

    losses = [
        GAN_BCELoss(),
        GAN_LSLoss(),
        GAN_HingeLoss(),
        GAN_WLoss(),
        GAN_WLossGP()
    ]

    for loss in losses
        d_loss = Crux.Lᴰ(loss, D, x, x; yG = yG, yD = yD)
        g_loss = Crux.Lᴳ(loss, G, D, z; yG = yG)

        @test d_loss isa Number
        @test g_loss isa Number
        @test isfinite(d_loss)
        @test isfinite(g_loss)
    end
end

@testset "GANLosses dictionary coverage" begin
    @test haskey(GANLosses, "BCE")
    @test haskey(GANLosses, "LS")
    @test haskey(GANLosses, "Hinge")
    @test haskey(GANLosses, "W")
    @test haskey(GANLosses, "WGP")
end

@testset "ConvSN coverage" begin
    x = rand(Float32, 8, 8, 1, 4)

    layer = ConvSN((3, 3), 1 => 2, relu; pad = 1, stride = 1)

    y = layer(x)

    @test size(y) == (8, 8, 2, 4)
    @test eltype(y) <: Real
    @test all(isfinite, y)

    ps = Flux.trainable(layer)
    @test length(ps) == 2

    io = IOBuffer()
    show(io, layer)
    str = String(take!(io))

    @test occursin("ConvSN", str)
end
@testset "BatchAdjacentBuffer coverage" begin
    ref = ExperienceBuffer()
    b = BatchAdjacentBuffer(elements = 3, default_z_dist = nothing, reference_buffer = ref)

    @test isprioritized(b) == false
    @test capacity(b) == 3
    @test length(b) == 0

    data1 = ExperienceBuffer((reward = [1.0, 2.0], episode_end = [false, true]))
    data2 = ExperienceBuffer((reward = [3.0], episode_end = [true]))
    data3 = ExperienceBuffer((reward = [4.0, 5.0], episode_end = [false, true]))
    data4 = ExperienceBuffer((reward = [6.0], episode_end = [true]))

    push!(b, data1, 0.1)
    push!(b, data2, 0.2)
    push!(b, data3, 0.3)

    @test length(b.batches) == 3
    @test length(b) == 5
    @test b.z_dists == [0.1, 0.2, 0.3]

    push!(b, data4, 0.4)

    @test length(b.batches) == 3
    @test b.z_dists[1] == 0.4
    @test b.next_ind == 2

    old_len = length(b.batches)
    push!(b, data1)
    @test length(b.batches) == old_len

    inds = get_last_N_indices(b, 2)
    @test length(inds) == 2
    @test all(1 .<= inds .<= capacity(b))
end

@testset "BatchAdjacentBuffer reservoir coverage" begin
    ref = ExperienceBuffer()
    b = BatchAdjacentBuffer(elements = 2, default_z_dist = nothing, reference_buffer = ref)

    data = ExperienceBuffer((reward = [1.0], episode_end = [true]))

    push_reservoir!(b, data, 0.1; weight = 0.0)
    @test length(b.batches) == 0

    push_reservoir!(b, data, 0.1; weight = 1.0)
    push_reservoir!(b, data, 0.2; weight = 1.0)
    push_reservoir!(b, data, 0.3; weight = 1.0)

    @test length(b.batches) == 2
    @test b.total_count == 3
end

@testset "BatchAdjacentBuffer log averages coverage" begin
    ref = ExperienceBuffer()
    b = BatchAdjacentBuffer(elements = 3, default_z_dist = nothing, reference_buffer = ref)

    push!(b, ExperienceBuffer((reward = [1.0, 3.0], episode_end = [false, true])), nothing)
    push!(b, ExperienceBuffer((reward = [2.0, 4.0], episode_end = [false, true])), nothing)

    logger = log_episode_averages(b, [:reward], 2)
    d = logger()

    @test haskey(d, :avg_reward)
    @test d[:avg_reward] ≈ 5.0
end
