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
