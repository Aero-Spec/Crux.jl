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
    @test eltype(y) == Float32

    ps = Flux.trainable(layer)

    @test length(ps) == 2

    io = IOBuffer()
    show(io, layer)
    s = String(take!(io))

    @test occursin("ConvSN", s)
end

@testset "device helpers coverage" begin
    x2 = rand(Float32, 3, 4)
    x3 = rand(Float32, 3, 4, 5)
    x4 = rand(Float32, 3, 4, 5, 6)
    x5 = rand(Float32, 2, 3, 4, 5, 6)

    @test device(x2) == cpu
    @test device(view(x2, :, 1:2)) == cpu

    @test cpucall(identity, x2) == x2
    @test mdcall(identity, x2, cpu) == x2

    @test collect(bslice(x2, 1)) == collect(view(x2, :, 1))
    @test collect(bslice(x3, 1)) == collect(view(x3, :, :, 1))
    @test collect(bslice(x4, 1)) == collect(view(x4, :, :, :, 1))
    @test collect(bslice(x5, 1)) == collect(view(x5, :, :, :, :, 1))
end
