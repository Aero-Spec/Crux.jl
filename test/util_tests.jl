using Crux
using Test
using POMDPModels
using Flux
using LinearAlgebra
using CUDA
using Distributions

# bslice
v = zeros(4, 4, 4)
@test size(bslice(v, 2)) == (4, 4)

# Constant Layer
c1 = ConstantLayer(ones(10))
@test Crux.device(c1) == cpu
@test c1(rand(100)) == c1.vec
@test Flux.params(c1)[1] == c1.vec

if USE_CUDA
    c2 = c1 |> gpu
    @test Crux.device(c2) == gpu
    @test c2(rand(100)) == c2.vec
end

# Distribution stuff
objs = [:up, :down]
o = ObjectCategorical(objs)
@test o isa DiscreteUnivariateDistribution
@test o.objs == objs
@test o.cat.p == Categorical(2).p
@test rand(o) in objs
@test size(rand(o, 10)) == (10,)

@test logpdf(o, [:up]) == logpdf(o, [:down])
@test size(logpdf(o, rand(o, 10))) == (1, 10)

@test entropy(o) == entropy(o.cat)

fitted = Distributions.fit(typeof(o), objs, [1.0, 2.0]; objs=objs)
@test fitted isa ObjectCategorical
@test fitted.objs == objs

# Useful functions
@test Crux.whiten([1.0, 2.0, 3.0], 2.0, 1.0) == [-1.0, 0.0, 1.0]

W3 = reshape(1:24, 2, 3, 4)
@test size(Crux.to2D(W3)) == (6, 4)

wm = Crux.weighted_mean([1.0, 2.0, 3.0])
@test wm([2.0, 4.0, 6.0]) == 28.0 / 3.0

@test Crux.logcomplement(0.5) ≈ log(0.5)

xw = [1.0, 2.0, 3.0]
ww = [1.0, 1.0, 1.0]
@test Crux.weighted_logsumexp(xw, ww) ≈ log(sum(exp.(xw)))

xw2 = [3.0, 2.0, 1.0]
ww2 = [1.0, 2.0, 3.0]
@test Crux.weighted_logsumexp(xw2, ww2) ≈ log(sum(ww2 .* exp.(xw2)))

# Flux Stuff
W = rand(2, 5)
b = rand(2)

predict(x) = (W * x) .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(5), rand(2)
lval = loss(x, y)

θ = Flux.params(W, b)
grads = Flux.gradient(() -> loss(x, y), θ)

@test norm(grads) > 1

# MultitaskDecay Schedule
m = MultitaskDecaySchedule(10, [1, 2, 3])
l = Crux.LinearDecaySchedule(1.0, 0.1, 10)

for i = 1:10
    @test m(i) == l(i)
end

for i = 11:20
    @test m(i) == l(i - 10)
end

for i = 21:30
    @test m(i) == l(i - 20)
end

m = MultitaskDecaySchedule(10, [1, 2, 1])

for i = 1:10
    @test m(i) == l(i)
end

for i = 11:20
    @test m(i) == l(i - 10)
end

for i = 21:30
    @test m(i) == l(i - 10)
end

@test m(31) == 0.1
@test m(0) == 1

# Early stopping coverage
@testset "stop_on_validation_increase coverage" begin
    infos = [Dict{String, Any}() for _ in 1:8]

    vals = Ref([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0])
    loss_fn(_, _, _) = popfirst!(vals[])

    stopper = Crux.stop_on_validation_increase(nothing, nothing, nothing, loss_fn; window=2)
    results = [stopper(infos[1:i]) for i in 1:length(infos)]

    @test any(results)
    @test all(haskey(info, "validation_error") for info in infos)
end

# Multi-loss coverage
@testset "multi loss coverage" begin
    net1(x) = x
    net2(x) = x

    π = (; networks = [net1, net2])
    P = nothing
    D = Dict(:s => [1.0], :a => [1.0])
    ys = [[1.0], [2.0]]

    td_loss(net, P, D, y; info=Dict()) = sum(abs.(net(y)))
    actor_loss(net, P, D; info=Dict()) = sum(abs.(net(D[:s])))

    loss_fn = Crux.multi_td_loss([td_loss, td_loss])
    @test loss_fn(π, P, D, ys) ≥ 0

    loss_fn_indexed = Crux.multi_td_loss([td_loss, td_loss]; indices=1:1)
    @test loss_fn_indexed(π, P, D, ys) ≥ 0

    actor_fn = Crux.multi_actor_loss(actor_loss, 2)
    @test actor_fn(π, P, D) ≥ 0

    actor_fn_indexed = Crux.multi_actor_loss(actor_loss, 2; indices=1:1)
    @test actor_fn_indexed(π, P, D) ≥ 0
end
