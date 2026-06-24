using Crux, Flux
## Gpu stuff
vcpu = zeros(Float32, 10, 10)
vgpu = cu(zeros(Float32, 10, 10))
@test Crux.device(vcpu) == cpu
@test Crux.device(vgpu) == gpu
@test Crux.device(view(vcpu,:,1)) == cpu
@test Crux.device(view(vgpu,:,1)) == gpu

c_cpu = Chain(Dense(5,2))
c_gpu = Chain(Dense(5,2)) |> gpu
@test Crux.device(c_cpu) == cpu
@test Crux.device(c_gpu) == gpu

@test Crux.device(mdcall(c_cpu, rand(5), cpu)) == cpu
@test Crux.device(mdcall(c_gpu, rand(5), gpu)) == cpu
@test Crux.device(mdcall(c_cpu, cu(rand(5)), cpu)) == gpu
@test Crux.device(mdcall(c_gpu, cu(rand(5)), gpu)) == gpu

@testset "device helpers coverage" begin
    vcpu = zeros(Float32, 10, 10)
    vgpu = cu(zeros(Float32, 10, 10))

    @test Crux.device(vcpu) == cpu
    @test Crux.device(vgpu) == gpu

    @test Crux.device(view(vcpu, :, 1)) == cpu
    @test Crux.device(view(vgpu, :, 1)) == gpu

    c_cpu = Chain(Dense(5, 2))
    c_gpu = Chain(Dense(5, 2)) |> gpu

    @test Crux.device(c_cpu) == cpu
    @test Crux.device(c_gpu) == gpu

    @test Crux.device(mdcall(c_cpu, rand(Float32, 5), cpu)) == cpu
    @test Crux.device(mdcall(c_gpu, rand(Float32, 5), gpu)) == cpu
    @test Crux.device(mdcall(c_cpu, cu(rand(Float32, 5)), cpu)) == gpu
    @test Crux.device(mdcall(c_gpu, cu(rand(Float32, 5)), gpu)) == gpu

    @test cpucall(identity, vcpu) == vcpu
    @test gpucall(identity, vcpu) == vcpu

    x2 = rand(Float32, 3, 4)
    x3 = rand(Float32, 3, 4, 5)
    x4 = rand(Float32, 3, 4, 5, 6)
    x5 = rand(Float32, 2, 3, 4, 5, 6)

    @test collect(bslice(x2, 1)) == collect(view(x2, :, 1))
    @test collect(bslice(x3, 1)) == collect(view(x3, :, :, 1))
    @test collect(bslice(x4, 1)) == collect(view(x4, :, :, :, 1))
    @test collect(bslice(x5, 1)) == collect(view(x5, :, :, :, :, 1))
end
