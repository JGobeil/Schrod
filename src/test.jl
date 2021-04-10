using Test
using Schrod
using AtomicUnits: eV, nm

@info "Creating a 1D gaussian set"

@time gs = GaussianSet(-2nm, 2nm, 1201)

@info "Using a box potential"

box(x::Number) = abs(x) > 0.2nm ? 100eV : 0eV
box(x::AbstractArray) = box(x[1])

@info "Solve using multithread and progressbar"
@time wf_tt = solve(box, gs, progressbar=true, multithread=true)

@info "Solve using no multithread and progressbar"
@time wf_tf = solve(box, gs, progressbar=true, multithread=false)

@info "Solve using multithread and no progressbar"
@time wf_ft = solve(box, gs, progressbar=false, multithread=true)

@info "Solve using no multithread and no progressbar"
@time wf_ff = solve(box, gs, progressbar=false, multithread=false)

@info "WaveFunction" wf_ff

@testset "Testing results equivalence" begin
    @test energy(wf_tt) ≈ energy(wf_tf) atol=0.0001eV
    @test energy(wf_tt) ≈ energy(wf_ft) atol=0.0001eV
    @test energy(wf_tt) ≈ energy(wf_ff) atol=0.0001eV
end;

@testset "Testing energy values" begin
    @test energy(wf, 1) ≈ 1.9497eV atol=0.0001eV
    @test energy(wf, 2) ≈ 7.7847eV atol=0.0001eV
end;