using Test
using Schrod
using AtomicUnits: eV, nm
using Base.Threads

@info "Creating a 1D gaussian set"

gs_small = GaussianSet(-2nm, 2nm, 111)
@time gs = GaussianSet(-2nm, 2nm, 1001)

@info "Using a box potential"

box(x::Number) = abs(x) > 0.2nm ? 100eV : 0eV
box(x::AbstractArray) = box(x[1])

@info "Number of threads" nthreads()

@info "Precompilation"
for pb=[true, false], mt=[true, false]
    solve(box, gs_small, progressbar=pb, multithread=mt)
end

@info "Solve using multithread and progressbar"
t_tt = @elapsed wf_tt = solve(box, gs, progressbar=true, multithread=true)

@info "Solve using no multithread and progressbar"
t_tf = @elapsed wf_tf = solve(box, gs, progressbar=true, multithread=false)

@info "Solve using multithread and no progressbar"
t_ft = @elapsed wf_ft = solve(box, gs, progressbar=false, multithread=true)

@info "Solve using no multithread and no progressbar"
t_ff = @elapsed wf_ff = solve(box, gs, progressbar=false, multithread=false)

@info "WaveFunction" wf_ff
@info "Timing\n\t" * join([
    "Progress Bar | Multithread | Elapsed time",
    "true         | true        | $t_tt s.",
    "true         | false       | $t_tf s.",
    "false        | true        | $t_ft s.",
    "false        | false       | $t_ff s.",
], "\n\t")

@testset verbose=true "Verify solutions" begin
    @testset "Testing results equivalence" begin
        @test all(abs.(1 .- energy(wf_tt) ./ energy(wf_tf)) .< 0.005)
        @test all(abs.(1 .- energy(wf_tt) ./ energy(wf_ft)) .< 0.005)
        @test all(abs.(1 .- energy(wf_tt) ./ energy(wf_ff)) .< 0.005)
    end;

    @testset "Testing energy values" begin
        @test energy(wf_ff, 1) ≈  1.9497eV atol=0.0005eV
        @test energy(wf_ff, 2) ≈  7.7847eV atol=0.0005eV
        @test energy(wf_ff, 3) ≈ 17.4591eV atol=0.0005eV
    end;
end;


##