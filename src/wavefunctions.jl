
abstract type AbstractWaveFunctions end
Base.broadcastable(p::AbstractWaveFunctions) = Ref(p)

""" Structure containing the results of a calculation with
eigenvalues, eigenvector, function set and the potential used."""
struct WaveFunctions{FLOAT, FSET, PFUNC} <: AbstractWaveFunctions
    energy::Vector{FLOAT}
	statevec::Matrix{FLOAT}
	functionset::FSET
	waves::Vector{FSET}
	potential::PFUNC
end

""" WaveFunction contructor """
function WaveFunctions(eigs, functionset::AbstractFunctionSet, potential)
    Ai = eigs.vectors
    Ei = eigs.values
    if eltype(Ei) <: Complex
        @warn "Eigenvalues are complex"
        Ei = abs.(Ei)
        Ai = abs.(Ai)
    end

    # permutation
    p = sortperm(Ei) 

    @debug "Energy" Ei[1:10]'/eV

    Ei = Ei[p]
    Ai = Ai[:, p]

	sets = [typeof(functionset)(
		functionset.σ,
		functionset.ν,
		functionset.μ,
		view(Ai, :, i) .* functionset.a,
		functionset.N,
	) for i in 1:size(Ei, 1)]
    WaveFunctions(Ei, collect(Ai), functionset, sets, potential)
end

energy(states, n::Integer) = energy(states)[n]
energy(wf::WaveFunctions) = wf.energy



function groupbyenergy(Ei, tol)
	cut = cutree(hclust([abs(E1 - E2) for E1 in Ei, E2 in Ei]), h=tol)
	N = maximum(cut)
	groups = [[i for i in 1:length(cut) if cut[i] == j] for j in 1:N]
	Eavg = [mean(Ei[g]) for g in groups]
	groups, Eavg
end

function groupbyenergy(wf::WaveFunctions, tol)
	groupbyenergy(wf.energy, tol)
end

struct GroupedWaveFunctions{FLOAT, FSET, PFUNC} <: AbstractFunctionSet
    energy::Vector{FLOAT}
    wavefunctions::WaveFunctions{FLOAT, FSET, PFUNC}
    groups::Vector{Vector{Int64}}
end

function GroupedWaveFunctions(wf::WaveFunctions, tol=0.01eV)
    gp, en = groupbyenergy(wf.energy, tol)
    GroupedWaveFunctions(en, wf, gp)
end

energy(gwf::GroupedWaveFunctions) = gwf.energy

function state(wf::WaveFunctions, n::Integer, x::Real)
    wf.waves[n](x)^2
end

function state(wf::WaveFunctions, n::Integer, x::Real, y::Real)
    wf.waves[n](x, y)^2
end

function state(wf::WaveFunctions, n::Integer, x::Real, y::Real, z::Real)
    wf.waves[n](x, y, z)^2
end

function state(wf::GroupedWaveFunctions, n::Integer, x::Real)
    sum(state(wf.wavefunctions, i, x) for i in wf.groups[n])
end

function state(wf::GroupedWaveFunctions, n::Integer, x::Real, y::Real)
    sum(state(wf.wavefunctions, i, x, y) for i in wf.groups[n])
end

function state(wf::GroupedWaveFunctions, n::Integer, x::Real, y::Real, z::Real)
    sum(state(wf.wavefunctions, i, x, y, z) for i in wf.groups[n])
end