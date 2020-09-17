
module Schrod

## export
export OT, V, solve
export WaveFunctions
export GroupedWaveFunctions
export state
export energy
export groupbyenergy
export dos
export integrate
export Vij

# Julia libs
using LinearAlgebra
using Statistics
import Base.show
using Random # for parallel loop

# Ext. libs
using ProgressMeter
using Cubature
using Clustering
using AtomicUnits: eV, nm

# Libs
#using Basis
using Gaussian

_showprogressbar = true

function showprogressbar(showit=true)
    global _showprogressbar
    _showprogressbar = showit
end


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

    if any(Ei .< 0.0)
        @debug "Some energy are negative" length(Ei) count(Ei .< 0.0) Ei[Ei .< 0.0]'./eV 
        #neg = Ei .< 0
        #pos = .~neg 
        #nneg = count(neg)
        #npos = count(pos)
        #tmp = copy(p)
        #p[1:npos] = tmp[nneg+1:end]
        #p[npos+1:end] = tmp[1:nneg]
    end

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


function show(io::IO, wf::WaveFunctions)
	print(io, "WaveFunctions with $(typeof(wf.functionset))")
	print(io, "\n\tN: $(length(wf.energy))")
	print(io, "\n\tenergy: $(wf.energy[1:8]/eV) eV")
	print(io, "\n")
end

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

#function (wf::WaveFunctions)(x::Number, y::Number, g::Vector, T::Number)
#	kb = 1.0
#	E = mean(wf.energy[g])
#	sum(exp(-abs(ϵ - E)/(kb*T))*w([x,y])^2 for (w, ϵ) in
#		zip(wf.waves[g], wf.energy[g]))
#end
#
#function (wf::WaveFunctions)(r::AbstractVector, E::Real, T::Number)
#	kb = 1.0
#	sum(exp(-abs(ϵ - E)/(kb*T))*w(r)^2 for (w, ϵ) in
#		zip(wf.waves[1:12], wf.energy[1:12]))
#end
#
#function (wf::WaveFunctions)(x::Number, E::Real, T::Number)
#	wf([x,], E, T)
#end
#
#function dos(wf::WaveFunctions, x, Δ)
#	[sum((xi-Δ/2) .<= wf.energy .< (xi+Δ/2)) for xi in x]
#end

struct GroupedWaveFunctions{FLOAT, FSET, PFUNC} <: AbstractFunctionSet
    energy::Vector{FLOAT}
    wavefunctions::WaveFunctions{FLOAT, FSET, PFUNC}
    groups::Vector{Vector{Int64}}
end

function show(io::IO, wf::GroupedWaveFunctions)
	print(io, "GroupedWaveFunctions with $(typeof(wf.wavefunctions.functionset))")
	print(io, "\n\tN: $(length(wf.energy)) ($(length(wf.wavefunctions.energy)))")
	print(io, "\n\teigenvalues: $(wf.energy[1:8]/eV) eV")
	print(io, "\n")
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

function solve(potential, fset::AbstractFunctionSet)
    @debug "Solving Schrodinger equation" potential fset
    H, O = operatorsHO(potential, fset)
	@debug "Diagolisation of the Hamiltonian"
    eigs = eigen(H, O)
    WaveFunctions(eigs, fset, potential)
end

struct Vij{I<:Integer, FS<:AbstractFunctionSet, F} <: Function
	i::I
	j::I
	fset::FS
	V::F
end

function (v::Vij)(r::AbstractArray)
	v.fset(v.i, r) * v.V(r) * v.fset(v.j, r)
end

function (v::Vij)(x::Number)
	r = [x, ]
	v.fset(v.i, r) * v.V(r) * v.fset(v.j, r)
end

integrate(V::Function, xmin, xmax; reltol=1e-8) = hcubature(V, xmin, xmax, reltol=reltol)

""" The overlap operator (O) calculation"""
function operatorO(gs::GaussianSet)
    N = gs.N
    ν = gs.ν
    μ = gs.μ
	a = gs.a
	steps = Int64(N*(N+1)/2)

    O = Array{Float64}(undef, (N, N))

    @debug "Calculating overlap (Oij) of $(D) Gaussian"
    for j in 1:N
        for i in 1:j
			μij = μ[i, :] - μ[j, :]
            αij = 0.5ν*dot(μij, μij)
            O[j, i] = O[i, j] = a[i]*a[j]*(π/2ν)^(D/2)*exp(-αij)
        end
    end
	O
end

""" The kinectic (T) operator calculation """
function operatorT(gs::GaussianSet, O=operatorO(gs))
    N = gs.N
    ν = gs.ν
    μ = gs.μ
	a = gs.a
	steps = Int64(N*(N+1)/2)
    @debug "Calculating kinetic energy (Tij) with $(D) Gaussian"

    if _showprogressbar 
        p = Progress(steps, desc="Calculating Tij")
    end
    for j in 1:N
        for i in 1:j
			μij = μ[i, :] - μ[j, :]
            H[i, j] = -0.5ν * (ν*dot(μij, μij) - D) * O[i, j]
            _showprogressbar && next!(p)
        end
    end
end

function operatorsHO(potential, gs::GaussianSet{T, D}) where {T, D}
    N = gs.N

    @debug "Calculating Halmiltonian with Gaussian set in $(D)D" N

    ν = gs.ν
    μ = gs.μ
	a = gs.a

    O = Array{Float64}(undef, (N, N))
    H = Array{Float64}(undef, (N, N))

	#x = collect(LinRange(xmin[1], xmax[1], 2000))

	steps = Int64(N*(N+1)/2)
    @debug "Calculating kinetic (Tij) and overlap (Oij) " steps
    for j in 1:N
        for i in 1:j
			μij = μ[i, :] - μ[j, :]
            αij = 0.5ν*dot(μij, μij)
            O[j, i] = O[i, j] = Oij = a[i]*a[j]*(π/2ν)^(D/2)*exp(-αij)
            H[i, j] = -0.5ν * (2αij - D) * Oij
        end
    end

    @debug "Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> " steps
	skipped = 0
    integrated = 0
    totalerr = 0.0
    totalvij = 0.0

	σmax = maximum(gs.σ) * 10
	#xmin = minimum(gs.μ, dims=1) .- σmax
    #xmax = maximum(gs.μ, dims=1) .+ σmax

    #Threads.@threads for j in 1:N
    #Threads.@threads for j in randperm(N) # Use a random array to *evently* paralize

    if _showprogressbar 
        p = Progress(steps, desc="Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ>: ") 
    end
    for j in 1:N 
        for i in 1:j
            if O[i, j] > 1e-8
                μ = gs.μ[[i, j], :]
                xmin = minimum(μ, dims=1) .- σmax
                xmax = maximum(μ, dims=1) .+ σmax
                #vij, err = integrate(Vij(i, j, gs, potential), xmin, xmax, reltol=1e-10)
                vij, err = hcubature(Vij(i, j, gs, potential), xmin, xmax, reltol=1e-8, abstol=1e-10)
                integrated += 1
                totalerr += err
                totalvij += vij
			else
				vij = 0.0
				skipped += 1
			end
            H[i, j] += vij
			H[j, i] = H[i, j]
            _showprogressbar && next!(p)
        end
    end
    errmean = totalerr / integrated
    vijmean = totalvij / integrated
	@debug "Calculation of Vᵢⱼ" integrated skipped totalerr  totalerr/integrated totalvij totalvij/integrated totalvij/N^2

    H, O
end

end
