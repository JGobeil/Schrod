
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
using Random # for parallel loop

# Ext. libs
using ProgressMeter
using Cubature
using Clustering
using AtomicUnits: eV, nm

# Libs
#using Basis
using Gaussian

include("wavefunctions.jl")
include("print.jl")
include("operators.jl")

_progressbar = true
_threaded = false

"""
Change options on how Schrod work (global option).

# Options:
    progressbar: show(true) or hide(false) the progress bar during calculation
    threaded: use(true) or not(false) multithreading to evaluation the solution

# example:    
    schrod_options(progressbar=true)
"""
function schrod_options(;progressbar=nothing, threaded=nothing)
    global _progressbar, _threaded
    progressbar !== nothing && (_progressbar = progressbar)
    threaded    !== nothing && (_threaded    = threaded)
end

"""
    solve(potential, fset::AbstractFunctionSet; maxstates=-1)

Find the solution (WaveFunctions) for a `potential` using the functions
set `fset`. Find a maximum of `maxstates` states (-1 = infinity).
"""
function solve(potential, fset::AbstractFunctionSet; 
    progressbar = _progressbar,
    threaded = _threaded,
    maxstates=-1)
    @debug "Solving Schrodinger equation" potential fset
    O, H = hamiltonian(potential, fset)

	@debug "Diagolisation of the Hamiltonian"
    eigs = eigen(H, O)

    WaveFunctions(eigs, fset, potential)
end

end

##