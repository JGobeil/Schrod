
module Schrod

## export
export OT, V, solve
export WaveFunctions
export GroupedWaveFunctions
export state, state!
export energy
export groupbyenergy
export dos
export integrate
export Vij

# Julia libs
using LinearAlgebra
using Statistics
using Random # for parallel loop
using SparseArrays

# Ext. libs
using ProgressMeter
using Clustering
using Arpack

using AtomicUnits: eV, nm


# Libs
#using Basis
using Gaussian

include("wavefunctions.jl")
include("print.jl")
include("operators.jl")

_progressbar = true
_multithread = false


"""
Change the default options on how Schrod work (global option). The options 
should be overriderable with arguments in the function calls.

# Options:
    progressbar: show a progress during the calculation if true
    multithread: multithread some calculations if true

# example:    
    schrod_options(progressbar=true)
"""
function schrod_options(;progressbar=nothing, multithread=nothing)
    global _progressbar, _multithread
    progressbar !== nothing && (_progressbar = progressbar)
    multithread    !== nothing && (_multithread = multithread)
end

"""
    solve(potential, fset::AbstractFunctionSet, 
          maxstates=100; tol = 1e-15,
          progressbar=nothing, multithreaded=nothing)

Find the first `maxstates` of solutions for a `potential` using the functions
set `fset`. `Return a WaveFunctions`. `tol` is used to limit the calculation 
of the integrals <ϕᵢ|V|ϕⱼ> for functions where <ϕᵢ|ϕⱼ> > `tol`.
"""
function solve(potential, fset::AbstractFunctionSet;
               maxstates=100, tol = 1e-15,
               progressbar=nothing, multithread=nothing)

    @debug "Solving Schrodinger equation" potential fset
    H, O = hamiltonian(potential, fset, tol; progressbar, multithread)

	@debug "Diagolisation of the Hamiltonian"

    ## weird bug on windows with matplotlib call (see double_eval_bug_test.jl)
    #Sys.iswindows() && eigen(collect(I(50)), collect(I(50)))

    eigens = eigs(H, O; nev=maxstates, which=:SM)

    #@debug "Eigen values 1:10" eigs.values[1:10]

    WaveFunctions(eigens[1], eigens[2], fset, potential, H, O)
end

end

##