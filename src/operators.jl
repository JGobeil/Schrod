
using HCubature
using SharedArrays
using Distributed
using Base.Threads
using Printf

using Schrod
using SparseArrays


ifisnothing(v1, v2) = v1 !== nothing ? v1 : v2

function hamiltonian(potential, 
    gs::GaussianSet{T, D}, 
    tol = 1e-15;
    multithread = nothing,
    progressbar = nothing,
    ) where {T, D} # D is the dimension

    multithread = ifisnothing(multithread, _multithread)
    progressbar = ifisnothing(progressbar, _progressbar)

    # this is quick => no progressbar or multithread
    H, O = calculate_T_and_O(gs)

    # Modify H to add V. dispatch on _Options
    if multithread & progressbar
        _calculate_V_mt_pb!(H, O, potential, gs, tol)
    elseif multithread & !progressbar
        _calculate_V_mt!(H, O, potential, gs, tol)
    elseif !multithread & progressbar
        _calculate_V_pb!(H, O, potential, gs, tol)
    elseif !multithread & !progressbar
        _calculate_V!(H, O, potential, gs, tol)
    end

    H, O
end

"""
    calculate_T_and_O(gs::GaussianSet{T, D})

Calculate the kinetic operator and the overlap operator.

#TODO Can this be calculated once and reuse? Is it worth it?
"""
function calculate_T_and_O(gs::GaussianSet{T, D}) where {T, D}
    N = gs.N
    ν = gs.ν
    μ = gs.μ
	a = gs.a

    O = zeros(T, (N, N))
    H = zeros(T, (N, N))

	steps = Int64(N*(N+1)/2)
    @debug "Calculating kinetic (Tij) and overlap (Oij) " steps
    for j in 1:N
        for i in 1:j
			μij = μ[i, :] - μ[j, :]
            αij = 0.5ν*dot(μij, μij)
            Oij = a[i]*a[j]*(π/2ν)^(D/2)*exp(-αij)
            O[j, i] = O[i, j] = Oij
            Hij = -0.5ν * (2αij - D) * Oij
            H[i, j] = Hij
        end
    end
    H, O
end

function _limited_todo(O, gs, tol)
    [(i, j) for j in 1:gs.N for i in 1:j if abs(O[i, j]) > tol*abs(O[1, 1])]
end

function _calculate_V_mt_pb!(H, O, potential, gs, tol)
    # The list of [i, j] to do
    todo = _limited_todo(O, gs, tol)
    steps = length(todo)

    @debug "Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> " steps
    # Create the progressbar
    p = Progress(length(todo), 
        desc=@sprintf("Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> [N=%d (%.2f %%)] : ", 
            length(todo), 
            100*length(todo)/steps))

	σmax = maximum(gs.σ) * 6
    # the channel writter (the calculation)
    @threads for (i, j) in todo
        Vᵢⱼ = calculate_Vij(i, j, gs, potential, σmax)
        H[i, j] += Vᵢⱼ 
        H[j, i] = H[i, j]
        next!(p)
    end
end


function _calculate_V_mt!(H, O, potential, gs, tol)
    # The list of [i, j] to do
    todo = [(i, j) for j in 1:gs.N for i in 1:j if abs(O[i, j]) > tol*abs(O[1, 1])]
    steps = length(todo)

    @debug "Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> " steps
	σmax = maximum(gs.σ) * 6
    # the channel writter (the calculation)
    @threads for (i, j) in todo
        Vᵢⱼ = calculate_Vij(i, j, gs, potential, σmax)
        H[i, j] += Vᵢⱼ 
        H[j, i] = H[i, j]
    end
end


function _calculate_V_pb!(H, O, potential, gs, tol)
    # The list of [i, j] to do
    todo = [(i, j) for j in 1:gs.N for i in 1:j if abs(O[i, j]) > tol*abs(O[1, 1])]
    steps = length(todo)

    @debug "Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> " steps
    # Create the progressbar
    p = Progress(length(todo), 
        desc=@sprintf("Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> [N=%d (%.2f %%)] : ", 
            length(todo), 
            100*length(todo)/steps))

	σmax = maximum(gs.σ) * 6
    # the channel writter (the calculation)
    for (i, j) in todo
        Vᵢⱼ = calculate_Vij(i, j, gs, potential, σmax)
        H[i, j] += Vᵢⱼ 
        H[j, i] = H[i, j]
        next!(p)
    end
end


function _calculate_V!(H, O, potential, gs, tol)
    # The list of [i, j] to do
    todo = [(i, j) for j in 1:gs.N for i in 1:j if abs(O[i, j]) > tol*abs(O[1, 1])]
    steps = length(todo)

    @debug "Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> " steps

	σmax = maximum(gs.σ) * 6
    # the channel writter (the calculation)
    for (i, j) in todo
        Vᵢⱼ = calculate_Vij(i, j, gs, potential, σmax)
        H[i, j] += Vᵢⱼ 
        H[j, i] = H[i, j]
    end
end

function calculate_Vij(i, j, gs, potential, σmax)
    μ = gs.μ[[i, j], :]
    xmin = vec(minimum(μ, dims=1)) .- σmax
    xmax = vec(maximum(μ, dims=1)) .+ σmax
    vij, err = hcubature(Vij(i, j, gs, potential), xmin, xmax)
    vij
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
