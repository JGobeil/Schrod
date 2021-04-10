
using HCubature
using SharedArrays
using Distributed
using Printf

using Gaussian
using Schrod
using SparseArrays


ifisnothing(v1, v2) = v1 !== nothing ? v1 : v2

struct _Options{Bool, Bool} 
    _Options(progressbar, multithread) = new{
        ifisnothing(progressbar, _progressbar),
        ifisnothing(multithread, _multithread),
        }()
end


function hamiltonian(potential, 
    gs::GaussianSet{T, D}, 
    tol = 1e-15;
    progressbar = nothing,
    multithread = nothing,
    opt = _Options(progressbar, multithread)
    ) where {T, D} # D is the dimension

    # this is quick => no progressbar or multithread
    H, O = calculate_T_and_O(gs)

    # Modify H to add V. dispatch on _Options
    _calculate_V!(H, O, potential, gs, tol, opt)

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

function _calculate_V!(H, O, potential, gs, tol, ::Schrod._Options{true, true})
    # The list of [i, j] to do
    todo = [(i, j) for j in 1:gs.N for i in 1:j if abs(O[i, j]) > tol*abs(O[1, 1])]
    steps = length(todo)

    @debug "Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> " steps
    # Create the progressbar
    p = Progress(length(todo), 
        desc=@sprintf("Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> [N=%d (%.2f %%)] : ", 
            length(todo), 
            100*length(todo)/steps))

    # Cammunication channel for completed calculation
    channel = RemoteChannel(()->Channel{Tuple{Bool, Int64, Int64, Float64}}(32))

    # create channel readears
    @async while (msg = take!(channel))[1] # stop if msg[1] == false
        i = msg[2]; j = msg[3]; Vᵢⱼ = msg[4]
        H[i, j] += Vᵢⱼ 
        H[j, i] = H[i, j]
        next!(p)
    end

    # distance max for integrtion
	σmax = maximum(gs.σ) * 6

    # the channel writter (the calculation)
    @sync @distributed for (i, j) in shuffle(todo)
        Vᵢⱼ = calculate_Vij(i, j, gs, potential, σmax)
        put!(channel, (true, i, j, Vᵢⱼ))
    end

    # calculation finished
    put!(channel, (false, 0, 0, 0.0)) 
end


function _calculate_V!(H, O, potential, gs, tol, ::_Options{false, true})
    # The list of [i, j] to do
    todo = [(i, j) for j in 1:gs.N for i in 1:j if abs(O[i, j]) > tol*abs(O[1, 1])]
    steps = length(todo)

    @debug "Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> " steps
    # Create the progressbar
    p = Progress(length(todo), 
        desc=@sprintf("Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> [N=%d (%.2f %%)] : ", 
            length(todo), 
            100*length(todo)/steps))

    # Cammunication channel for completed calculation
    channel = RemoteChannel(()->Channel{Tuple{Bool, Int64, Int64, Float64}}(32))

    # create channel readears
    @async while (msg = take!(channel))[1] # stop if msg[1] == false
        i = msg[2]; j = msg[3]; Vᵢⱼ = msg[4]
        H[i, j] += Vᵢⱼ 
        H[j, i] = H[i, j]
    end

    # distance max for integrtion
	σmax = maximum(gs.σ) * 6

    # the channel writter (the calculation)
    @sync @distributed for (i, j) in shuffle(todo)
        Vᵢⱼ = calculate_Vij(i, j, gs, potential, σmax)
        put!(channel, (true, i, j, Vᵢⱼ))
    end

    # calculation finished
    put!(channel, (false, 0, 0, 0.0)) 
end


function _calculate_V!(H, O, potential, gs, tol, ::_Options{true, false})
    # The list of [i, j] to do
    todo = [(i, j) for j in 1:gs.N for i in 1:j if abs(O[i, j]) > tol*abs(O[1, 1])]
    steps = length(todo)

    @debug "Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> " steps
    # Create the progressbar
    p = Progress(length(todo), 
        desc=@sprintf("Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> [N=%d (%.2f %%)] : ", 
            length(todo), 
            100*length(todo)/steps))

    # distance max for integration
	σmax = maximum(gs.σ) * 6

    for (i, j) in todo
        H[i, j] += calculate_Vij(i, j, gs, potential, σmax) 
        H[j, i] = H[i, j]
        next!(p)
    end
end


function _calculate_V!(H, O, potential, gs, tol, ::_Options{false, false})
    # The list of [i, j] to do
    todo = [(i, j) for j in 1:gs.N for i in 1:j if abs(O[i, j]) > tol*abs(O[1, 1])]
    steps = length(todo)

    # distance max for integration
	σmax = maximum(gs.σ) * 6

    # the channel writter (the calculation)
    for (i, j) in todo
        H[i, j] += calculate_Vij(i, j, gs, potential, σmax) 
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


##
 
#module Testing
#
#struct OptionsTest{V1, V2} 
#    OptionsTest(v1, v2) = new{
#        v1, 
#        v2
#        }()
#end
#
#f1(v1, ::OptionsTest{true, true})   = println("$v1 - true - true")
#f1(v1, ::OptionsTest{true, false})  = println("$v1 - true - false")
#f1(v1, ::OptionsTest{false, true})  = println("$v1 - false - true")
#f1(v1, ::OptionsTest{false, false}) = println("$v1 - false - false")
#end
#
#
#Testing.f1(10, Testing.OptionsTest(true, true))
#Testing.f1(15, Testing.OptionsTest(false, true))
#Testing.f1(20, Testing.OptionsTest(true, false))
#Testing.f1(25, Testing.OptionsTest(false, false))
#
##

