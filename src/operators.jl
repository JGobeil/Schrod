
using HCubature
using SharedArrays
using Distributed
using Printf

using Gaussian
using Schrod
using SparseArrays

function hamiltonian(potential, gs::GaussianSet{T, D}, tol=1e-15) where {T, D} # D is the dimension
    N = gs.N

    @debug "Calculating Halmiltonian with Gaussian set in $(D)D" N

    ν = gs.ν
    μ = gs.μ
	a = gs.a

    #O = Array{Float64}(0.0, (N, N))
    #H = Array{Float64}(0.0, (N, N))

    O = zeros(Float64, (N, N))
    H = zeros(Float64, (N, N))


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

    @debug "Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> " steps

	σmax = maximum(gs.σ) * 6

    todo = [(i, j) for j in 1:N for i in 1:j if abs(O[i, j]) > tol*abs(O[1, 1])]

    p = Progress(size(todo, 1), 
        desc=@sprintf("Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ> [N=%d (%.2f %%)] : ", length(todo), 100*length(todo)/steps))
    channel = RemoteChannel(()->Channel{Tuple{Bool, Int64, Int64, Float64}}(32))

    @info "Running calculation on $(nprocs()) process"

    @async while (msg = take!(channel))[1]
        i = msg[2]; j = msg[3]; Vᵢⱼ = msg[4]
        H[i, j] += Vᵢⱼ 
        H[j, i] = H[i, j]
        next!(p)
    end

    @sync @distributed for (i, j) in shuffle(todo)
        Vᵢⱼ = calculate_Vij(i, j, gs, potential, σmax)
        put!(channel, (true, i, j, Vᵢⱼ))
    end

    put!(channel, (false, 0, 0, 0.0)) # calculation finished

    @debug "Calculation of Vᵢⱼ" integrated skipped totalerr  totalerr/integrated totalvij totalvij/integrated totalvij/N^2

    H, O
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




