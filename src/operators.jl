
integrate(V::Function, xmin, xmax; reltol=1e-8) = hcubature(V, xmin, xmax, reltol=reltol)

function hamiltonian(potential, gs::GaussianSet{T, D}) where {T, D}
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

    _progressbar && (p = Progress(steps, desc="Calculating Vᵢⱼ = <ϕᵢ|V|ϕⱼ>: "))

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
            _progressbar && next!(p)
        end
    end
    errmean = totalerr / integrated
    vijmean = totalvij / integrated
	@debug "Calculation of Vᵢⱼ" integrated skipped totalerr  totalerr/integrated totalvij totalvij/integrated totalvij/N^2

    O, H
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




