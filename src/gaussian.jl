
"Gaussian function for and set in 1D, 2D and 3D"

export gaussian, GaussianSet, GaussianSetHex2D

import Base.size
import Base.show

default_overlap = 2.0

"Convertion from σ to ν."
sigmatonu(σ) = 0.5σ^-2

"Convertion from ν to σ."
nutosigma(ν) = 1/sqrt(2ν)

"Compute a 'good' value for sigma to get a good coverage of the space."
nicesigma(μ::AbstractMatrix, d=default_overlap) = nicesigma.((view(μ, :, i) for i=1:size(μ, 2)), d)
nicesigma(μ::AbstractVector, d=default_overlap) = nicesigma(minimum(μ), maximum(μ), size(μ, 1), d)
nicesigma(μmin, μmax, N, d=default_overlap) = nicesigma((μmax - μmin)/(N - 1), d)
"Δ: distance between two gaussian peak. Δ*d: distance for half value of the guaussian"
function nicesigma(Δ::Real, d=default_overlap)
    Δ*d/sqrt(log(4))
end


"Compute the normalisation so <ϕi|ϕi> == 1."
norm(νx) = (2*νx/π)^(0.25)
norm(νx, νy) = (4*νx*νy/π^2)^(0.25)
norm(νx, νy, νz) = (8*νx*νy*νz/π^3)^(0.25)
norm(ν, dim::Integer) = (2*ν/π)^(0.25dim)

"Compute the value at 'x' of a gaussian"
function gaussian(x, μ, ν, a=norm(ν, 1))
    a*exp(-ν*(x - μ)^2)
end

function gaussian(x, y, μx, μy, ν, a=norm(ν, 2))
    a*exp(-ν*((x - μx)^2+(y - μy)^2))
end

function gaussian(x, y, z, μx, μy, μz, ν, a=norm(ν, 3))
    a*exp(-ν*((x - μx)^2+(y - μy)^2+(z - μz)^2))
end

## Gaussian Set
## g(x) = a * exp(ν * (x - μ)^2)
struct GaussianSet{T<:Real, dim} <: AbstractFunctionSet
    σ::T    # sigma
    ν::T    # nu
    μ::Matrix{T}  # position
    a::Vector{T}  # normalization
    N::Int64
end

size(gs::GaussianSet, i) = size(gs.μ, i)
size(gs::GaussianSet) = size(gs.μ)

function show(io::IO, gs::GaussianSet{T, dim}) where {T, dim}
    print(io, "\n$(dim)D Gaussian set with $T (GaussianSet{$T, 1})")
    print(io, "\n\tN: $(gs.N)")
    print(io, "\n\tσ: $(gs.σ)\tν: $(gs.ν)")
    for (s, i) in zip("xyz", 1:dim)
        print(io, "\n\tμ$s: $(minimum(gs.μ[:, i])) .. $(maximum(gs.μ[:, i]))")
    end
end

#function (gs::GaussianSet)(r::AbstractVector) where {T}
#    sum(gs(i, r) for i in 1:gs.N)
#end
#
#function (gs::GaussianSet)(i::Integer, r::AbstractVector) where {T}
#    gaussian(r, gs.μ[i, :], gs.ν, gs.a[i])
#end

## 1D Gaussian

function (gs::GaussianSet{T, 1})(x::Real) where {T}
    sum(gs(i, x) for i in 1:gs.N)
end

function (gs::GaussianSet{T, 1})(r::AbstractVector) where {T}
    sum(gs(i, r[1]) for i in 1:gs.N)
end

function (gs::GaussianSet{T, 1})(i::Integer, x::Real) where {T}
    gaussian(x, gs.μ[i], gs.ν, gs.a[i])
end

function (gs::GaussianSet{T, 1})(i::Integer, r::AbstractVector) where {T}
    gaussian(r[1], gs.μ[i], gs.ν, gs.a[i])
end

## 2D Gaussian

function (gs::GaussianSet{T, 2})(x::Real, y::Real) where {T}
    sum(gs(i, x, y) for i in 1:gs.N)
end

function (gs::GaussianSet{T, 2})(r::AbstractArray) where {T}
    sum(gs(i, r[1], r[2]) for i in 1:gs.N)
end

function (gs::GaussianSet{T, 2})(i::Integer, x::Real, y::Real) where {T}
    gaussian(x, y,
             gs.μ[i, 1], gs.μ[i, 2],
             gs.ν,
             gs.a[i])
end

function (gs::GaussianSet{T, 2})(i::Integer, r::AbstractArray) where {T}
    gaussian(r[1], r[2],
             gs.μ[i, 1], gs.μ[i, 2],
             gs.ν,
             gs.a[i])
end

## 3D Gaussian

function (gs::GaussianSet{T, 3})(x::Real, y::Real, z::Real) where {T}
    sum(gs(i, x, y, z) for i in 1:gs.N)
end

function (gs::GaussianSet{T, 3})(r::AbstractArray) where {T}
    sum(gs(i, r[1], r[2], r[3]) for i in 1:gs.N)
end

function (gs::GaussianSet{T, 3})(i::Integer, x::Real, y::Real, z::Real) where {T}
    gaussian(x, y, z,
             gs.μ[i, 1], gs.μ[i, 2], gs.μ[i, 3],
             gs.ν,
             gs.a[i])
end

function (gs::GaussianSet{T, 3})(i::Integer, r::AbstractArray) where {T}
    gaussian(r[1], r[2], r[3],
             gs.μ[i, 1], gs.μ[i, 2], gs.μ[i, 3],
             gs.ν,
             gs.a[i])
end

## Constructors

function GaussianSet(μ::AbstractArray, σ::Real, a=nothing)
    μ = convert(Array, μ)

    if any((maximum(μ; dims=1) .- minimum(μ; dims=1)) .< 1e-10)
        @warn "Spacing between gaussian functions is really small"
    end

    N = size(μ, 1)
    dim = size(μ, 2)
    ν = sigmatonu(σ)
    #a = (a === nothing) ? fill(norm(ν, dim), N) : convert(Array, a)
    a = (a === nothing) ? fill(1.0, N) : convert(Array, a)
    GaussianSet{eltype(μ), dim}(σ, ν, μ, a, N)
end


function GaussianSet(xmin, xmax, xn, Δ=default_overlap)
    μx = LinRange(xmin, xmax, xn)

    μ = Matrix{eltype(μx)}(undef, xn, 1)
    μ[:, 1] = [μx[i] for i=1:xn]

    GaussianSet(μ, maximum(nicesigma(step(μx), Δ)))
end


function GaussianSet(xmin, xmax, xn,
                     ymin, ymax, yn, Δ=default_overlap)
    μx = LinRange(xmin, xmax, xn)
    μy = LinRange(ymin, ymax, yn)

    μ = Matrix{eltype(μx)}(undef, xn*yn, 2)
    μ[:, 1] = [μx[i] for i=1:xn for j=1:yn]
    μ[:, 2] = [μy[j] for i=1:xn for j=1:yn]

    GaussianSet(μ, maximum(nicesigma.(step.((μx, μy)), Δ)))
end

function GaussianSetHex2D(xmin, xmax, xn,
                          ymin, ymax, Δ=default_overlap)
    μx = LinRange(xmin, xmax, xn)
    R = step(μx)

    Δy = sqrt(3R^2/4)
    μy = collect(range(ymin, ymax+Δy, step=Δy))
    yn = length(μy)
    if iseven(yn)
        μy = μy[1:end-1]
        yn -= 1
    end

    mid1 = (ymax + ymin)/2
    mid2 = (maximum(μy) + minimum(μy))/2
    μy .-= (mid2 - mid1)

    μtmp = Matrix{eltype(μx)}(undef,  xn*yn, 2)
    μtmp[:, 1] = [μx[i] + (iseven(j) ? R/2 : 0.0) for i=1:xn for j=1:yn]
    μtmp[:, 2] = [μy[j] for i=1:xn for j=1:yn]

    keep = μtmp[:, 1] .<= xmax
    N = sum(keep)
    μ = Matrix{eltype(μx)}(undef, N, 2)
    μ[:, 1] = μtmp[keep, 1]
    μ[:, 2] = μtmp[keep, 2]

    GaussianSet(μ, nicesigma(R, Δ))
end

function GaussianSet(xmin, xmax, xn,
                     ymin, ymax, yn,
                     zmin, zmax, zn,
                     Δ=default_overlap)
    μx = LinRange(xmin, xmax, xn)
    μy = LinRange(ymin, ymax, yn)
    μz = LinRange(zmin, zmax, zn)

    μ = Matrix{eltype(μx)}(undef, xn*yn*zn, 3)
    μ[:, 1] = [μx[i] for i=1:xn for j=1:yn for k=1:zn]
    μ[:, 2] = [μy[j] for i=1:xn for j=1:yn for k=1:zn]
    μ[:, 3] = [μz[k] for i=1:xn for j=1:yn for k=1:zn]

    GaussianSet(μ, maximum(nicesigma.(step.((μx, μy)), Δ)))
end
