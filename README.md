# Schrod

Schrod is an arbitrary potential time-independent n-dimension Schrodinger equation solver written in Julia.

The code implements a variational method using a basis of shifted Gaussian functions, but can also be easily extended to use any basis set. For basic usage, see `src/test.jl`.

For an introduction to the theory used in the implementation, you can consult section 3.3 of my PhD thesis (https://doi.org/10.4233/uuid:1ff19056-6af0-4b96-b278-84907c20ec77) 
