include("initial_guess_CG.jl")
using IterativeSolvers
using CUDA

# test_initial_guess_CG(;level=6)
# test_initial_guess_CG(;level=7)
# test_initial_guess_CG(;level=8)
# test_initial_guess_CG(;level=9)
# test_initial_guess_CG(;level=10)
test_initial_guess_CG(;level=11)
# test_initial_guess_CG(;level=12)
# test_initial_guess_CG(;level=13)