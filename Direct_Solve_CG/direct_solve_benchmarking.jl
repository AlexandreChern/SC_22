using Random
using BenchmarkTools

include("../src/assembling_SBP.jl")

Random.seed!(123)

if length(ARGS) != 0
    level = parse(Int,ARGS[1])
    # Iterations = parse(Int,ARGS[2])
else
    level = 10
    # Iterations = 10000
end


function test_direct_solve(;max_level=13,SBPp=2)
    # io = IOBuffer()
    for level in 6:13
        (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level,p=SBPp);
        A_lu = lu(A)
        A_lu \ b
        @show Nx
        benchmark_result = @benchmark $A_lu \ $b;
        # show(io,"text/plain",benchmark_result)
        display(benchmark_result)
        println()
    end
end


test_direct_solve()
