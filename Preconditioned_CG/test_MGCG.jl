include("MGCG.jl")
include("../src/assembling_SBP.jl")


function test_matrix_free_MGCG(;level=6,nu=3,ω=2/3,SBPp=2)
    (A,b,H_tilde,Nx,Ny,analy_sol) = Assembling_matrix(level;SBPp=SBPp);
    (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h,analy_sol_2h) = Assembling_matrix(level-1;SBPp=SBPp);
    (A_4h,b_4h,H_tilde_4h,Nx_4h,Ny_4h,analy_sol_4h) = Assembling_matrix(level-2;SBPp=SBPp);

    # Forming sparse CuArrays
    A_GPU_sparse = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
    x_GPU_sparse = CuArray(zeros(Nx*Ny))
    b_GPU_sparse = CuArray(b)

    A_2h_GPU_sparse = CUDA.CUSPARSE.CuSparseMatrixCSC(A_2h)
    b_2h_GPU = CuArray(b_2h)

    A_2h_lu = lu(A_2h)
    A_4h_lu = lu(A_4h)
    A_lu = lu(A)
    direct_sol = A_lu\b
    direct_err = sqrt((direct_sol-analy_sol)'*H_tilde*(direct_sol-analy_sol))

    x = zeros(Nx*Ny)
    reltol = sqrt(eps(real(eltype(b))))
    abstol = norm(A*x-b)*reltol

    x_GPU = CuArray(zeros(Nx,Ny))
    b_GPU = CuArray(reshape(b,Nx,Ny))

    x_CPU = zeros(Nx*Ny)

    ω_richardson = 0.15
    h = 1/(Nx-1)

    # Checking MGCG performance
    num_iter_steps_MGCG_matrix_free_GPU, norms_MGCG_matrix_free_GPU = matrix_free_MGCG(b_GPU,x_GPU;A_2h = A_2h_lu,maxiter=length(b_GPU),abstol=reltol,nu=nu)
    error_MGCG_matrix_free_GPU = sqrt((Array(x_GPU[:])-analy_sol)'*H_tilde*(Array(x_GPU[:])-analy_sol))

    
    x_MGCG_GPU, num_iter_steps_MGCG_GPU, norms_MGCG_GPU, errors_mg_cg_GPU = mg_preconditioned_CG_GPU(A_GPU_sparse,b_GPU_sparse,x_GPU_sparse;maxiter=length(b_GPU_sparse),A_2h = A_2h_lu, abstol=reltol,NUM_V_CYCLES=1,nu=nu,use_galerkin=true,H_tilde=H_tilde,SBPp=SBPp)
    error_MGCG_GPU = sqrt((Array(x_MGCG_GPU) - analy_sol)'*H_tilde*(Array(x_MGCG_GPU)-analy_sol))
    num_iters_MGCG_GPU = length(norms_MGCG_GPU) - 1

    x_MGCG_CPU, num_iter_steps_MGCG_CPU, norms_MGCG_CPU, errors_MGCG_CPU = mg_preconditioned_CG(A,b,x;maxiter=length(b),A_2h = A_2h_lu, abstol=reltol,NUM_V_CYCLES=1,nu=nu,use_galerkin=true,direct_sol=direct_sol,H_tilde=H_tilde,SBPp=SBPp)
    error_MGCG_CPU = sqrt((x_MGCG_CPU - analy_sol)'*H_tilde*(x_MGCG_CPU -analy_sol))
    num_iters_MGCG_CPU = length(norms_MGCG_CPU) - 1

    x_CPU,num_iters_CG_CPU = CG_CPU(A,b,x_CPU;abstol=reltol)
    error_CG_CPU = sqrt((x_CPU-analy_sol)'*H_tilde*(x_CPU-analy_sol))

    

    @show error_MGCG_matrix_free_GPU, num_iter_steps_MGCG_matrix_free_GPU
    @show error_MGCG_GPU, num_iters_MGCG_GPU
    @show error_MGCG_CPU, num_iters_MGCG_CPU
    @show error_CG_CPU, num_iters_CG_CPU


    # Benchmarking Time

    REPEAT = 2

    t_CG_CPU = @elapsed for _ in 1:REPEAT
        x_CPU .= 0
        CG_CPU(A,b,x_CPU;abstol=reltol)
    end

    t_MGCG_CPU = @elapsed for _ in 1:REPEAT
        x .= 0
        mg_preconditioned_CG(A,b,x;maxiter=length(b),A_2h = A_2h_lu, abstol=reltol,NUM_V_CYCLES=1,nu=nu,use_galerkin=true,direct_sol=direct_sol,H_tilde=H_tilde,SBPp=SBPp)
    end

    t_MGCG_GPU = @elapsed for _ in 1:REPEAT
        x_GPU_sparse .= 0
        mg_preconditioned_CG_GPU(A_GPU_sparse,b_GPU_sparse,x_GPU_sparse;maxiter=length(b_GPU_sparse),A_2h = A_2h_lu, abstol=reltol,NUM_V_CYCLES=1,nu=nu,use_galerkin=true,H_tilde=H_tilde,SBPp=SBPp)
    end

    t_matrix_free_MGCG_GPU = @elapsed for _ in 1:REPEAT
        x_GPU = CuArray(zeros(Nx,Ny))
        matrix_free_MGCG(b_GPU,x_GPU;A_2h=A_2h_lu,maxiter=length(b_GPU),abstol=reltol,nu=nu)
    end

    t_CG_CPU /= REPEAT
    t_MGCG_CPU /= REPEAT
    t_MGCG_GPU /= REPEAT
    t_matrix_free_MGCG_GPU /= REPEAT

    @show t_CG_CPU
    @show t_MGCG_CPU
    @show t_MGCG_GPU 
    @show t_matrix_free_MGCG_GPU

end


test_matrix_free_MGCG(;level=6)
test_matrix_free_MGCG(;level=7)
test_matrix_free_MGCG(;level=8)
test_matrix_free_MGCG(;level=9)
test_matrix_free_MGCG(;level=10)
test_matrix_free_MGCG(;level=11)
test_matrix_free_MGCG(;level=12)
test_matrix_free_MGCG(;level=13)