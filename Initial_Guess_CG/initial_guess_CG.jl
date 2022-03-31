include("../src/diagonal_sbp.jl")
include("../src/assembling_SBP.jl")
include("../src/GPU_CG.jl")
include("../Preconditioned_CG/two_level_mg.jl")

using IterativeSolvers

function initial_guess_interpolation_CG(A,b,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    x_2h = A_2h \ b_2h
    x_interpolated = prolongation_2d(Nx_2h) * x_2h
    # x,history = cg!(x_interpolated,A,b;abstol = abstol,log=true)
    num_iter_steps, norms = CG_CPU(A,b,x_interpolated;abstol=abstol)
    x = x_interpolated
    return x,num_iter_steps,norms[end]
end

function initial_guess_interpolation_CG_GPU(A_GPU_sparse,b_GPU_sparse,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    x_2h = A_2h \ b_2h
    x_interpolated_sparse = prolongation_2d_GPU(Nx_2h) * CuArray(x_2h)
    # @show size(x_interpolated_sparse)
    # @show size(A_GPU_sparse)
    # @show size(b_GPU_sparse)
    num_iter_steps, norms_GPU = CG_GPU_sparse(x_interpolated_sparse,A_GPU_sparse,b_GPU_sparse;abstol=abstol)
    x = x_interpolated_sparse
    return x, num_iter_steps, norms_GPU[end]
end


# function MG_interpolation_CG_Matrix_Free_GPU(A_GPU,b_GPU,b_2h,x,Nx_2h;Nx=Nx,Ny=Ny,A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
#     x_MG_initial_guess_reverse = Two_level_multigrid(A,b,Nx,Ny,A_2h;nu=100,NUM_V_CYCLES=1,SBPp=2)[1]
#     x_MG_initial_guess_reverse_reshaped = reshape(x_MG_initial_guess_reverse,Nx,Ny)
#     x_MG_initial_guess = CuArray(reverse(x_MG_initial_guess_reverse_reshaped,dims=2))
#     nums_CG_Matrix_Free_GPU = 0
#     norms_Matrix_Free = [0]
#     # nums_CG_Matrix_Free_GPU, norms_Matrix_Free = CG_Matrix_Free_GPU_v2(x_MG_initial_guess,Ap_GPU,b_GPU,Nx,Ny;abstol=sqrt(eps(real(eltype(b_GPU))))) 
#     # x = reverse(x_MG_initial_guess[:])
#     return x, nums_CG_Matrix_Free_GPU, norms_Matrix_Free[end]
# end

function initial_guess_interpolation_three_level_CG_GPU(A_GPU_sparse,A_2h_GPU_sparse,b_GPU,b_2h_GPU,b_2h,b_4h,x,Nx_2h,Nx_4h;A_2h = A_2h_lu, A_4h = A_4h_lu,abstol=abstol,maxiter=length(b))
    x_4h = A_4h \ b_4h
    x_2h_interpolated  = prolongation_2d_GPU(Nx_4h) * CuArray(x_4h)
    x_2h_interpolated,history_2h = cg!(x_2h_interpolated,A_2h_GPU_sparse,b_2h_GPU;abstol=abstol,log=true)
    x_interpolated = prolongation_2d_GPU(Nx_2h) * CuArray(x_2h_interpolated)
    x,history_h = cg!(x_interpolated,A_GPU_sparse,b_GPU;abstol=abstol,log=true)
    return x, history_2h.iters,history_h.iters,history_2h.data[:resnorm],history_h.data[:resnorm]
end


function initial_guess_interpolation_CG_Matrix_Free_GPU(A_GPU,b_GPU_v2,b_2h,x,Nx_2h;Nx=Nx,Ny=Ny,A_2h = A_2h_lu,abstol=abstol,maxiter=length(b),CG_Matrix_Free_GPU_v2=CG_Matrix_Free_GPU_v2)
    x_2h = A_2h \ b_2h
    x_interpolated = prolongation_2d_GPU(Nx_2h) * CuArray(x_2h)
    # x_interpolated = reverse(x_interpolated)
    # x_interpolated_reshaped = reshape(x_interpolated,size(b_GPU))
    x_interpolated_reshaped = reshape(x_interpolated,size(b_GPU_v2))
    Ap_GPU = similar(b_GPU_v2)

    nums_CG_Matrix_Free_GPU = 0
    norms = [0]

    nums_CG_Matrix_Free_GPU, norms =  CG_Matrix_Free_GPU_v2(x_interpolated_reshaped,Ap_GPU,b_GPU_v2,Nx,Nx;abstol=sqrt(eps(real(eltype(b_GPU_v2)))),maxiter=1000) 
    x = x_interpolated_reshaped[:]

    return x, nums_CG_Matrix_Free_GPU, norms[end]
end



function  initial_guess_interpolation_three_level_Matrix_Free_CG_GPU(A_GPU,A_2h_GPU,b_GPU_v2,b_2h_GPU_v2,b_2h,b_4h,x,Nx,Nx_2h,Nx_4h;A_2h = A_2h_lu, A_4h = A_4h_lu,abstol=abstol,maxiter=length(b),CG_Matrix_Free_GPU_v2=CG_Matrix_Free_GPU_v2)
    x_4h = A_4h \ b_4h
    # x_2h_interpolated  = prolongation_2d_GPU(Nx_4h) * CuArray(x_4h)
    # x_2h_interpolated_reshaped = reshape(x_2h_interpolated,size(b_2h_GPU_v2))
    # x_2h_interpolated_reshaped = CuArray(zeros(Nx_2h,Nx_2h))
    x_2h_interpolated_reshaped = similar(b_2h_GPU_v2)
    matrix_free_prolongation_2d_GPU(reshape(CuArray(x_4h),Nx_4h,Nx_4h),x_2h_interpolated_reshaped)
    Ap_GPU_2h = similar(b_2h_GPU_v2)
    Ap_GPU = similar(b_GPU_v2)
    nums_iters_2h, norms_2h =  CG_Matrix_Free_GPU_v2(x_2h_interpolated_reshaped,Ap_GPU_2h,b_2h_GPU_v2,Nx_2h,Nx_2h;abstol=sqrt(eps(real(eltype(b_GPU_v2))))) 
    # x_interpolated = prolongation_2d_GPU(Nx_2h) * x_2h_interpolated_reshaped[:]
    # x_interpolated_reshaped = reshape(x_interpolated,size(b_GPU_v2))
    # x_interpolated_reshaped = CuArray(zeros(Nx,Nx))
    x_interpolated_reshaped = similar(b_GPU_v2)
    matrix_free_prolongation_2d_GPU(x_2h_interpolated_reshaped,x_interpolated_reshaped)
    nums_iter=0
    norms=[0]

    # CG_Matrix_Free_GPU_v2(x_interpolated_reshaped,Ap_GPU,b_GPU_v2,Nx,Nx;abstol=sqrt(eps(real(eltype(b_GPU_v2))))) 
    # CG_Matrix_Free_GPU_v2(x_interpolated_reshaped,Ap_GPU,b_GPU_v2,Nx,Nx;abstol=sqrt(eps(real(eltype(b_GPU_v2))))) 

    nums_iter, norms = CG_Matrix_Free_GPU_v2(x_interpolated_reshaped,Ap_GPU,b_GPU_v2,Nx,Nx;abstol=sqrt(eps(real(eltype(b_GPU_v2)))),maxiter=1000) 

    return x_interpolated_reshaped, nums_iters_2h, nums_iter, norms_2h[end], norms[end]
end

function test_initial_guess_CG(;level=6,nu=3,ω=2/3,SBPp=2)
    # level=7
    nu=3
    ω=2/3
    SBPp=2

    (A,b,H_tilde,Nx,Ny,analy_sol) = Assembling_matrix(level;SBPp=SBPp);
    (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h,analy_sol_2h) = Assembling_matrix(level-1;SBPp=SBPp);
    (A_4h,b_4h,H_tilde_4h,Nx_4h,Ny_4h,analy_sol_4h) = Assembling_matrix(level-2;SBPp=SBPp);

    A_2h_GPU_sparse = CUDA.CUSPARSE.CuSparseMatrixCSC(A_2h)
    b_2h_GPU = CuArray(b_2h)
    b_2h_GPU_v2 = CuArray(reshape(reverse(b_2h),Nx_2h,Ny_2h))

    A_2h_lu = lu(A_2h)
    A_4h_lu = lu(A_4h)
  
    reltol = sqrt(eps(real(eltype(b))))
    x = zeros(Nx*Ny);
    # abstol = norm(A*x-b) * reltol
    abstol = reltol

    x_GPU = CuArray(zeros(Nx,Ny))
    b_GPU = CuArray(reshape(b,Nx,Ny))
    b_GPU_v2 = CuArray(reshape(reverse(b),Nx,Ny)) # Remember to reverse b_GPU when using matrix_free

    ω_richardson = 0.15
    h = 1/(Nx-1)
    # (M,R,H,I_p,_,I_r,IN) = precond_matrix(A,b;m=3,solver="richardson",ω_richardson=ω_richardson,h=h,SBPp=SBPp)


    A_GPU_sparse = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
    x_GPU_sparse = CuArray(zeros(Nx*Ny))
    b_GPU_sparse = CuArray(b)

  

    x_GPU = CuArray(zeros(Nx,Ny))
    x_GPU_flat = x_GPU[:]
    b_GPU_flat = CuArray(b)

    println("################# TEST ####################")
    @show Nx,Ny
    @show Base.summarysize(A) Base.summarysize(b)


    # x_initial_guess, iter_initial_guess_cg_CPU, norms_initial_guess_cg = initial_guess_interpolation_CG(A,b,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    # initial_guess_cg_error = sqrt((x_initial_guess - analy_sol)'*H_tilde*(x_initial_guess-analy_sol))

    # x_initial_guess_GPU, iter_initial_guess_cg_GPU, norms_initial_guess_cg_GPU = initial_guess_interpolation_CG_GPU(A_GPU_sparse,b_GPU_sparse,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    # initial_guess_cg_GPU_error = sqrt((Array(x_initial_guess_GPU) - analy_sol)'*H_tilde*(Array(x_initial_guess_GPU)-analy_sol))

    x_initial_guess_Matrix_Free_GPU, iter_initial_guess_cg_Matrix_Free_GPU, norm_initial_guess_cg_Matrix_Free_GPU = initial_guess_interpolation_CG_Matrix_Free_GPU(A_GPU_sparse,b_GPU_v2,b_2h,x,Nx_2h;Nx=Nx,Ny=Ny,A_2h = A_2h_lu,abstol=abstol,maxiter=length(b_GPU))
    initial_guess_cg_Matrix_Free_GPU_error = sqrt((Array(x_initial_guess_Matrix_Free_GPU[:])-analy_sol)'*H_tilde*(Array(x_initial_guess_Matrix_Free_GPU[:])-analy_sol))

    x_initial_guess_three_level_GPU,iter_initial_guess_three_level_cg_GPU_2h,iter_initial_guess_three_level_cg_GPU_h, norm_initial_guess_three_level_cg_GPU_2h,norm_initial_guess_three_level_cg_GPU_h = initial_guess_interpolation_three_level_CG_GPU(A_GPU_sparse,A_2h_GPU_sparse,b_GPU,b_2h_GPU,b_2h,b_4h,x,Nx_2h,Nx_4h;A_2h = A_2h_lu, A_4h = A_4h_lu,abstol=abstol,maxiter=length(b))
    initial_guess_three_level_cg_GPU_error = sqrt((Array(x_initial_guess_three_level_GPU) - analy_sol)'*H_tilde*(Array(x_initial_guess_three_level_GPU) - analy_sol))

    x_initial_guess_three_level_Matrix_Free_GPU,iter_initial_guess_three_level_cg_Matrix_Free_GPU_2h,iter_initial_guess_three_level_cg_Matrix_Free_GPU_h, norm_initial_guess_three_level_cg_Matrix_Free_GPU_2h,norm_initial_guess_three_level_cg_Matrix_Free_GPU_h = initial_guess_interpolation_three_level_Matrix_Free_CG_GPU(A_GPU_sparse,A_2h_GPU_sparse,b_GPU_v2,b_2h_GPU_v2,b_2h,b_4h,x,Nx,Nx_2h,Nx_4h;A_2h = A_2h_lu, A_4h = A_4h_lu,abstol=abstol,maxiter=length(b))
    initial_guess_three_level_cg_Matrix_Free_GPU_error = sqrt((Array(x_initial_guess_three_level_Matrix_Free_GPU[:]) - analy_sol)'*H_tilde*(Array(x_initial_guess_three_level_Matrix_Free_GPU[:]) - analy_sol))

    println("############################################# START TIMING ####################################################")

    REPEAT = 5

    # t_initial_guess_CPU = @elapsed for _ in 1:REPEAT
    #     initial_guess_interpolation_CG(A,b,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    # end

    # t_initial_guess_GPU = @elapsed for _ in 1:REPEAT
    #     initial_guess_interpolation_CG_GPU(A_GPU_sparse,b_GPU_sparse,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    # end

    t_initial_guess_Matrix_Free_GPU = @elapsed for _ in REPEAT
        initial_guess_interpolation_CG_Matrix_Free_GPU(A_GPU_sparse,b_GPU_v2,b_2h,x,Nx_2h;Nx=Nx,Ny=Ny,A_2h = A_2h_lu,abstol=abstol,maxiter=length(b_GPU))
    end

    t_initial_guess_three_level_GPU = @elapsed for _ in 1:REPEAT
        initial_guess_interpolation_three_level_CG_GPU(A_GPU_sparse,A_2h_GPU_sparse,b_GPU,b_2h_GPU,b_2h,b_4h,x,Nx_2h,Nx_4h;A_2h = A_2h_lu, A_4h = A_4h_lu,abstol=abstol,maxiter=length(b))
    end

    t_intial_guess_three_level_Matrix_Free_GPU = @elapsed for _ in 1:REPEAT
        initial_guess_interpolation_three_level_Matrix_Free_CG_GPU(A_GPU_sparse,A_2h_GPU_sparse,b_GPU_v2,b_2h_GPU_v2,b_2h,b_4h,x,Nx,Nx_2h,Nx_4h;A_2h = A_2h_lu, A_4h = A_4h_lu,abstol=abstol,maxiter=length(b))
    end

    # t_initial_guess_GPU /= REPEAT
    t_initial_guess_Matrix_Free_GPU /= REPEAT
    t_initial_guess_three_level_GPU /= REPEAT
    t_intial_guess_three_level_Matrix_Free_GPU /= REPEAT

    # @show t_initial_guess_CPU, iter_initial_guess_cg_CPU
    # @show t_initial_guess_GPU, iter_initial_guess_cg_GPU
    @show t_initial_guess_Matrix_Free_GPU, iter_initial_guess_cg_Matrix_Free_GPU
    @show t_initial_guess_three_level_GPU, iter_initial_guess_three_level_cg_GPU_2h, iter_initial_guess_three_level_cg_GPU_h
    @show t_intial_guess_three_level_Matrix_Free_GPU, iter_initial_guess_three_level_cg_Matrix_Free_GPU_2h, iter_initial_guess_three_level_cg_Matrix_Free_GPU_h

    println()

    # @show initial_guess_cg_error
    # @show initial_guess_cg_GPU_error

    # @show initial_guess_cg_Matrix_Free_GPU_error
    # @show initial_guess_three_level_cg_GPU_error
    # @show initial_guess_three_level_cg_Matrix_Free_GPU_error

    println()

end