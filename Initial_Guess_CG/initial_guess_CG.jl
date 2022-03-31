include("../src/diagonal_sbp.jl")
include("../src/assembling_SBP.jl")
include("../src/GPU_CG.jl")
include("../Preconditioned_CG/two_level_mg.jl")

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


function initial_guess_interpolation_CG_Matrix_Free_GPU(A_GPU,b_GPU,b_2h,x,Nx_2h;Nx=Nx,Ny=Ny,A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    x_2h = A_2h \ b_2h
    x_interpolated = prolongation_2d_GPU(Nx_2h) * CuArray(x_2h)
    x_interpolated = reverse(x_interpolated)
    x_interpolated_reshaped = reshape(x_interpolated,size(b_GPU))
    Ap_GPU = similar(b_GPU)
    nums_CG_Matrix_Free_GPU, norms =  CG_Matrix_Free_GPU_v2(x_interpolated_reshaped,Ap_GPU,b_GPU,Nx,Ny;abstol=sqrt(eps(real(eltype(b_GPU))))) 
    x = reverse(x_interpolated_reshaped[:])
    return x, nums_CG_Matrix_Free_GPU, norms[end]
end

function MG_interpolation_CG_Matrix_Free_GPU(A_GPU,b_GPU,b_2h,x,Nx_2h;Nx=Nx,Ny=Ny,A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    x_MG_initial_guess_reverse = Two_level_multigrid(A,b,Nx,Ny,A_2h;nu=100,NUM_V_CYCLES=1,SBPp=2)[1]
    x_MG_initial_guess_reverse_reshaped = reshape(x_MG_initial_guess_reverse,Nx,Ny)
    x_MG_initial_guess = CuArray(reverse(x_MG_initial_guess_reverse_reshaped,dims=2))
    nums_CG_Matrix_Free_GPU, norms_Matrix_Free = CG_Matrix_Free_GPU_v2(x_MG_initial_guess,Ap_GPU,b_GPU,Nx,Ny;abstol=sqrt(eps(real(eltype(b_GPU))))) 
    x = reverse(x_MG_initial_guess[:])
    return x, nums_CG_Matrix_Free_GPU, norms_Matrix_Free[end]
end

function initial_guess_interpolation_three_level_CG_GPU(A_GPU,A_2h_GPU,b_GPU,b_2h_GPU,b_2h,b_4h,x,Nx_2h,Nx_4h;A_2h = A_2h_lu, A_4h = A_4h_lu,abstol=abstol,maxiter=length(b))
    x_4h = A_4h \ b_4h
    x_2h_interpolated  = prolongation_2d_GPU(Nx_4h) * CuArray(x_4h)
    x_2h_interpolated,history_2h = cg!(x_2h_interpolated,A_2h_GPU,b_2h_GPU;abstol=abstol,log=true)
    x_interpolated = prolongation_2d_GPU(Nx_2h) * CuArray(x_2h_interpolated)
    x,history_h = cg!(x_interpolated,A_GPU,b_GPU;abstol=abstol,log=true)
    return x, history_2h.iters,history_h.iters,history_2h.data[:resnorm],history_h.data[:resnorm]
end


function test()
    level=6
    nu=3
    ω=2/3
    SBPp=2

    (A,b,H_tilde,Nx,Ny,analy_sol) = Assembling_matrix(level;SBPp=SBPp);
    (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h,analy_sol_2h) = Assembling_matrix(level-1;SBPp=SBPp);
    (A_4h,b_4h,H_tilde_4h,Nx_4h,Ny_4h,analy_sol_4h) = Assembling_matrix(level-2;SBPp=SBPp);

    A_2h_GPU_sparse = CUDA.CUSPARSE.CuSparseMatrixCSC(A_2h)
    b_2h_GPU = CuArray(b_2h)

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


    x_initial_guess, iter_initial_guess_cg, norms_initial_guess_cg = initial_guess_interpolation_CG(A,b,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    initial_guess_cg_error = sqrt((x_initial_guess - analy_sol)'*H_tilde*(x_initial_guess-analy_sol))

    x_initial_guess_GPU, iter_initial_guess_cg_GPU, norms_initial_guess_cg_GPU = initial_guess_interpolation_CG_GPU(A_GPU_sparse,b_GPU_sparse,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    initial_guess_cg_GPU_error = sqrt((Array(x_initial_guess_GPU) - analy_sol)'*H_tilde*(Array(x_initial_guess_GPU)-analy_sol))

    
end