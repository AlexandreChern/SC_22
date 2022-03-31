function test_initial_guess_CG(;level=6,nu=3,ω=2/3,SBPp=2)
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




    
end