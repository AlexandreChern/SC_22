include("../src/diagonal_sbp.jl")
include("../src/split_matrix_free.jl")
using CUDA

if length(ARGS) != 0
    level = parse(Int,ARGS[1])
    Iterations = parse(Int,ARGS[2])
else
    level = 10
    Iterations = 10000
end


function prolongation_matrix(N)
    # SBP preserving
    # N = 2^level + 1
    odata = spzeros(2*N-1,N)
    for i in 1:2*N-1
        if i % 2 == 1
            odata[i,div(i+1,2)] = 1
        else
            odata[i,div(i,2)] = 1/2
            odata[i,div(i,2)+1] = 1/2
        end
    end
    return odata
end

function restriction_matrix(N)
    # SBP preserving
    odata = spzeros(div(N+1,2),N)
    odata[1,1] = 1/2
    odata[1,2] = 1/2
    odata[end,end-1] = 1/2
    odata[end,end] = 1/2
    for i in 2:div(N+1,2)-1
        odata[i,2*i-2] = 1/4
        odata[i,2*i-1] = 1/2
        odata[i,2*i] = 1/4
    end
    return odata
end

function restriction_matrix_normal(N)
    # SBP preserving
    odata = spzeros(div(N+1,2),N)
    odata[1,1] = 1/2
    odata[1,2] = 1/4
    odata[end,end-1] = 1/4
    odata[end,end] = 1/2
    for i in 2:div(N+1,2)-1
        odata[i,2*i-2] = 1/4
        odata[i,2*i-1] = 1/2
        odata[i,2*i] = 1/4
    end
    return odata
end

function prolongation_2d(N)
    prolongation_1d = prolongation_matrix(N)
    prolongation_2d = kron(prolongation_1d,prolongation_1d)
    return prolongation_2d
end

function restriction_2d(N)
    restriction_1d = restriction_matrix_normal(N)
    # restriction_1d = restriction_matrix(N)
    restriction_2d = kron(restriction_1d,restriction_1d)
    return restriction_2d
end


function restriction_2d_GPU(N)
    return CUDA.CUSPARSE.CuSparseMatrixCSC(restriction_2d(N))
end


function prolongation_2d_GPU(N)
    return CUDA.CUSPARSE.CuSparseMatrixCSC(prolongation_2d(N))
end


# Matrix-free Interpolation Functions
function matrix_free_prolongation_2d(idata,odata)
    size_idata = size(idata)
    odata_tmp = zeros(size_idata .* 2)
    for i in 1:size_idata[1]-1
        for j in 1:size_idata[2]-1
            odata[2*i-1,2*j-1] = idata[i,j]
            odata[2*i-1,2*j] = (idata[i,j] + idata[i,j+1]) / 2
            odata[2*i,2*j-1] = (idata[i,j] + idata[i+1,j]) / 2
            odata[2*i,2*j] = (idata[i,j] + idata[i+1,j] + idata[i,j+1] + idata[i+1,j+1]) / 4
        end
    end
    for j in 1:size_idata[2]-1
        odata[end,2*j-1] = idata[end,j]
        odata[end,2*j] = (idata[end,j] + idata[end,j+1]) / 2 
    end
    for i in 1:size_idata[1]-1
        odata[2*i-1,end] = idata[i,end]
        odata[2*i,end] = (idata[i,end] + idata[i+1,end]) / 2
    end
    odata[end,end] = idata[end,end]
    return nothing
end

function matrix_free_restriction_2d(idata,odata)
    size_idata = size(idata)
    size_odata = div.(size_idata .+ 1,2)
    idata_tmp = zeros(size_idata .+ 2)
    idata_tmp[2:end-1,2:end-1] .= idata

    for i in 1:size_odata[1]
        for j in 1:size_odata[2]
            odata[i,j] = (4*idata_tmp[2*i,2*j] + 
            2 * (idata_tmp[2*i,2*j-1] + idata_tmp[2*i,2*j+1] + idata_tmp[2*i-1,2*j] + idata_tmp[2*i+1,2*j]) +
             (idata_tmp[2*i-1,2*j-1] + idata_tmp[2*i-1,2*j+1] + idata_tmp[2*i+1,2*j-1]) + idata_tmp[2*i+1,2*j+1]) / 16
        end
    end
    return nothing
end

function prolongation_2D_kernel(idata,odata,Nx,Ny,::Val{TILE_DIM_1},::Val{TILE_DIM_2}) where {TILE_DIM_1,TILE_DIM_2}
    tidx = threadIdx().x
    tidy = threadIdx().y
    i = (blockIdx().x - 1) * TILE_DIM_1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM_2 + tidy


    if 1 <= i <= Nx-1 && 1 <= j <= Ny-1
        odata[2*i-1,2*j-1] = idata[i,j]
        odata[2*i-1,2*j] = (idata[i,j] + idata[i,j+1]) / 2
        odata[2*i,2*j-1] = (idata[i,j] + idata[i+1,j]) / 2
        odata[2*i,2*j] = (idata[i,j] + idata[i+1,j] + idata[i,j+1] + idata[i+1,j+1]) / 4
    end 

    if 1 <= j <= Ny-1
        odata[end,2*j-1] = idata[end,j]
        odata[end,2*j] = (idata[end,j] + idata[end,j+1]) / 2 
    end

    if 1 <= i <= Nx-1
        odata[2*i-1,end] = idata[i,end]
        odata[2*i,end] = (idata[i,end] + idata[i+1,end]) / 2
    end

    odata[end,end] = idata[end,end]
    return nothing
end

function matrix_free_prolongation_2d_GPU(idata,odata)
    (Nx,Ny) = size(idata)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx+TILE_DIM_1-1,TILE_DIM_1), div(Ny+TILE_DIM_2-1,TILE_DIM_2))
	blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim prolongation_2D_kernel(idata,odata,Nx,Ny,Val(TILE_DIM_1),Val(TILE_DIM_2))
    nothing
end

function restriction_2D_kernel(idata_tmp,odata,Nx,Ny,::Val{TILE_DIM_1},::Val{TILE_DIM_2}) where {TILE_DIM_1,TILE_DIM_2}
    tidx = threadIdx().x
    tidy = threadIdx().y
    i = (blockIdx().x - 1) * TILE_DIM_1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM_2 + tidy

    # idata_tmp = CuArray(zeros(Nx+2,Ny+2))
    # idata_tmp[2:end-1,2:end-1] .= idata

    size_odata = (div(Nx+1,2),div(Ny+1,2))

    if 1 <= i <= size_odata[1] && 1 <= j <= size_odata[2]
        odata[i,j] = (4*idata_tmp[2*i,2*j] + 
        2 * (idata_tmp[2*i,2*j-1] + idata_tmp[2*i,2*j+1] + idata_tmp[2*i-1,2*j] + idata_tmp[2*i+1,2*j]) +
         (idata_tmp[2*i-1,2*j-1] + idata_tmp[2*i-1,2*j+1] + idata_tmp[2*i+1,2*j-1]) + idata_tmp[2*i+1,2*j+1]) / 16
        # odata[i,j] = idata_tmp[2*i,2*j]
        # odata[i,j] = 1
    end
   
    return nothing
end

function matrix_free_restriction_2d_GPU(idata,odata)
    (Nx,Ny) = size(idata)
    idata_tmp = CuArray(zeros(Nx+2,Ny+2))
    copyto!(view(idata_tmp,2:Nx+1,2:Ny+1),idata)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx+TILE_DIM_1-1,TILE_DIM_1), div(Ny+TILE_DIM_2-1,TILE_DIM_2))
	blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim restriction_2D_kernel(idata_tmp,odata,Nx,Ny,Val(TILE_DIM_1),Val(TILE_DIM_2))
    nothing

end


function modified_richardson!(x,A,b;maxiter=3,ω=0.15)
    for _ in 1:maxiter
        x[:] .= x[:] + ω*(b .- A*x[:])
    end
end

function modified_richardson_GPU!(x_GPU,A_GPU,b_GPU;maxiter=3,ω=0.15)
    for _ in 1:maxiter
        x_GPU .= x_GPU .+ ω*(b_GPU .- A_GPU*x_GPU)
    end
end


function Two_level_multigrid(A,b,Nx,Ny,A_2h;nu=3,NUM_V_CYCLES=1,SBPp=2)
    v_values = Dict(1=>zeros(Nx*Ny))
    rhs_values = Dict(1 => b)
    N_values = Dict(1 => Nx)
    N_values[2] = div(Nx+1,2)

    x = zeros(length(b));
    v_values[1] = x
    
    for cycle_number in 1:NUM_V_CYCLES
        # jacobi_brittany!(v_values[1],A,b;maxiter=nu);
        modified_richardson!(v_values[1],A,b,maxiter=nu)
        r = b - A*v_values[1];
        f = restriction_2d(Nx) * r;
        v_values[2] = A_2h \ f

        # println("Pass first part")
        e_1 = prolongation_2d(N_values[2]) * v_values[2];
        v_values[1] = v_values[1] + e_1;
        # println("After coarse grid correction, norm(A*x-b): $(norm(A*v_values[1]-b))")
        # jacobi_brittany!(v_values[1],A,b;maxiter=nu);
        modified_richardson!(v_values[1],A,b,maxiter=nu)
    end
    return (v_values[1],norm(A * v_values[1] - b))
end


function Two_level_multigrid_GPU(A_GPU,b_GPU,Nx,Ny,A_2h;nu=3,NUM_V_CYCLES=1,SBPp=2)
    v_values = Dict(1=>CuArray(zeros(Nx*Ny)))
    rhs_values = Dict(1 => b_GPU)
    N_values = Dict(1 => Nx)
    N_values[2] = div(Nx+1,2)

    x = CuArray(zeros(length(b_GPU)));
    v_values[1] = x
    
    for cycle_number in 1:NUM_V_CYCLES
        # jacobi_brittany!(v_values[1],A,b;maxiter=nu);
        modified_richardson_GPU!(v_values[1],A_GPU,b_GPU,maxiter=nu)
        r = b_GPU - A_GPU*v_values[1];
        f = Array(restriction_2d_GPU(Nx) * r);
        v_values[2] = CuArray(A_2h \ f)

        # println("Pass first part")
        e_1 = prolongation_2d_GPU(N_values[2]) * v_values[2];
        v_values[1] = v_values[1] + e_1;
        # println("After coarse grid correction, norm(A*x-b): $(norm(A*v_values[1]-b))")
        modified_richardson_GPU!(v_values[1],A_GPU,b_GPU,maxiter=nu)
    end
    return (v_values[1],norm(A_GPU * v_values[1] - b_GPU))
end

function matrix_free_richardson(idata_GPU,odata_GPU,b_GPU;maxiter=3,ω=0.15)
    for _ in 1:maxiter
        matrix_free_A_full_GPU(idata_GPU,odata_GPU) # matrix_free_A_full_GPU is -A here, becareful
        odata_GPU .= idata_GPU .+ ω * (b_GPU .+ odata_GPU)
        idata_GPU .= odata_GPU
    end
end

function matrix_free_Two_level_multigrid(b_GPU,A_2h;nu=3,NUM_V_CYCLES=1,SBPp=2)
    (Nx,Ny) = size(b_GPU)
    level = Int(log(2,Nx-1))
    (Nx_2h,Ny_2h) = div.((Nx,Ny) .+ 1,2)
    # (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1,p=SBPp)
    v_values_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    v_values_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h))
    v_values_out_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    v_values_out_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h))
    Av_values_out_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    rhs_values_GPU = Dict(1=>b_GPU)
    N_values = Dict(1=>Nx)
    N_values[2] = div(Nx+1,2)
    f_GPU = Dict(1=>CuArray(zeros(Nx_2h,Ny_2h)))
    e_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    
    for cycle_number in 1:NUM_V_CYCLES
        matrix_free_richardson(v_values_GPU[1],v_values_out_GPU[1],rhs_values_GPU[1];maxiter=nu)
        matrix_free_A_full_GPU(v_values_out_GPU[1],Av_values_out_GPU[1])
        r_GPU = b_GPU + Av_values_out_GPU[1]
        matrix_free_restriction_2d_GPU(r_GPU,f_GPU[1])
        # v_values_GPU[2] = reshape(CuArray(A_2h) \ f_GPU[1][:],Nx_2h,Ny_2h)
        # v_values_GPU[2] = reshape(CUDA.CUSPARSE.CuSparseMatrixCSC(A_2h) \ f_GPU[1][:],Nx_2h,Ny_2h)
        v_values_GPU[2] = reshape(CuArray(A_2h \ Array(f_GPU[1][:])),Nx_2h,Ny_2h)

        # matrix_free_richardson(v_values_out_GPU[2],v_values_GPU[2],f_GPU[1];maxiter=20)

        matrix_free_prolongation_2d_GPU(v_values_GPU[2],e_GPU[1])
        v_values_GPU[1] .+= e_GPU[1]
        matrix_free_richardson(v_values_GPU[1],v_values_out_GPU[1],rhs_values_GPU[1];maxiter=nu)
    end
    matrix_free_A_full_GPU(v_values_out_GPU[1],Av_values_out_GPU[1])
    return (v_values_out_GPU[1],norm(-Av_values_out_GPU[1]-b_GPU))
end


function Three_level_multigrid(A,b,A_2h,b_2h,A_4h,b_4h,Nx,Ny;nu=3,NUM_V_CYCLES=1,SBPp=2)
    v_values = Dict(1=>zeros(Nx*Ny))
    Nx_2h = Ny_2h = div(Nx+1,2)
    Nx_4h = Ny_4h = div(Nx_2h+1,2)

    rhs_values = Dict(1 => b)
    N_values = Dict(1 => Nx)
    N_values[2] = Nx_2h
    N_values[3] = Nx_4h

    x = zeros(length(b));
    v_values[1] = x
    v_values[2] = zeros(Nx_2h*Ny_2h)
    
    for cycle_number in 1:NUM_V_CYCLES
        # jacobi_brittany!(v_values[1],A,b;maxiter=nu);
        modified_richardson!(v_values[1],A,b,maxiter=nu)
        r_h = b - A*v_values[1];

        rhs_values[2] = restriction_2d(Nx) * r_h;

        modified_richardson!(v_values[2],A_2h,rhs_values[2],maxiter=nu)
        # r_2h = b_2h - A_2h * v_values[2]
        r_2h = - A_2h * v_values[2]
        rhs_values[3] = restriction_2d(Nx_2h) * r_2h

        v_values[3] = A_4h \ rhs_values[3]

        # println("Pass first part")

        e_2 = prolongation_2d(N_values[3]) * v_values[3];
        v_values[2] = v_values[2] + e_2;
        # println("After coarse grid correction, norm(A*x-b): $(norm(A*v_values[1]-b))")
        # jacobi_brittany!(v_values[1],A,b;maxiter=nu);
        modified_richardson!(v_values[2],A_2h,rhs_values[2],maxiter=nu)
        e_1 = prolongation_2d(N_values[2]) * v_values[2]
        v_values[1] = v_values[1] + e_1
        modified_richardson!(v_values[1],A,b,maxiter=nu)
    end
    return (v_values[1],norm(A * v_values[1] - b))
end

function matrix_free_Three_level_multigrid(b_GPU,A_4h;nu=3,NUM_V_CYCLES=1,SBPp=2)
    (Nx,Ny) = size(b_GPU)
    level = Int(log(2,Nx-1))
    (Nx_2h,Ny_2h) = div.((Nx,Ny) .+ 1,2)
    (Nx_4h,Ny_4h) = div.((Nx_2h,Ny_2h) .+ 1,2)
    # (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1,p=SBPp)
    v_values_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    v_values_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h))

    v_values_out_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    v_values_out_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h))

    Av_values_out_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    Av_values_out_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h))
    rhs_values_GPU = Dict(1=>b_GPU)
    rhs_values_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h))
    rhs_values_GPU[3] = CuArray(zeros(Nx_4h,Ny_4h))

    N_values = Dict(1=>Nx)
    N_values[2] = Nx_2h
    N_values[3] = Nx_4h
    f_GPU = Dict(1=>CuArray(zeros(Nx_2h,Ny_2h)))
    f_GPU[2] = CuArray(zeros(Nx_4h,Ny_4h))

    e_GPU = Dict(1=>CuArray(zeros(Nx,Ny)));
    e_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h));
    
    for cycle_number in 1:NUM_V_CYCLES
        matrix_free_richardson(v_values_GPU[1],v_values_out_GPU[1],rhs_values_GPU[1];maxiter=nu)
        matrix_free_A_full_GPU(v_values_out_GPU[1],Av_values_out_GPU[1])
        r_h_GPU = b_GPU + Av_values_out_GPU[1]

        matrix_free_restriction_2d_GPU(r_h_GPU,rhs_values_GPU[2])

        matrix_free_richardson(v_values_GPU[2],v_values_out_GPU[2],rhs_values_GPU[2];maxiter=nu)
        matrix_free_A_full_GPU(v_values_out_GPU[2],Av_values_out_GPU[2])

        # r_2h_GPU = b_2h_GPU + Av_values_out_GPU[2]
        r_2h_GPU = Av_values_out_GPU[2]
        matrix_free_restriction_2d_GPU(r_2h_GPU,rhs_values_GPU[3])

        v_values_GPU[3] = reshape(CuArray(A_4h \ Array(rhs_values_GPU[3][:])),Nx_4h,Ny_4h)

        # matrix_free_richardson(v_values_out_GPU[2],v_values_GPU[2],f_GPU[1];maxiter=20)

        matrix_free_prolongation_2d_GPU(v_values_GPU[3],e_GPU[2])
        v_values_out_GPU[2] += e_GPU[2]
        v_values_GPU[2] .= v_values_out_GPU[2]
        matrix_free_richardson(v_values_GPU[2],v_values_out_GPU[2],rhs_values_GPU[2];maxiter=nu)
        matrix_free_prolongation_2d_GPU(v_values_out_GPU[2],e_GPU[1])
        v_values_out_GPU[1] += e_GPU[1]
        v_values_GPU[1] .= v_values_out_GPU[1]
        matrix_free_richardson(v_values_out_GPU[1],v_values_out_GPU[1],rhs_values_GPU[1];maxiter=nu)
        matrix_free_richardson(v_values_GPU[1],v_values_out_GPU[1],rhs_values_GPU[1];maxiter=nu)
    end
    matrix_free_A_full_GPU(v_values_out_GPU[1],Av_values_out_GPU[1])
    return (v_values_out_GPU[1],norm(-Av_values_out_GPU[1]-b_GPU))
end

function precond_matrix(A, b, A_2h; m=3, solver="jacobi") # For convergence analysis
    #pre and post smoothing 
    N = length(b)
    IN = sparse(Matrix(I, N, N))
    P = Diagonal(diag(A))
    Pinv = Diagonal(1 ./ diag(A))
    Q = P-A
    L = A - triu(A)
    U = A - tril(A)

    if solver == "jacobi"
       ω = 2/3
        H = ω*Pinv*Q + (1-ω)*IN 
        R = ω*Pinv 
        R0 = ω*Pinv 
    elseif solver == "ssor"
        ω = 1.4  #this is just a guess. Need to compute ω_optimal (from jacobi method)
        B1 = (P + ω*U)\Matrix(-ω*L + (1-ω)*P)
        B2 = (P + ω*L)\Matrix(-ω*U + (1-ω)*P) 
        H = B1*B2
        X = (P+ω*L)\Matrix(IN)
   
        R = ω*(2-ω)*(P+ω*U)\Matrix(P*X)
        R0 = ω*(2-ω)*(P+ω*U)\Matrix(P*X)
    elseif solver == "richardson"
        ω =ω_richardson
        H = IN - ω*A
        R = ω*IN
        # R = spzeros(N,N)
        R0 = ω*IN
    elseif solver == "richardson_chebyshev" #TODO: FIX ME FOR CHEB
        ω =ω_richardson
        H = IN - ω*A
        R = ω*IN
        R0 = ω*IN
    elseif solver == "chebyshev" #TODO: FIX ME FOR CHEB
        ω =ω_richardson
        H = IN - ω*A
        R = ω*IN
        R0 = ω*IN
    else   
    end

    for i = 1:m-1
        R += H^i * R0
    end

    # (A_2h, b_2h, x_2h, H1_2h) = get_operators(SBPp, 2*h);
    A_2h = A_2h
    I_r = standard_restriction_matrix_2D(N)
    
    I_p = standard_prolongation_matrix_2D(length(b_2h))
    M = H^m * (R + I_p * (A_2h\Matrix(I_r*(IN - A * R)))) + R
    M_alternative = R + R + I_p * (A_2h \ Matrix(I_r * (IN - A*R)))
    
    M_v0 = I_p * (A_2h \ Matrix(I_r)) # No richardson iteration
    M_v1_no_post = ω*IN + I_p * (A_2h \ Matrix(I_r * (IN - ω*A)))

    M_v1 = (IN - ω*A)*(ω*IN + I_p * (A_2h \ Matrix(I_r * (IN - ω*A)))) + ω*IN # one pre and post richardson iteration

    M_v1_alternative = H * (R0 + I_p * (A_2h \ Matrix(I_r * H))) + R0 # alternative form

    M_test = H^m * R + R + H^m * (I_p * (A_2h \ Matrix(I_r * H^m))) # need to change, a generalized representation of M for richardson iteration

    # return (M, R, H, I_p, A_2h, I_r, IN)
    return (M_test,R,H,I_p,A_2h,I_r,IN)
end


function mg_preconditioned_CG(A,b,x;A_2h = A_2h_lu,maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),NUM_V_CYCLES=1,nu=3,use_galerkin=true,direct_sol=0,H_tilde=0,SBPp=2)
    Nx = Ny = Int(sqrt(length(b)))
    level = Int(log(2,Nx-1))
    # (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1,p=SBPp);
    # A_2h = lu(A_2h)
    r = b - A * x;
    # (M, R, H, I_p, A_2h, I_r, IN) = precond_matrix(A,b;m=nu,solver="jacobi",p=p)
    z = Two_level_multigrid(A,r,Nx,Ny,A_2h;nu=nu,NUM_V_CYCLES=1)[1]
    # z = M*r
    p = z;
    num_iter_steps = 0
    norms = [norm(r)]
    errors = []
    if direct_sol != 0 && H_tilde != 0
        append!(errors,sqrt(direct_sol' * A * direct_sol))
    end

    rzold = r'*z

    for step = 1:maxiter
    # for step = 1:5
        num_iter_steps += 1
        alpha = rzold / (p'*A*p)
        x .= x .+ alpha * p;
        r .= r .- alpha * A*p
        rs = r' * r
        append!(norms,sqrt(rs))
        if direct_sol != 0 && H_tilde != 0
            error = sqrt((x - direct_sol)' * A * (x - direct_sol))
            # @show error
            append!(errors,error)
        end
        if sqrt(rs) < abstol
            break
        end
        z = Two_level_multigrid(A,r,Nx,Ny,A_2h;nu=nu,NUM_V_CYCLES=1)[1]
        # z = M*r
        rznew = r'*z
        beta = rznew/(rzold);
        p = z + beta * p;
        rzold = rznew
    end
    # @show num_iter_steps
    return x,num_iter_steps, norms, errors
end

function mg_preconditioned_CG_GPU(A_GPU,b_GPU,x_GPU;A_2h = A_2h_lu,maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),NUM_V_CYCLES=1,nu=3,use_galerkin=true,direct_sol=0,H_tilde=0,SBPp=2)
    Nx = Ny = Int(sqrt(length(b_GPU)))
    level = Int(log(2,Nx-1))
    # (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1,p=SBPp);
    # A_2h = lu(A_2h)
    r_GPU = b_GPU - A_GPU * x_GPU;
    # (M, R, H, I_p, A_2h, I_r, IN) = precond_matrix(A,b;m=nu,solver="jacobi",p=p)
    z_GPU = Two_level_multigrid_GPU(A_GPU,r_GPU,Nx,Ny,A_2h;nu=nu,NUM_V_CYCLES=1)[1]
    # z = M*r
    p_GPU = z_GPU;
    num_iter_steps = 0
    norms = [norm(r_GPU)]
    errors = []
    if direct_sol != 0 && H_tilde != 0
        append!(errors,sqrt(direct_sol' * A * direct_sol))
    end

    rzold = r_GPU'*z_GPU

    for step = 1:maxiter
    # for step = 1:5
        num_iter_steps += 1
        alpha = rzold / (p_GPU'*A_GPU*p_GPU)
        x_GPU .= x_GPU .+ alpha * p_GPU;
        r_GPU .= r_GPU .- alpha * (A_GPU*p_GPU)
        rs = r_GPU' * r_GPU
        append!(norms,sqrt(rs))
        # if direct_sol != 0 && H_tilde != 0
        #     error = sqrt((x_GPU - direct_sol)' * A_GPU * (x_GPU - direct_sol))
        #     # @show error
        #     append!(errors,error)
        # end
        if sqrt(rs) < abstol
            break
        end
        z_GPU = Two_level_multigrid_GPU(A_GPU,r_GPU,Nx,Ny,A_2h;nu=nu,NUM_V_CYCLES=1)[1]
        # z = M*r
        rznew = r_GPU'*z_GPU
        beta = rznew/(rzold);
        p_GPU .= z_GPU .+ beta * p_GPU;
        rzold = rznew
    end
    # @show num_iter_steps
    return x_GPU,num_iter_steps, norms, errors
end

function test_preconditioned_CG(;level=level,nu=3,ω=2/3,SBPp=2)
    (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level,p=SBPp);
    (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1,p=SBPp);
    A_2h_lu = lu(A_2h)
    direct_sol = A\b
    reltol = sqrt(eps(real(eltype(b))))
    x = zeros(Nx*Ny);
    abstol = norm(A*x-b) * reltol
    h = 1/(Nx-1)
    ω_richardson = 0.15
    p = SBPp
    (M, R, H, I_p, A_2h, I_r, IN) = precond_matrix(A,b,A_2h;m=nu,solver="richardson")

    cond_A_M = cond(M*A)
    x = zeros(Nx*Ny);
    x_mgcg_GPU,iter_mg_cg, norm_mg_cg, error_mg_cg = mg_preconditioned_CG(A,b,x;A_2h = A_2h_lu, maxiter=length(b),abstol=abstol,NUM_V_CYCLES=1,nu=nu,use_galerkin=true,direct_sol=direct_sol,H_tilde=H_tilde,SBPp=SBPp)
    error_mg_cg_bound_coef = (sqrt(cond_A_M) - 1) / (sqrt(cond_A_M) + 1)
    error_mg_cg_bound = error_mg_cg[1] .* 2 .* error_mg_cg_bound_coef .^ (0:1:length(error_mg_cg)-1)
    scatter(log.(10,error_mg_cg),label="error_mg_cg", markercolor = "darkblue")
    plot!(log.(10,error_mg_cg_bound),label="error_mg_cg_bound",linecolor = "darkblue")


    cond_A = cond(Matrix(A))
    x0 = zeros(Nx*Ny)
    (E_cg, num_iter_steps_cg, norms_cg) = regularCG!(A,b,x0,H_tilde,direct_sol;maxiter=20000,abstol=abstol)

    scatter!(log.(10,E_cg),label="error_cg", markercolor = "darksalmon")
    error_cg_bound_coef = (sqrt(cond_A) - 1) / (sqrt(cond_A) + 1)
    error_cg_bound = E_cg[1] .* 2 .* error_cg_bound_coef .^ (0:1:length(E_cg)-1)
    plot!(log.(10,error_cg_bound),label="error_cg_bound", linecolor = "darksalmon")


    savefig("convergence_richardson.png")
    # my_solver = "jacobi"
    # x0 = zeros(Nx*Ny)
    # (E_mgcg, num_iter_steps_mgcg, norms_mgcg) = MGCG!(A,b,x0,H_tilde,direct_sol,my_solver;smooth_steps = nu,maxiter=20000,abstol=abstol)


    # plot(error_mg_cg,label="error_mg_cg")
    # plot!(error_mg_cg_bound,label="error_mg_cg_bound")
end