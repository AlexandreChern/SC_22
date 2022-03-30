include("../src/diagonal_sbp.jl")
include("two_level_mg.jl")
include("../src/split_matrix_free.jl")

function matrix_free_MGCG(b_GPU,x_GPU;A_2h = A_2h_lu,maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),NUM_V_CYCLES=1,nu=3,use_galerkin=true,direct_sol=0,H_tilde=0,SBPp=2)
    (Nx,Ny) = size(b_GPU)
    level = Int(log(2,Nx-1))
    # (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1,p=SBPp)
    # A_2h = lu(A_2h)
    Ax_GPU = CuArray(zeros(size(x_GPU)))
    matrix_free_A_full_GPU(x_GPU,Ax_GPU)
    r_GPU = b_GPU + Ax_GPU
    z_GPU = matrix_free_Two_level_multigrid(r_GPU,A_2h;nu=nu)[1]
    p_GPU = copy(z_GPU)
    Ap_GPU = copy(p_GPU)
    num_iter_steps_GPU = 0
    norms_GPU = [norm(r_GPU)]
    errors_GPU = []
    if direct_sol != 0 && H_tilde != 0
        append!(errors,sqrt(direct_sol' * A * direct_sol)) # need to rewrite
    end

    rzold_GPU = sum(r_GPU .* z_GPU)

    for step = 1:maxiter
        num_iter_steps_GPU += 1
        matrix_free_A_full_GPU(p_GPU,Ap_GPU)
        alpha_GPU = - rzold_GPU / sum(p_GPU .* Ap_GPU)
        x_GPU .+= alpha_GPU .* p_GPU
        r_GPU .+= alpha_GPU .* Ap_GPU
        rs_GPU = sum(r_GPU .* r_GPU)
        append!(norms_GPU,sqrt(rs_GPU))
        if direct_sol != 0 && H_tilde != 0
            error = sqrt((x - direct_sol)' * A * (x - direct_sol)) # need to rewrite
            # @show error
            append!(errors,error)
        end
        if sqrt(rs_GPU) < abstol
            break
        end
        z_GPU .=  matrix_free_Two_level_multigrid(r_GPU,A_2h;nu=nu)[1]
        rznew_GPU = sum(r_GPU .* z_GPU)
        beta_GPU = rznew_GPU / rzold_GPU
        p_GPU .= z_GPU .+ beta_GPU .* p_GPU
        rzold_GPU = rznew_GPU
    end
    return num_iter_steps_GPU, norms_GPU, x_GPU
end

function matrix_free_MGCG_Three_level(b_GPU,x_GPU;A_4h = A_4h_lu,maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),NUM_V_CYCLES=1,nu=3,use_galerkin=true,direct_sol=0,H_tilde=0,SBPp=2)
    (Nx,Ny) = size(b_GPU)
    level = Int(log(2,Nx-1))
    # (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1,p=SBPp)
    # A_2h = lu(A_2h)
    Ax_GPU = CuArray(zeros(size(x_GPU)))
    matrix_free_A_full_GPU(x_GPU,Ax_GPU)
    r_GPU = b_GPU + Ax_GPU
    z_GPU = matrix_free_Three_level_multigrid(r_GPU,A_4h;nu=nu)[1]
    p_GPU = copy(z_GPU)
    Ap_GPU = copy(p_GPU)
    num_iter_steps_GPU = 0
    norms_GPU = [norm(r_GPU)]
    errors_GPU = []
    if direct_sol != 0 && H_tilde != 0
        append!(errors,sqrt(direct_sol' * A * direct_sol)) # need to rewrite
    end

    rzold_GPU = sum(r_GPU .* z_GPU)

    for step = 1:maxiter
        num_iter_steps_GPU += 1
        matrix_free_A_full_GPU(p_GPU,Ap_GPU)
        alpha_GPU = - rzold_GPU / sum(p_GPU .* Ap_GPU)
        x_GPU .+= alpha_GPU .* p_GPU
        r_GPU .+= alpha_GPU .* Ap_GPU
        rs_GPU = sum(r_GPU .* r_GPU)
        append!(norms_GPU,sqrt(rs_GPU))
        if direct_sol != 0 && H_tilde != 0
            error = sqrt((x - direct_sol)' * A * (x - direct_sol)) # need to rewrite
            # @show error
            append!(errors,error)
        end
        if sqrt(rs_GPU) < abstol
            break
        end
        z_GPU .=  matrix_free_Three_level_multigrid(r_GPU,A_4h;nu=nu)[1]
        rznew_GPU = sum(r_GPU .* z_GPU)
        beta_GPU = rznew_GPU / rzold_GPU
        p_GPU .= z_GPU .+ beta_GPU .* p_GPU
        rzold_GPU = rznew_GPU
    end
    return num_iter_steps_GPU, norms_GPU
end