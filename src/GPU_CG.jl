include("diagonal_sbp.jl")
include("split_matrix_free.jl")



function CG_CPU(A,b,x;maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),direct_sol=0,H_tilde=0)
    r = b - A * x;
    p = copy(r);
    rsold = dot(r,r)
    # Ap = spzeros(length(b))
    Ap = similar(b);

    num_iter_steps = 0
    norms = [sqrt(rsold)]
    errors = []
    if direct_sol != 0 && H_tilde != 0
        append!(errors,sqrt(direct_sol' * A * direct_sol))
    end
    # @show rsold
    for step = 1:maxiter
        num_iter_steps += 1
        mul!(Ap,A,p);
        alpha = rsold / dot(p,Ap)
        r .= r .- alpha .* Ap;
        x .= x .+ alpha .* p;
        rsnew = r' * r
        append!(norms,sqrt(rsnew))
        if direct_sol != 0 && H_tilde != 0
            error = sqrt((x - direct_sol)' * A * (x - direct_sol))
            append!(errors,error)
        end
        if rsnew < abstol^2
              break
        end
        p .= r .+ (rsnew / rsold) * p;
        rsold = rsnew
    end
    # return (num_iter_steps,norms,errors)
    return num_iter_steps, norms
end


function CG_GPU_sparse(x_GPU_sparse,A_GPU_sparse,b_GPU_sparse;abstol=reltol)
    Ap_GPU_sparse = A_GPU_sparse * x_GPU_sparse
    r_GPU_sparse = b_GPU_sparse - Ap_GPU_sparse
    p_GPU_sparse = copy(r_GPU_sparse)
    rsold_GPU = dot(r_GPU_sparse, r_GPU_sparse)
    num_iter_steps = 0
    norms = [sqrt(rsold_GPU)]
    for i in 1:length(x_GPU_sparse)
        num_iter_steps += 1
        # Ap_GPU_sparse .= A_GPU_sparse * p_GPU_sparse
        mul!(Ap_GPU_sparse,A_GPU_sparse,p_GPU_sparse)
        # alpha_GPU = rsold_GPU / (sum(p_GPU_sparse .* Ap_GPU_sparse))
        alpha_GPU = rsold_GPU / dot(p_GPU_sparse,Ap_GPU_sparse)
        r_GPU_sparse .-= alpha_GPU .* Ap_GPU_sparse
        x_GPU_sparse .+= alpha_GPU .* p_GPU_sparse
        # rsnew_GPU = sum(r_GPU_sparse .* r_GPU_sparse)
        rsnew_GPU = dot(r_GPU_sparse,r_GPU_sparse)
        append!(norms,sqrt(rsnew_GPU))
        if rsnew_GPU < abstol^2
            break
        end
        p_GPU_sparse .= r_GPU_sparse .+ (rsnew_GPU / rsold_GPU) .* p_GPU_sparse
        rsold_GPU = rsnew_GPU
    end
    return num_iter_steps,norms
end



function CG_full_GPU(b_reshaped_GPU,x_GPU;abstol=sqrt(eps(real(eltype(b_reshaped_GPU)))))
    (Nx,Ny) = size(b_reshaped_GPU);
    odata = CuArray(zeros(Nx,Ny))
    # matrix_free_A_full_GPU(x_GPU,odata);
    matrix_free_A_full_GPU(x_GPU,odata)
    r_GPU = b_reshaped_GPU - odata;
    p_GPU = copy(r_GPU);
    # rsold_GPU = sum(r_GPU .* r_GPU)
    rsold_GPU = dot(r_GPU,r_GPU)
    # Ap_GPU = CUDA.zeros(Nx,Ny);
    Ap_GPU = CuArray(zeros(Nx,Ny))
    num_iter_steps = 0
    machine_eps = sqrt(eps(real(eltype(b_reshaped_GPU))))
    # rel_tol = machine_eps * max(sqrt(rsold_GPU),1)
    rel_tol = machine_eps
    norms = [sqrt(rsold_GPU)]
    for i in 1:Nx*Ny
    # for i in 1:20
        num_iter_steps += 1
        matrix_free_A_full_GPU(p_GPU,Ap_GPU);
        alpha_GPU = rsold_GPU / (dot(p_GPU,Ap_GPU))
        # x_GPU = x_GPU + alpha_GPU * p_GPU;
        # r_GPU = r_GPU - alpha_GPU * Ap_GPU;
        r_GPU .-= alpha_GPU .* Ap_GPU
        x_GPU .+= alpha_GPU .* p_GPU
        # CUDA.CUBLAS.axpy!()
        # rsnew_GPU = sum(r_GPU .* r_GPU)
        rsnew_GPU = dot(r_GPU,r_GPU)
        if sqrt(rsnew_GPU) < abstol
            break
        end
        p_GPU .= r_GPU .+ (rsnew_GPU/rsold_GPU) .* p_GPU;
        rsold_GPU = rsnew_GPU
        append!(norms,sqrt(rsold_GPU))
        # if i < 20
        #     @show rsold_GPU
        # end
    end
    # @show num_iter_steps
    # (num_iter_steps,abstol,norms[end])
    # return x_GPU, num_iter_steps
    return num_iter_steps, norms
end

function CG_Matrix_Free_GPU_v2(x_GPU,Ap_GPU,b_reshaped_GPU,Nx,Ny;abstol=reltol,maxiter=Nx*Ny) # optimized performance
    matrix_free_A_full_GPU(x_GPU,Ap_GPU)
    r_GPU = b_reshaped_GPU - Ap_GPU
    p_GPU = copy(r_GPU)
    # rsold_GPU = sum(r_GPU .* r_GPU)
    rsold_GPU = dot(r_GPU,r_GPU)
    num_iter_steps = 0
    # for i in 1:Nx*Ny
    norms = [sqrt(rsold_GPU)]
    for i in 1:maxiter
        num_iter_steps += 1
        matrix_free_A_full_GPU(p_GPU,Ap_GPU)
        # alpha_GPU = rsold_GPU / (sum(p_GPU .* Ap_GPU))
        alpha_GPU = rsold_GPU / dot(p_GPU,Ap_GPU)
        r_GPU .-= alpha_GPU .* Ap_GPU
        x_GPU .+= alpha_GPU .* p_GPU
        # rsnew_GPU = sum(r_GPU .* r_GPU)
        rsnew_GPU = dot(r_GPU,r_GPU)
        if rsnew_GPU < abstol^2
            break
        end
        p_GPU .= r_GPU .+ (rsnew_GPU/rsold_GPU) .* p_GPU
        rsold_GPU = rsnew_GPU
        append!(norms,sqrt(rsold_GPU))
    end
    return num_iter_steps, norms
end

