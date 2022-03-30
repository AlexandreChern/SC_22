include("diagonal_sbp.jl")
include("split_matrix_free.jl")



function CG_CPU(A,b,x;maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),direct_sol=0,H_tilde=0)
    r = b - A * x;
    p = r;
    rsold = r' * r
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
        alpha = rsold / (p' * Ap)
        x .= x .+ alpha * p;
        r .= r .- alpha * Ap;
        rsnew = r' * r
        append!(norms,sqrt(rsnew))
        if direct_sol != 0 && H_tilde != 0
            error = sqrt((x - direct_sol)' * A * (x - direct_sol))
            append!(errors,error)
        end
        if rsnew < abstol^2
              break
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew
    end
    return (num_iter_steps,norms,errors)
end


function CG_GPU_sparse(x_GPU_sparse,A_GPU_sparse,b_GPU_sparse;abstol=reltol)
    Ap_GPU_sparse = A_GPU_sparse * x_GPU_sparse
    r_GPU_sparse = b_GPU_sparse - Ap_GPU_sparse
    p_GPU_sparse = copy(r_GPU_sparse)
    rsold_GPU = sum(r_GPU_sparse .* r_GPU_sparse)
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
        if rsnew_GPU < abstol^2
            break
        end
        p_GPU_sparse .= r_GPU_sparse .+ (rsnew_GPU / rsold_GPU) .* p_GPU_sparse
        rsold_GPU = rsnew_GPU
        push!(norms,sqrt(rsnew_GPU))
    end
    return num_iter_steps,abstol,norms[end]
end



