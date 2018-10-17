function Theta = point_estimate_theta(saved_para)

T = length(find(~cellfun(@isempty,saved_para.Phi)));

Theta = cell(T,1);

for t = T:-1:1
    if t == T
        shape = saved_para.r_k;
    else
        shape = saved_para.Phi{t+1}*Theta{t+1};
    end
    if t > 1 

        Theta{t} = bsxfun(@times, randg(bsxfun(@plus,shape,saved_para.Xt_to_t1{t})),  1 ./ (saved_para.c_j{t+1}-log(max(1-saved_para.p_j{t},realmin))) );

        if nnz(isnan(Theta{t}))
            warning('Theta Nan');
            Theta{t}(isnan(Theta{t}))=0;
        end
    else
        Theta{1} = bsxfun(@times, randg(bsxfun(@plus,shape,saved_para.Xt_to_t1{1})),  saved_para.p_j{2});
    end
end