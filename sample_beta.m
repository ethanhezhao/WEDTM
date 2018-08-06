function [beta1, beta_para] = sample_beta(n_topic_word, F, beta1, beta_para)

%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou, 
% ?Inter and Intra Topic Structure Learning with Word Embeddings,? 
% in International Conference on Machine Learning (ICML) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************

a0 = 0.01;
b0 = 0.01;
e0 = 1;
f0 = 1;

S = length(beta_para);
L = size(F,2);

% The word count for each v and k in the first layer
[K,V] = size(n_topic_word);
n_sum = sum(n_topic_word,2);

%% Eq. (3)
log_inv_q = -log(betarnd(sum(beta1,2),n_sum));
log_log_inv_q = log(log_inv_q);

% Active topics in the first layer
active_k = ~isnan(log_inv_q) & ~isinf(log_inv_q) & n_sum >0 & log_inv_q ~=0;

%% Eq. (4) and (6)
h = zeros(K,V,S);
for k = 1:K
    for v = 1:V
        for j=1:n_topic_word(k,v)
            if j == 1
                is_add_table = 1;
            else
                is_add_table = double(rand() < beta1(k,v) ./ (beta1(k,v) + j));
            end
            if is_add_table > 0
                p = zeros(S,1);
                for s = 1:S
                    p(s) = beta_para{s}.beta_s(k,v);
                end
                sum_cum = cumsum(p);
                ss = find(sum_cum > rand() * sum_cum(end),1);
                h(k,v,ss) = h(k,v,ss) + 1;
            end
        end
    end
end

beta1 = 0;

for s = 1:S
    %% For each sub-topic s
    alpha_k = beta_para{s}.alpha_k;
    pi_pg = beta_para{s}.pi;
    W = beta_para{s}.W;
    c0 = beta_para{s}.c0;
    alpha0 = beta_para{s}.alpha0;
    h_s = h(:,:,s);
    
    %% Sample alpha_k for each sub-topic s with the hierarchical gamma
    h_st = zeros(K,V);
    % Eq. (11)
    h_st(h_s>0) = 1;
    for k = 1:K
        for v = 1:V
            for j=1:h_s(k,v)-1
                h_st(k,v) = h_st(k,v) + double(rand() < alpha_k(k) ./ (alpha_k(k) + j));
            end
        end
    end
    % Eq. (10)
    h_st_dot = sum(h_st,2);
    % Active topics in each sub-topic s
    local_active_k = h_st_dot > 0 & active_k;
    l_a_K = sum(local_active_k);
    temp = sum(logOnePlusExp(pi_pg + log_log_inv_q),2);
    % Eq. (9)
    alpha_k = randg(alpha0/l_a_K + h_st_dot) ./ (c0 + temp);
    h_stt = zeros(K,1);
    h_stt(h_st_dot > 0) = 1;
    for k = 1:K
        for j=1:h_st_dot(k)-1
            h_stt(k) = h_stt(k) + double(rand() < (alpha0/l_a_K) ./ (alpha0/l_a_K + j));
        end
    end
    temp2 = temp ./ (c0 + temp);
    % L17 in Figure 1 in the appendix
    alpha0 = randg(a0 + sum(h_stt)) ./ (b0 - sum(log(1-temp2(local_active_k)))/l_a_K);
    c0 = randg(e0 + alpha0) ./ (f0 + sum(alpha_k(local_active_k)));
    
    %% Sample Polya-Gamma variables
    % Eq. (15)
    pi_pg_vec = reshape(pi_pg + log_log_inv_q,K*V,1);
    omega_vec = PolyaGamRnd_Gam(reshape(h_s + alpha_k, K*V,1),pi_pg_vec,2);
    omega_mat = reshape(omega_vec,K,V);
    
    %% Sample sigma
    sigma_w = randg(1e-2 + 0.5 * l_a_K)./(1e-2 + sum(W(local_active_k,:).^2,1) * 0.5);
    sigma_w = repmat(sigma_w, K, 1);
    
    %% Sample W
    % Eq. (14)
    for k = 1:K
        if local_active_k(k) > 0
            Hgam = bsxfun(@times,F',omega_mat(k,:));
            invSigmaW = diag(sigma_w(k,:)) + Hgam*F;
            MuW = invSigmaW\(sum(bsxfun(@times,F',0.5 * h_s(k,:)-0.5 * alpha_k(k,:)-(log_log_inv_q(k))*omega_mat(k,:)),2));
            R = choll(invSigmaW);
            W(k,:) = (MuW + R\randn(L,1))';
        else
            W(k,:) = 1e-10;
        end
    end
    W(logical(sum(isnan(W) | isinf(W),2)),:) = 1e-10;
    
    % Update pi, Eq. (8)
    pi_pg = W * F';
    
    %% Sample beta for each sub-topic s
    % Eq. (7)
    beta_s = randg(alpha_k + h_s) ./ (exp(-pi_pg) + log_inv_q);
    beta_s(~local_active_k,:) = 0.05/S;
    beta_s(logical(sum(isnan(beta_s),2)),:) = 0.05/S;
    beta_s(logical(sum(isnan(beta_s)|isinf(beta_s),2)),:) = 0.05/S;
    beta_s(~logical(sum(beta_s,2)),:) = 0.05/S;
    
    
    %% Update beta1
    beta1 = beta1 + beta_s;
    
    %% Collect results
    beta_para{s}.beta_s = beta_s;
    beta_para{s}.pi = pi_pg;
    beta_para{s}.W = W;
    beta_para{s}.alpha_k = alpha_k;
    beta_para{s}.sigma = sigma_w;
    beta_para{s}.h_s = sparse(h_s);
    beta_para{s}.c0 = c0;
    beta_para{s}.alpha0 = alpha0;
    
end



end