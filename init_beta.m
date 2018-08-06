function [beta1, beta_para] = init_beta(K,V,S, F, init_beta)

%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou, 
% ?Inter and Intra Topic Structure Learning with Word Embeddings,? 
% in International Conference on Machine Learning (ICML) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************

% Word embedding dimensions
L = size(F,2);

beta_para = cell(S,1);

for s = 1:S
    % variables for sub-topic s
    beta_para{s}.beta_s = init_beta/S .* ones(K,V);
    beta_para{s}.alpha_k = 0.1 * ones(K,1);
    beta_para{s}.W = 0.1 * ones(K,L);
    beta_para{s}.pi = beta_para{s}.W*F';
    beta_para{s}.sigma = 1 .* ones(K,L);      
    beta_para{s}.c0 = 1;
    beta_para{s}.alpha0 = 1;
end

beta1 = init_beta .* ones(K,V);

end