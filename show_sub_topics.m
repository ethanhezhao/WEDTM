
function show_sub_topics(ParaGlobal, voc)

%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou, 
% ?Inter and Intra Topic Structure Learning with Word Embeddings,? 
% in International Conference on Machine Learning (ICML) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************

T = length(ParaGlobal);
beta_para = ParaGlobal{T}.beta_para;
S = length(beta_para);
K = size(beta_para{1}.beta_s,1);

% The number of words you want to show
n_top_w = 10;


sub_weights = zeros(K,S);
for k = 1:K
    for s = 1:S
        sub_weights(k,s) = sum(beta_para{s}.beta_s(k,:));
    end
end

normal_weights = sum(sub_weights,2);
[~,topic_idx_normal] = sort(normal_weights, 'descend');
sub_weights = sub_weights ./ sum(sub_weights,2);

[~, tw_idx_normal] = sort(ParaGlobal{T}.Phi{1}',2, 'descend');

tw_idx_sub = cell(S,1);
for s = 1:S
    [~, tw_idx_sub{s}] = sort(beta_para{s}.beta_s,2, 'descend');
end

for k = 1:K    
    sorted_k = topic_idx_normal(k);
    [~,sub_idx] = sort(sub_weights(sorted_k,:),'descend');
    
    top_words = [];
    for v = 1:n_top_w
        top_words = [top_words, ' ', voc{tw_idx_normal(sorted_k,v)}];
    end
    
    fprintf('Normal topic %d: %s\n', k, top_words);
  
    for s = 1:S
        top_words = [];
        for v = 1:n_top_w
            top_words = [top_words, ' ', voc{tw_idx_sub{sub_idx(s)}(sorted_k,v)}];
        end
        
        fprintf('Sub-topic %d, %f: %s\n',sub_idx(s), sub_weights(sorted_k,sub_idx(s)),top_words);
        
    end
    
    fprintf('--------------------\n');
    
end


