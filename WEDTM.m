function [ParaGlobal,ParaLocal,Accuracy_all] = WEDTM(X_all, F, K, T, S, Para)

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


% beta for t > 1
beta0 = 0.05;


if strcmp(Para.evaluation,'dc')

    X = X_all(:,Para.train_idx);
else
    X = X_all;
end


[V,N] = size(X);

[Xtrain,Xtest,WS,DS,WordTrainS,DocTrainS]= PartitionX_v1(X,Para.train_word_prop);

WS = WS(WordTrainS); 
DS = DS(WordTrainS); 


Yflagtrain = Xtrain>0;
Yflagtest = Xtest>0;
loglikeTrain = []; loglike=[];
ave.loglike=[];
ave.K=[];  
ave.gamma0=[];
ave.PhiTheta = 0;
ave.PhiThetaSum = 0;
ave.Count = 0;
Xmask=sparse(X);


Phi = cell(T,1);    

% Add the default word embedding
F = [F, ones(V,1)];

Theta = cell(T,1); 
c_j = cell(T+1,1);
for t=1:T+1
    c_j{t}=ones(1,N);
end
Xt_to_t1 = cell(T,1);   WSZS = cell(T,1);

ParaGlobal = cell(T,1);      
ParaLocal = cell(T,1);  
Accuracy_all = cell(T,1);

% Initialise beta for t = 1 
[beta1, beta_para] = init_beta(K(1), V, S, F, beta0);



for Tcurrent = 1:T
    if Tcurrent == 1
        ZS = randi(K(Tcurrent),length(DS),1);
        ZSDS = full(sparse(ZS,DS,1,K(Tcurrent),N));
        ZSWS = full(sparse(ZS,WS,1,K(Tcurrent),V));
        WSZS{Tcurrent}=ZSWS';
        Xt_to_t1{Tcurrent}=ZSDS;
        n_dot_k = sum(ZSDS,2);            
        p_j = Calculate_pj(c_j,Tcurrent);
        r_k = 1/K(Tcurrent)*ones(K(Tcurrent),1);
        gamma0 = 1;       c0 = 1;
    else
        K(Tcurrent) = K(Tcurrent-1); 
        if K(Tcurrent)  <= 4
            break;
        end
        Phi{Tcurrent} = rand(K(Tcurrent-1),K(Tcurrent));            
        Phi{Tcurrent} = bsxfun(@rdivide, Phi{Tcurrent}, max(realmin,sum(Phi{Tcurrent},1)));
        Theta{Tcurrent} = ones(K(Tcurrent),N)/K(Tcurrent);
        p_j = Calculate_pj(c_j,Tcurrent);
        r_k = 1/K(Tcurrent)*ones(K(Tcurrent),1);
        gamma0 = K(Tcurrent)/K(1);       c0 = 1;   
    end
    
    for iter = 1 : (Para.TrainBurnin(Tcurrent) + Para.TrainCollection(Tcurrent) )
        tic
        if iter == Para.TrainBurnin(Tcurrent)
            TrimTcurrent_WEDTM; 
        end
  
        for t = 1:Tcurrent
            if t == 1 
                dex111=randperm(length(ZS)); 
                ZS=ZS(dex111); DS=DS(dex111); WS=WS(dex111); 
                if Tcurrent==1
                    shape = r_k*ones(1,N);
                else
                    shape = Phi{2}*Theta{2};
                end

                beta1_sum = sum(beta1,2);
                % Modified from GNBP_mex_collapsed_deep.c in the GBN code,
                % to support a full matrix of beta1
                [ZSDS,ZSWS,n_dot_k,ZS] = GNBP_mex_collapsed_deep_WEDTM(ZSDS,ZSWS,n_dot_k,ZS,WS,DS,shape,beta1,beta1_sum);

                WSZS{t}=ZSWS';
                Xt_to_t1{t}=ZSDS;
                % Sample the variables related to sub-topics
                [beta1, beta_para] = sample_beta(WSZS{t}',F, beta1, beta_para);

                
            else
 
                [Xt_to_t1{t},WSZS{t}] = CRT_Multrnd_Matrix(sparse(Xt_to_t1{t-1}),Phi{t},Theta{t});
                
            end
            
            
            if t > 1 
                Phi{t} = SamplePhi(WSZS{t},beta0);            
                if nnz(isnan(Phi{t}))
                    warning('Phi Nan');
                    Phi{t}(isnan(Phi{t})) = 0;
                end
            end
        end

        
        Xt = CRT_sum_mex_matrix_v1(sparse(Xt_to_t1{Tcurrent}'),r_k')';
        [r_k,gamma0,c0]=Sample_rk(full(Xt),r_k,p_j{Tcurrent+1},gamma0,c0);
                
        if iter>10
            if Tcurrent > 1
                p_j{2} = betarnd(  sum(Xt_to_t1{1},1)+a0   ,   sum(Theta{2},1)+b0  );
            else
                p_j{2} = betarnd(  sum(Xt_to_t1{1},1)+a0   ,   sum(r_k,1)+b0  );
            end
            p_j{2} = min( max(p_j{2},eps) , 1-eps);
            c_j{2} = (1-p_j{2})./p_j{2};
            for t = 3:(Tcurrent+1)
                if t == Tcurrent+1
                    c_j{t} = randg(sum(r_k)*ones(1,N)+e0) ./ (sum(Theta{t-1},1)+f0);
                else
                    c_j{t} = randg(sum(Theta{t},1)+e0) ./ (sum(Theta{t-1},1)+f0);
                end
            end
            p_j_temp = Calculate_pj(c_j,Tcurrent);
            p_j(3:end)=p_j_temp(3:end);
        end
        
        for t = Tcurrent:-1:1
            if t == Tcurrent
                shape = r_k;
            else
                shape = Phi{t+1}*Theta{t+1};
            end
            if t > 1 

                Theta{t} = bsxfun(@times, randg(bsxfun(@plus,shape,Xt_to_t1{t})),  1 ./ (c_j{t+1}-log(max(1-p_j{t},realmin))) );
                        
                if nnz(isnan(Theta{t}))
                    warning('Theta Nan');
                    Theta{t}(isnan(Theta{t}))=0;
                end
            end
        end
        Timetmp = toc;
        
        if mod(iter,Para.CollectionStep)==0 && Para.train_word_prop<1
            
            Phi{1} = SamplePhi(WSZS{1},beta1');
            if Tcurrent==1
                shape = r_k*ones(1,N);
            else
                shape = Phi{2}*Theta{2};
            end
            Theta{1} = bsxfun(@times, randg(bsxfun(@plus,shape,Xt_to_t1{1})),  p_j{2});
            
            X1 = Mult_Sparse(Xmask,Phi{1},Theta{1});
            X1sum = sum(Theta{1},1);
            
            X2 = bsxfun(@rdivide, X1,X1sum);
            loglike(end+1)=sum(Xtest(Yflagtest).*log(X2(Yflagtest)))/sum(Xtest(:));
            loglikeTrain(end+1)=sum(Xtrain(Yflagtrain).*log(X2(Yflagtrain)))/sum(Xtrain(:));
            
            if iter>Para.TrainBurnin(Tcurrent)
                ave.PhiTheta = ave.PhiTheta + X1;
                ave.PhiThetaSum = ave.PhiThetaSum + X1sum;
                ave.Count = ave.Count+1;
                X1 = ave.PhiTheta/ave.Count;
                X1sum = ave.PhiThetaSum/ave.Count;
                X1= bsxfun(@rdivide, X1,X1sum);
                ave.loglike(end+1) = sum(Xtest(Yflagtest).*log(X1(Yflagtest)))/sum(Xtest(:));
            else
                ave.loglike(end+1)  = NaN;
            end
            
            
                        
            clear X1 X2;
        end
        
        
        
        if mod(iter,10)==0
            fprintf('JointTrain Layer: %d, iter: %d, K: %d, TimePerIter: %d seconds. \n',Tcurrent,iter,nnz(sum(Xt,2)),Timetmp);
            if Para.train_word_prop<1 && strcmp(Para.evaluation,'perplexity')
                fprintf('train: %0.2f, test_avg: %0.2f \n',exp(-loglikeTrain(end)),exp(-ave.loglike(end)));
            end
        end

    end
    

    for t = 1:Tcurrent
        if t == 1
            Phi{t} = SamplePhi(WSZS{t},beta1',true);
        else
            Phi{t} = SamplePhi(WSZS{t},beta0,true);
        end
    end
        
    ParaGlobal{Tcurrent}.Phi = Phi;
    ParaGlobal{Tcurrent}.r_k = r_k;
    ParaGlobal{Tcurrent}.gamma0 = gamma0;
    ParaGlobal{Tcurrent}.c0 = c0;
    ParaGlobal{Tcurrent}.K = K(1:Tcurrent);
    ParaGlobal{Tcurrent}.beta0 = beta0;
    ParaGlobal{Tcurrent}.beta_para = beta_para;
    ParaGlobal{Tcurrent}.ave = ave;
    
    % for theta
    ParaGlobal{Tcurrent}.p_j = p_j;
    ParaGlobal{Tcurrent}.c_j = c_j;
    ParaGlobal{Tcurrent}.Xt_to_t1 = Xt_to_t1;
    
    
    

    
    for t = 1:Tcurrent+1
        ParaGlobal{Tcurrent}.cjmedian{t} = median(c_j{t});
    end
    
    if strcmp(Para.evaluation,'dc')
        Para.DataType = 'Count';
        Para.ParallelProcessing = false;
        c_jmean = zeros(1,Tcurrent+1);
        for t = 1:(Tcurrent+1)
            c_jmean(t) = median(c_j{t});
        end
        [Accuracy_all{Tcurrent},ParaLocal{Tcurrent}] = GBN_Testing(X_all,ParaGlobal{Tcurrent},Tcurrent,Para,c_jmean);
    end
    

    
end 




