
%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou, 
% ?Inter and Intra Topic Structure Learning with Word Embeddings,? 
% in International Conference on Machine Learning (ICML) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************


% Change this to where GBN is
GBN_folder = '';

addpath(GBN_folder); 


% The number of the vertical layers
T = 3;
% The number of the sub-topics
S = 3;

% The number of topics in each layer
K0 = 100;
K = ones(1,T)*K0;  

% Document classification, where we need to divide the corpus into
% training/testing docs
% Para.evaluation = 'dc'; 

% Perplexity, where we need to divide one doc into training/testing words
Para.evaluation = 'perplexity'; 

% The proportion of the words for training in the training docs
Para.train_word_prop= 0.2;

% Burnin and collection iterations for training, using more iterations may
% lead to better performance than repored in the paper
Para.TrainBurnin = 2000*ones(1,T);
Para.TrainCollection = 2000*ones(1,T);

Para.CollectionStep = 5;

% Burnin and collection iterations for testing, i.e. inferece \theta for
% testing docs
Para.TestBurnin = 500;       
Para.TestCollection = 500;




% The WS dataset used in the paper. If you want to use the dataset, please
% cite the original paper cited in our paper
dataset = load('data/WS.mat');

% We pre-split the docs into training (80%) and testing (20%), as in the
% paper
Para.train_idx = dataset.train_idx;
Para.test_idx = dataset.test_idx;

Para.Y =  dataset.labels;

trial = 1;
rng(trial,'twister');

% Run WEDTM
[ParaGlobal,ParaLocal,Accuracy_all] = ...
WEDTM(dataset.doc, dataset.embeddings, K, T, S, Para);

% Save the model
if ~exist('./save_demo','dir')
   mkdir('./save_demo') 
end
save('./save_demo/model_WS.mat','ParaGlobal','ParaLocal','Accuracy_all');

% Show the firs-layer normal topics and sub-topics
show_sub_topics(ParaGlobal, dataset.vocabulary);




