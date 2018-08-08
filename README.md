# WEDTM

The code for the paper "Inter and Intra Topic Structure Learning with Word Embeddings" in ICML 2018 [PDF](http://proceedings.mlr.press/v80/zhao18a/zhao18a.pdf).

Key features:

1. WEDTM is a deep topic model that discovers topic hierarchies. 
2. WEDTM is also able to discover "sub-topics" with the help of word embeddings.
3. Excellent performance on perplexity, document classification, and topic coherence.

# Run WEDTM

0. The code has been tested in MacOS and Linux (Ubuntu). To run it on Windows, you need to re-compile ```GNBP_mex_collapsed_deep_WEDTM.c``` with MEX and a C++ complier.

1. Requirements: Matlab 2016b (or later) and the code of [GBN](https://github.com/mingyuanzhou/GBN).

2. Make sure GBN runs properly on your machine. 

3. We have offered the WS dataset used in the paper, which is stored in MAT format, with the following contents:
- doc: a V by N count (sparse) matrix for N documents with V words in the vocabulary
- embeddings: a V by L matrix for the L dimensional word embeddings for V words
- vocabulary: the words in the vocabulary
- labels: the label matrix for the documents (only for document classification)
- label_names: the label names (only for document classification)
- train_idx: the indexes of documents for training (only for document classification)
- test_idx: the indexes of documents for testing (only for document classification)

Please prepare your own documents in the above format. If you want to use this dataset, please cite the original papers, which are cited in our paper.

4. Run ```demo_WEDTM.m```:
- Specify where the GBN code is installed and some model parameters.
- Follow the comments and run it.
- The code should yield the results reported in the paper.
- I've found that if you use more MCMC iterations, the model will have better performance than reported in the paper.😂

# Notes

1. As WEDTM adapts GBN for a part of its model structure, the code heavily relies on GBN and basically follows the code structure of GBN.

2. For the Polya-Gamma sampler (```PolyaGamRnd_Gam.m```), I used [Mingyuan Zhou](https://mingyuanzhou.github.io)'s implementation, described in  ["Parsimonious Bayesian deep networks"](https://arxiv.org/abs/1805.08719). If you want to use the sampler, please cite the paper. 

3. For the sampling of W, I partly referred to the implementation of [DPFA](https://github.com/zhegan27/dpfa_icml2015) by [Gan Zhe](https://zhegan27.github.io).
