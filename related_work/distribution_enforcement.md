# Distribution Enforcement and Probabilistic Assumptions in Deep Learning

## Key Papers on Distribution Enforcement

### Probability Engineering (Zhang, 2025)
- **Citation**: Zhang, J. "Advancing Deep Learning through Probability Engineering: A Pragmatic Paradigm for Modern AI." arXiv:2503.18958
- **Key Innovation**: Treats learned probability distributions as engineering artifacts that can be actively modified
- **Relevance**: Directly addresses our concept of distribution enforcement - explicitly modifying distributions rather than just fitting them
- **Core Idea**: Rather than merely fitting or inferring distributions, actively modify and reinforce them to meet diverse AI requirements
- **Applications**: Bayesian deep learning, Edge AI, Generative AI

### Distributional Adversarial Loss (Ahmadi et al., 2024)
- **Citation**: Ahmadi, S., et al. "Distributional Adversarial Loss." arXiv:2406.03458
- **Key Innovation**: Adversarial perturbation sets are families of distributions rather than point sets
- **Relevance**: Shows how distributional assumptions can be enforced via adversarial training
- **Method**: For each example, the perturbation set is a family of distributions, loss is maximum over all associated distributions

### Distributional Input Projection Networks (Hao et al., 2025)
- **Citation**: Hao, Y., et al. "Towards Better Generalization via Distributional Input Projection Network." arXiv:2506.04690
- **Key Innovation**: Projects inputs into learnable distributions at each layer to induce smoother loss landscapes
- **Relevance**: Practical implementation of distribution enforcement throughout network layers
- **Method**: DIPNet projects inputs into learnable distributions, promoting smoothness and better generalization

## Probabilistic Representation and Verification

### Probabilistic Representation of Deep Learning (Lan & Barner, 2019)
- **Citation**: Lan, X., Barner, K.E. "A Probabilistic Representation of Deep Learning." arXiv:1908.09772
- **Key Innovation**: Explicit probabilistic interpretation where neurons define Gibbs distributions
- **Relevance**: Provides foundation for understanding how distributional assumptions manifest in neural networks
- **Core Framework**: 
  - Neurons define energy of Gibbs distribution
  - Hidden layers formulate Gibbs distributions
  - Whole DNN architecture as Bayesian neural network

### Probabilistic Robustness in Deep Learning (Zhao, 2025)
- **Citation**: Zhao, X. "Probabilistic Robustness in Deep Learning: A Concise yet Comprehensive Guide." arXiv:2502.14833
- **Key Innovation**: Quantifies likelihood of failures under stochastic perturbations
- **Relevance**: Shows how probabilistic verification can be used to test distributional assumptions
- **Method**: PR offers practical perspective by quantifying failure probability rather than worst-case scenarios

## Neural Network Verification and Testing

### Radon-Kolmogorov-Smirnov Test (Paik et al., 2023)
- **Citation**: Paik, S., et al. "Maximum Mean Discrepancy Meets Neural Networks: The Radon-Kolmogorov-Smirnov Test." arXiv:2309.02422
- **Key Innovation**: Neural network-based generalization of K-S test to multiple dimensions
- **Relevance**: Provides computational framework for our "random probe" concept using neural networks
- **Method**: Uses ridge splines (single neurons) as witnesses in statistical tests
- **Connection**: Direct implementation of statistical testing via neural network optimization

### Normality Testing with Neural Networks (Simić, 2020)  
- **Citation**: Simić, M. "Normality Testing with Neural Networks." arXiv:2009.13831
- **Key Innovation**: Treats normality testing as binary classification, outperforms traditional tests
- **Relevance**: Shows neural networks can effectively perform distributional verification
- **Performance**: AUROC ≈ 1, >96% accuracy on samples with 250-1000 elements

### Robustness Distributions in Neural Network Verification (Bosman et al.)
- **Citation**: Bosman, A.W., et al. "Robustness Distributions in Neural Network Verification." JAIR
- **Key Innovation**: Uses Kolmogorov-Smirnov tests to analyze critical epsilon distributions
- **Relevance**: Shows practical application of K-S testing for neural network analysis
- **Finding**: Critical ε values of 11/12 networks follow log-normal distribution

## Summary: Connections to DERP Framework

1. **Distribution Enforcement**: Multiple papers show active modification of distributions (Probability Engineering, DIPNet)
2. **Random Probe**: Neural network-based statistical testing provides computational framework (RKS test, NN normality testing)
3. **Verification**: K-S tests and variants can verify distributional assumptions in neural networks
4. **Applications**: Practical implementations across VAE, robustness analysis, and generative models