Refine section 1 of this research concept and direction based on current writing and lit review.&#x20;

Wri [https://arxiv.org/pdf/1908.09772](https://arxiv.org/pdf/1908.09772)

# Distribution Enforcement via Random Probe and Distribution Nudging

\*\*Section 1. Motivation\*\*

In many deep learning applications, the general approach is to maximize some likelihood functions or equivalently minimize some variations of entropy functions.  In addition, the data is typically high dimensional, and highly nonlinear. For instance, the data can contain images, voice or video files.  To extract features from the data, at some hidden layer(s), Gaussian or other assumptions are usually assumed.  

\*\*\*Distribution Enforcement (DE)\*\*\*

To construct a model, the mathematically rigorous approach is to specify model assumptions explicitly. These are assumed to be probabilistic statements throughout this paper.  Granted, from the engineering perspective, this is sometimes cumbersome and, at other times, may be even impossible. Nonetheless, it is usually desirable.  

In this paper, it is argued that the set of assumptions should be rigorously enforced via conventional backpropagation training. There are consequences when this is not emphasized enough. For instance, “posterior collapse” \\\[add ref.] is observed in practice.

\\\[add more examples of bad consequences due to the lack of “distribution enforcement”.]

\*\*\*Random Probe (RP)\*\*\*

The set of assumptions is frequently probabilistic in nature. There are many probabilistic verification tools that can be used, such as those adapted from “hypotheses testing” in statistics.  High dimensionality can be a hurdle.  However, this difficulty can be overcome by picking a random one-dimensional random projection. This method will be referred to as “random probe” method in this paper.  In the case of high-dimensional Gaussian distribution, random probe reduces it to one dimensional standard Gaussian, and then loss function can be constructed by using conventional statistics to measure distributional discrepancy.  As an example, Kolmogorov-Smirnov distance can be used.

\*\*\*Distribution Nudging\*\*\*

Once it is decided that “distribution enforcement” is necessary, it is usually possible to pick tools and embed them into losses in backpropagation training. Indeed, many of the tools are relatively inexpensive in terms of computational cost.  \\\[add other minor technical contributions besides RP].  In this paper, this is illustrated through examples.

\*\*\*Scope of investigation\*\*\*

The above line of investigation will be referred to as DERP (Distribution Enforcement with Random Probe).  Two situational examples will be used to illustrate DERP:

Suppose images or other input data of nontrivial type are given.  It is usually a good idea to represent them as vectors by invoking a trained encoder. In Variational AutoEncoder (VAE) \\\[add ref.], the representation is in the hidden feature layer of which is assumed to have Gaussian distribution. The hidden layer can then in turn drives reconstruction of input and prediction. With the ELBO (Evidence Lower BOund \\\[add ref.]) calculation, it has achieved some impressive results with clean architecture.  With DERP, the original layout of the problem will be reformulated with a set of rigorous mathematical assumptions, then training is done by adding other elements, including RP (Random Probe).

Vector Quantization (VQ \\\[add ref.]) is a discretization method with an internal “codebook” (or dictionary) that can be combined with VAE or can be used independently.  It is natural to desire the codebook to be well spread out in its representation space. It will be shown that normality of the codebook can be trained via RP in this paper.

\*\*Section 2. Technical Framework\*\*

\*\*\*Notations\*\*\*

Capital letter, such as X, is a r.v. (random variable) on R^1, while \<u>\*\*X\*\*\</u> is a r.v. on higher dimensions; sampled values are denoted in lower case x, \<u>\*\*x\*\*\</u>, respectively.

p(.) is used to denote a generic p.d.f. (probability density function) or a probability function; it is also used to denote the probability law or distribution. When there is a danger of inducing confusion, it will be written more explicitly - So, p(x|z) \\\= p(x|Z\\\=z) is the conditional distribution of x given Z\\\=z. q(.) \\\= q(.; theta) is used to identify the current trained version of p(.), with parameters (usually the layer weights) theta - In most cases, theta will be suppressed. As a more complicated example, suppose \<u>\*\*x\*\*\</u> is an image, \<u>\*\*z\*\*\</u> is its feature representation, \<u>\*\*x\*\*\</u>^hat is the recovered image via \<u>\*\*z\*\*\</u>. Then 

\&#x9;p(\<u>\*\*x\*\*\</u>^hat | \<u>\*\*x\*\*\</u>) \\\= integral ( p(\<u>\*\*x\*\*\</u>^hat | \<u>\*\*z\*\*\</u>) p(\<u>\*\*z\*\*\</u>|\<u>\*\*x\*\*\</u>) d\<u>\*\*z\*\*\</u> )

\*\*\*VAE (Variational AutoEncoder)\*\*\*

The input data consists of N i.i.d. samples from p(.) \\\= p (\<u>\*\*x\*\*\</u>, y), where \<u>\*\*x\*\*\</u> is m-by-m, representing an image, Y is a label from {0,1}.  The unobservable \<u>\*\*z\*\*\</u> is a hidden k-vector representation of \<u>\*\*x\*\*\</u>, with a priori distribution p(\<u>\*\*z\*\*\</u>) \\\= N (0, \*\*I\*\*\\\_k) where \*\*I\*\*\\\_k is the identity matrix of dimension k by k. Thus, we are working with the triplet p(\<u>\*\*x\*\*\</u>, \<u>\*\*z\*\*\</u>, y) where distributional manipulations are carried out.