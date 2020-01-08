# MoDT #

Mixture of drifting t-distribution model for sorting spikes and measuring unit isolation

Shan KQ, Lubenov EV, and Siapas AG (2017). Model-based spike sorting with a mixture of drifting t-distributions. *J Neurosci Methods* 288, 82-98.

# What is this? #

MoDT is an acronym for "mixture of drifting t-distributions", a mixture model that is useful for clustering non-stationary, heavy-tailed data. It also provides a way to estimate the misclassification error.

This repository contains a MATLAB class `MoDT` that represents a MoDT model and implements the expectation-maximization (EM) algorithm for fitting this model to the given data. It also provides some functions for other model manipulations.

## Tell me more about this mixture model ##

So let's start with a mixture of Gaussians (a.k.a. Gaussian mixture model).

First, we'll replace the Gaussians with multivariate t-distributions. The t-distribution is a lot like the Gaussian, except that it has an additional parameter (*&nu;*, known as "degrees of freedom") that controls how heavy the tails of the distribution are. Although *&nu;* could be fitted to the data, we are going to treat it as a user-defined parameter, and use the same *&nu;* for all clusters.

The multivariate t-distribution has two other parameters: the location *&mu;* and the scale matrix *&Sigma;*. These are analogous to the Gaussian mean and covarance parameters, respectively. I will try to remember to refer to these as the cluster "location" and "scale", rather than the cluster "mean" and "covariance", but I might slip up sometimes.

Next, we are going to allow the *&mu;* to change over time, although we will hold the *&Sigma;* fixed. We will represent these time-varying cluster location parameters by discretizing the time axis into a finite number of frames. We include a prior (which you should think of as a regularizer, rather than an attempt to accurately model the cluster drift) that encourages the clusters to not move too much from one frame to the next. How much is "too much" is controlled by a drift covariance matrix *Q*, which we will treat as a user-defined parameter.

So that's basically it. The mixture of drifting t-distributions is a lot like a mixture of Gaussians, except that clusters are instead t-distributions, and they are allowed to drift over time.

## What are the free parameters? ##

A MoDT model with *K* clusters in a *D*-dimensional space and *T* time frames, with *&nu;* and *Q* treated as user-defined parameters, has the following free parameters:
* *K* mixing proportions *&alpha;*, which sum to 1.
* *DTK* cluster locations *&mu;*
* *D<sup>2</sup>K* cluster scales *&Sigma;*, which are symmetric and positive-definite.

I would be hesitant using these parameter counts in something like the Akiake Information Criterion (AIC) or Bayes Information Criterion (BIC) though, because the drift regularizer effectively decreases the "free-ness" of the cluster location parameter. Model selection (e.g. choosing the appropriate number of clusters) will be a tricky problem.

## How is this represented by the `MoDT` class? ##

The class properties can be broken into a few sections.

**Dimensions:**
These are read-only, and are automatically updated if the underlying data change (e.g. if new data are attached).
* `D` Number of dimensions in the feature space
* `K` Number of components (a.k.a. "clusters") in the mixture model
* `T` Number of time frames
* `N` Number of spikes currently attached

**Fitted model parameters:**
These are fitted to the given data using the EM algorithm. You *could* set these manually using the `setParams` method, but that'd be a little unusual.
* `alpha` [*K*] vector of mixing proportions *&alpha;*. These describe the relative size (in terms of the number of spikes) of each cluster, and must sum to 1.
* `mu` [*D* x *T* x *K*] cluster locations (in *D* dimensions) over the *T* time frames, for each of the *K* clusters.
* `C` [*D* x *D* x *K*] cluster scale matrices ([*D* x *D*] symmetric positive-definite matrices) for each of the *K* clusters.

**User-defined parameters:**
These are set using the `setParams` method. If you change these, remember that you will need to run `EM` again to refit the model parameters.
* `mu_t` [*T*+1] vector of time frame boundaries. A spike that falls within the half-closed interval `mu_t[t] <= spk_t < mu_t[t+1]` is considered to be in time frame `t`. This can also be set by the `attachData` method.
* `nu` t-distribution degrees-of-freedom parameter *&nu;*. This controls how heavy the tails of the distribution are. Smaller values correspond to heavier tails, and infinity corresponds to a Gaussian distribution.
* `Q` [*D* x *D*] symmetric positive-definite drift regularization matrix. This can also be a scalar, in which case it is interpreted as the identity matrix times that scalar. Small values correspond to more regularization, producing smoother cluster trajectories. However, interpreting these values will depend on the scaling of your feature space as well as the time frame duration.
* `C_reg` Don't use this; use `max_cond` instead. The goal here is to ensure the scale matrices are numerically well-conditioned, and `max_cond` is just less invasive in how it achieves this.

**Attached data:**
These are set using the `attachData` method. This defines the spike data that we are fitting the model to when we call the `EM` method. Alternatively, you can attach a new set of data (without re-fitting the model) to evaluate what the model has to say about *that* data.
* `spk_Y` [*D* x *N*] spike feature vectors (in *D* dimensions) for *N* spikes.
* `spk_t` [*N*] spike times. These are used to determine which time frame each spike belongs to, and all of these must fall within the range covered by `mu_t`. You can use whatever time units you like, as long as it's consistent with `mu_t`, but if you like other people telling you what to do, then use milliseconds, and make 0 the start of the recording.
* `spk_w` [*N*] spike weights. We have found that you can get a pretty good model fit in a fraction of the time by using only a subset of your data for fitting. However, you will need to weight this subset in order to maintain consistency with the full dataset. Even if all the spikes are weighted equally, you still need to specify these weights because of the drift regularizer.

**Other class properties:**
These control certain aspects of the computations.
* `max_cond` Maximum allowable condition number of the cluster scale matrices `C`. The scale matrices are inverted during the M-step, so they need to be numerically well-conditioned.
* `use_gpu` Use GPU for computation, and store certain data matrices in GPU memory.
* `use_mex` Use precompiled subroutines (a.k.a. MEX files) to accelerate certain computations.

# What's a typical workflow using this package? #

Let's say you have a collection of spike times `spk_t` and spike features `spk_Y`.

## Choose values for the user-defined parameters ##

The user-defined parameters are `mu_t`, `nu`, `Q`, and `C_reg`. Of these, `mu_t` will be automatically set when you call `attachData` and you should just leave `C_reg` as its default of zero.

We discuss the choice of `nu` in our paper, and I think 7 is a pretty good choice.

    >> nu = 7

We also discuss the choice of `Q` in our paper, but the right choice is going to depend a lot on your data. Obviously it depends on how stable your recordings are (more stable = smaller `Q`), but also the firing rates (higher firing = larger `Q` for the same amount of smoothing). But at a practical level, it also depends on the scaling of your feature space (if you multiply your features by a factor of `x`, then you should also scale your `Q` by a factor of `x^2`) and the duration of your time frames (since `Q` is in units of (feature space units)<sup>2</sup> per time frame).

So if you're looking for a rule of thumb to start, let's assume we'll be using the default one-minute time frames and set

    >> q = 0.0001 * mean(var(spk_Y,0,2))

This is an isotropic regularizer (same in all directions, will be interpreted as `Q = q*eye(D)`) and it's set as a small fraction of the average variance of each dimension of the data.

Now we can construct a `MoDT` object with these parameters:

    >> model = MoDT('nu',nu, 'Q',q)

## Attach data ##

Attach our data to the model so that we can fit to it later.

    >> model.attachData(spk_Y, spk_t)

This will also define the `mu_t` parameter so that the frame duration is 1 minute (assuming `spk_t` is in ms) and it spans the range covered by `spk_t`.

You may need to be careful with this automatic initialization of `mu_t` if your spikes only contain a subset of the full data, because then the full data might contain spikes outside of this time range. You can address this by ensuring that your subset contains the first and last spike, by specifying the full time range to `attachData`, or by manually defining `mu_t` before attaching the data.

## Initialize the model ##

We now need to assign initial values to the fitted parameters `alpha`, `mu`, and `C`.

If you have an initial set of cluster assignments, perhaps obtained through a different clustering method, you can initialize the MoDT model based on those assignments. Assuming that `spk_cl` is a [*N*]-length vector specifyng which of the *K* clusters each of the *N* spikes belongs to:

    >> model.initFromAssign(spk_cl)

Alternatively, you could start with everything in a single cluster, and rely entirely on the split and merge operations in the next section to arrive at the desired model:

    >> model.initFromAssign(ones(model.N,1))

## Fit the model to the data ##

Running the EM algorithm is fairly straightforward:

    >> model.EM()

However, this will only bring you from your initial point to a nearby local optimum. If that's all you want, then great! Otherwise, it is up to you to take steps to escape local optima and to determine the appropriate number of clusters.

We've had pretty good success with a split-and-merge approach (Ueda N, Nakano R, Ghahramani Z, and Hinton GE (2000). SMEM algorithm for mixture models. *Neural Computation* 12(9), 2109-2128), which also happens to mirror a typical strategy that human operators use during interactive clustering.

Unfortunately, this package does not implement the split-and-merge algorithm, nor does it provide the visualizations that would help guide interactive clustering. All it provides are the following elementary model operations:

* `split` Split a single cluster into two.
* `merge` Merge two or more clusters into a single cluster.
* `remove` Remove the selected cluster(s). The spikes will still remain and will be reassigned to other clusters.

Performing these operations should be followed by a call to `EM` to refit the model.

## Query the model ##

Once you've fit the model, you can use `getValue` to return a variety of computed values from the model. I'll group them here by potential use case:

* Cluster assignments
  * `posterior` [*N* x *K*] posterior probability that spike *n* belongs to cluster *k*, a.k.a. "soft assignments".
  * `assignment` [*N*] vector indicating the most likely cluster (1..*K*) for each spike, a.k.a. "hard assignments".
  * `clusters` Reverse lookup of `assignment`, indicating which spikes (1..*N*) belong to each cluster.
* Estimated misclassification errors
  * `confusionMat` [*K* x *K*] confusion matrix. The `(i,j)`th entry tells us the number of spikes assigned to cluster `i` that were actually generated by cluster `j`, according to our mixture model. You can then obtain the expected number of false positives or false negatives by subtracting the diagonal (true positives) from the row sum (total assigned) or column sum (total generated), respectively.
* Other spike-level info
  * `mahalDist` [*N* x *K*] squared Mahalanobis distance from cluster *k* to spike *n*.
  * `spkFrame` [*N*] vector indicating which time frame (1..*T*) each spike belogns to.
* Log-likelihoods
  * `logLike` The sum of `dataLogLike` and `priorLogLike`, and the objective value being maximized by the EM algorithm.
  * `dataLogLike` Data log-likelihood, i.e. the log of the probability density of observing the attached spike data given the current model parameters.
  * `priorLogLike` Log of the probability density of observing the current model parameters given our prior on cluster drift. Well, sorta. It's not really a true distribution (its integral over parameter space is not 1). Maybe better to think of it as a regularizer that adds a penalty term to the overall objective.

Remember that you don't need to refit the model after attaching new data. You could fit the model to a subset, attach the full dataset, and then immediately call `getValue` to obtain the cluster assignments and misclassification errors on this full dataset.

# Other features #

There are a few other features provided by this package that you may find useful.

## Serialization ##

MATLAB objects are great and all, but sometimes we want structs so that we can save them in a more easily-accessible format. The `saveobj` and `loadobj` methods have been appropriately overridden to provide these.

## GPU support ##

If you can create `gpuArray` objects in MATLAB, then you should consider setting `use_gpu=true` because these computations can be up to 10x faster on the GPU.

Most of this increase is due to faster memory bandwidth, rather than increased computational power (FLOPS). The arithmetic intensity of these operations scales with *D*, and typical values of *D* are low enough that we are memory-bound instead of compute-bound.

## MEX routines ##

This package also contains C++/CUDA source code for a handful of subroutines. You can compile these using the class method

    >> MoDT.buildMexFiles()

And assuming the build was successful, you can run unit tests on these MEX files using

    >> MoDT.validateMex()

Finally, enable the use of these MEX files by setting the class property `use_mex=true`.
