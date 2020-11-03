#Example
Gibbs sampling is widely used in Bayesian inference. The underlying logic of MCMC sampling is that we can extimate any desired expecation by ergodic averages.
In this example we developed Gibbs sampler and HMC for change-point model to learn about MCMC techniques.
[change-point model](http://www2.bcs.rochester.edu/sites/jacobslab/cheat_sheet/GibbsSampling.pdf)
1. *.scipy.py used Gibbs sampler developed based on scipy library
2. *.pyro.py used Gibbs sampler developed based on pyro library
3. hmc_sampling.py used HMC sampler with using the [No-U-Turn Sampler (NUTS)](https://arxiv.org/abs/1111.4246) with adaptive path length and mass matrix adaptation.

