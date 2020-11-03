import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, gamma, poisson
from numpy import log,exp
from numpy.random import multinomial

import pyro
#import pyro.distributions as dist
import torch
#from torch import log,exp
import time

smoke_test = ('CI' in os.environ)
pyro.enable_validation(True)

pyro.set_rng_seed(123456789)

## Hyperparameters
N=50
a=2
b=1
E=1200
BURN_IN=200

def generate_data():
    n = pyro.sample("n", pyro.distributions.Uniform(0,1))
    n = torch.round(n*N).int()
    lamda1 = pyro.sample("lamda1", pyro.distributions.Gamma(a,b))
    lamda2 = pyro.sample("lamda2", pyro.distributions.Gamma(a,b))
    lamdas = torch.ones(N,1)*lamda1
    lamdas[n:N] = lamda2
    x = torch.poisson(lamdas)
    print ("default chain n: "+str(n))
    print ("default chain r1: "+str(lamda1))
    print ("default chain r2: "+str(lamda2))
    return n, x, lamdas

def gibbs_sampling(x):
    # Initialize the chain
    n = int(round(uniform.rvs()*N)) #pyro.sample("n", pyro.distributions.Uniform(0,1))
                                    #n = torch.round(n*N).int()
    lamda1 = gamma.rvs(a,scale=1./b)#pyro.sample("lamda1", pyro.distributions.Gamma(a,b))
    lamda2 = gamma.rvs(a,scale=1./b)#pyro.sample("lamda2", pyro.distributions.Gamma(a,b))
    print ("intial chain n: "+str(n))
    print ("intial chain lamda1: "+str(lamda1))
    print ("intial chain lamda2: "+str(lamda2))
    # Store the samples
    chain_n=np.array([0.]*(E))
    chain_lambda1=np.array([0.]*(E))
    chain_lambda2=np.array([0.]*(E))
    start_time = time.time()
    for e in range(E):
        if e%100 == 0:
            print("itertaion "+str(e))
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
        lambda1 = gamma.rvs(a+sum(x[0:n]), scale=1./(n+b))#pyro.sample("lamda1", pyro.distributions.Gamma(a+sum(x[0:n]), (n.item()+b)))
        lambda2 = gamma.rvs(a+sum(x[n:N]), scale=1./(N-n+b))#pyro.sample("lamda2", pyro.distributions.Gamma(a+sum(x[n:N]), (N-n.item()+b)))
        # sample n, Equation 10
        mult_n=np.array([0]*N)#torch.zeros(N,1)
        for i in range(N):
            mult_n[i]=sum(x[0:i])*log(lambda1)-i*lambda1+sum(x[i:N])*log(lambda2)-(N-i)*lambda2
        mult_n=exp(mult_n-max(mult_n))
        n=np.where(multinomial(1,mult_n/sum(mult_n),size=1)==1)[1][0]

        # store
        chain_n      [e]=n
        chain_lambda1[e]=lambda1
        chain_lambda2[e]=lambda2

    return chain_n, chain_lambda1, chain_lambda2
# Store the samples
chain_n=np.array([0.]*(E-BURN_IN))
chain_lambda1=np.array([0.]*(E-BURN_IN))
chain_lambda2=np.array([0.]*(E-BURN_IN))
# Change-point: where the intensity parameter changes.
n, x, lamdas = generate_data()
n = n.item()
x = x.numpy()
lamdas = lamdas.numpy()
#x = np.loadtxt("./random_data.txt").reshape(50,1)
# make one big subplots and put everything in it.
# Gibbs sampler
chain_n, chain_lambda1, chain_lambda2 = gibbs_sampling(x)
chain_n =chain_n[BURN_IN:E-1]
chain_lambda1=chain_lambda1[BURN_IN:E-1]
chain_lambda2=chain_lambda2[BURN_IN:E-1]
f, (ax1,ax2,ax3,ax4,ax5)=plt.subplots(5,1)
# Plot the data
ax1.stem(range(N),x,linefmt='b-', markerfmt='bo')
ax1.plot(range(N),lamdas,'r--')
ax1.set_ylabel('Counts')
ax2.plot(chain_lambda1,'b',chain_lambda2,'g')
ax2.set_ylabel('$\lambda$')
ax3.hist(chain_lambda1,20)
ax3.set_xlabel('$\lambda_1$')
ax3.set_xlim([0,12])
ax4.hist(chain_lambda2,20, color='g')
ax4.set_xlim([0,12])
ax4.set_xlabel('$\lambda_2$')
ax5.hist(chain_n,50)
ax5.set_xlabel('n')
ax5.set_xlim([1,50])
plt.tight_layout()
plt.show()

