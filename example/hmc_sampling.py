import os
import torch
import numpy as np
import matplotlib.pyplot as plt


import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
import time

smoke_test = ('CI' in os.environ)
pyro.enable_validation(True)
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS

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

def model(data):
    # Global variables.
    n = pyro.sample("n", pyro.distributions.Uniform(0,1))
    n = torch.round(n*N).int()
    with pyro.plate('components', 2):
        locs = pyro.sample('locs', dist.Gamma(a, b))

    with pyro.plate('data', len(data)):
        # Local variables.
        assignment = torch.zeros(N, 1).long()
        assignment[n:N] = 1
        assignment = torch.reshape(assignment, (-1,))
        pyro.sample('obs', dist.Normal(locs[assignment], 2), obs=data)

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
n, data, lamdas = generate_data()
#data = np.loadtxt("./random_data.txt").reshape(50,1)
#data = torch.tensor(data)

kernel = NUTS(model)
mcmc = MCMC(kernel, num_samples=E-BURN_IN, warmup_steps=BURN_IN)
mcmc.run(data)
posterior_samples = mcmc.get_samples()

n = posterior_samples['n']
n = torch.round(n*N).int()
locs = posterior_samples['locs']
lamda1, lamda2 = posterior_samples["locs"].t()

chain_n=torch.zeros(E-BURN_IN)
chain_lambda1=torch.zeros(E-BURN_IN)
chain_lambda2=torch.zeros(E-BURN_IN)

chain_n =n[BURN_IN:E].numpy()
chain_lambda1=lamda1[BURN_IN:E].numpy()
chain_lambda2=lamda2[BURN_IN:E].numpy()
print(chain_n)
f, (ax1,ax2,ax3,ax4,ax5)=plt.subplots(5,1)
# Plot the data
ax1.stem(range(N),data,linefmt='b-', markerfmt='bo')
#ax1.plot(range(N),lamdas,'r--')
ax1.set_ylabel('Counts')
ax2.plot(chain_lambda1,'b',chain_lambda2,'g')
ax2.set_ylabel('$\lambda$')
ax3.hist(chain_lambda1,20)
ax3.set_xlabel('$\lambda_1$')
ax3.set_xlim([0,12])
ax4.hist(chain_lambda2,20,color='g')
ax4.set_xlim([0,12])
ax4.set_xlabel('$\lambda_2$')
ax5.hist(chain_n,50)
ax5.set_xlabel('n')
ax5.set_xlim([1,50])
plt.show()

