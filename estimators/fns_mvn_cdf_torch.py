import torch
import numpy as np
from scipy.stats import multivariate_normal


#calculate prob
def compute_mvn_cdf(z, cov, maxpts=10000, abseps=1e-4, releps=1e-4):
    '''
    Compute MVN-CDF probability from torch tensors
    z: tensor, [b, n_c-1]
    cov: tensor, [b, n_c-1, n_c-1]
    maxpts, abseps, releps: arguments to control precision/time for multivariate_normal.cdf 
    '''
    #create numpy tensors
    z_np = z.detach().cpu().numpy() #[b, 9]
    mean_np = np.zeros_like(z_np) #[b, 9]
    cov_np = cov.detach().cpu().numpy() #[b, 9, 9]
    n = z.size(0) #n=b

    #create tensor to store probabilities
    p = np.zeros(n)

    #calculate mvncdf probabilities
    for i in range(n):
        p[i] = multivariate_normal.cdf(x=z_np[i], mean=mean_np[i], cov=cov_np[i], maxpts=maxpts, abseps=abseps, releps=releps)

    p = torch.from_numpy(p)
    p = p.to(z.device) #put p on original device (whatever device z and cov are on)

    return p #[b]


#calculate neg-log-prob
def compute_mvn_cdf_nlp(z, cov, maxpts=10000, abseps=1e-4, releps=1e-4):
    prob = compute_mvn_cdf(z, cov, maxpts, abseps, releps)
    return prob.log().neg()

