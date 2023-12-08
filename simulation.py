import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import skewnorm
from scipy.stats import norm
from scipy.interpolate import griddata
from pp_mix.src.proto.proto import *
from pp_mix.src.proto.Params import *
from pp_mix.interface import ConditionalMCMC
from pp_mix.params_helper import *

def generate_data(dim):
    if (dim == 1):
        data = np.concatenate([
            np.random.normal(loc = 5, size=100), 
            np.random.normal(loc = -5, size=100)
        ])
        return data.reshape(-1,1)
    else:
        data = np.vstack([
           [mvn.rvs(mean=np.ones(dim) * 5, cov=np.eye(dim)) for _ in range(100)],
           [mvn.rvs(mean=np.ones(dim) * (-5), cov=np.eye(dim)) for _ in range(100)]])
        return data
    

nrep = 10
dimrange = [1, 2, 3, 4, 5]
strauss_times = np.zeros((nrep, len(dimrange)))

nrange = [5, 10, 20]
dpp_times = np.zeros((nrep, len(dimrange), len(nrange)))

for i in range(nrep):
    print("Running rep: {0}".format(i))
#   for j, dim in enumerate([1, 2, 3, 4, 5]):
    for j, dim in enumerate([2, 3, 4, 5]):
        data = generate_data(dim)
        if dim == 1:
            #prec_params = GammaParams(alpha=1.0, beta=1.0)
            prec_params = GammaParams()
            prec_params.alpha = 1.0
            prec_params.beta = 1.0
        else:
            # prec_params = WishartParams(
            #     nu=dim+1,identity=True, sigma=3, dim=dim)
            prec_params = WishartParams()
            # WishartParams.nu = dim+1
            # WishartParams.identity = True
            # WishartParams.sigma = 3
            # WishartParams.dim = dim
            prec_params.nu = dim+1
            prec_params.identity = True
            prec_params.sigma = 3
            prec_params.dim = dim
        # gamma_jump_params = GammaParams(alpha=1, beta=1)
        gamma_jump_params = GammaParams()
        gamma_jump_params.alpha = 1
        gamma_jump_params.beta = 1


        strauss_params = make_default_strauss(data)
        strauss_params.fixed_params = False
        strauss_sampler = ConditionalMCMC(
            pp_params=strauss_params,
            prec_params=prec_params,
            jump_params=gamma_jump_params)

        start = time.time()
        strauss_sampler.run(20, 10, 5, data)
        strauss_times[i, j] = time.time() - start

        print(strauss_sampler.chains)

            
        for k, n in enumerate(nrange):
            
            if n == 20 and dim > 3:
                continue
            
            if n == 10 and dim > 4:
                continue
                
            #dpp_params = DPPParams(nu=2.0, rho=3.0, N=n)
            #dpp_params = DPPParams()
            dpp_params = DPPParams()
            dpp_params.nu = 2.0
            dpp_params.rho = 3.0
            dpp_params.N = n

            dpp_sampler = ConditionalMCMC(
                pp_params=dpp_params, 
                prec_params=prec_params,
                jump_params=gamma_jump_params)
            start = time.time()
            dpp_sampler.run(0, 2, 5, data)
            dpp_times[i, j, k] = time.time() - start