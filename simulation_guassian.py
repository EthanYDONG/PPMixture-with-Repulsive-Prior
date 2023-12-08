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
from sklearn.datasets import make_blobs  # 用于创建模拟数据

np.random.seed(0)
num_samples = 1000
num_features = 2
num_clusters = 4

data, labels = make_blobs(n_samples=num_samples, n_features=num_features, centers=num_clusters, random_state=0)

nrep = 10
# dimrange = [1, 2, 3, 4, 5]
# strauss_times = np.zeros((nrep, len(dimrange)))

# nrange = [5, 10, 20]
#dpp_times = np.zeros((nrep, len(dimrange), len(nrange)))

dim = 2
"""
###################strauss pp#######################
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
strauss_sampler.run(0, 1000, 100, data, log_every = 1)
strauss_times = time.time() - start

print(strauss_sampler.chains)"""
prec_params = WishartParams()
prec_params.nu = dim+1
prec_params.identity = True
prec_params.sigma = 3
prec_params.dim = dim

n =5
dpp_params = DPPParams()
dpp_params.nu = 2.0
dpp_params.rho = 3.0
dpp_params.N = n

gamma_jump_params = GammaParams()
gamma_jump_params.alpha = 1
gamma_jump_params.beta = 1

dpp_sampler = ConditionalMCMC(
    pp_params=dpp_params, 
    prec_params=prec_params,
    jump_params=gamma_jump_params)
start = time.time()
dpp_sampler.run(0, 10000, 5, data)
dpp_times = time.time() - start
print("运行时间为",dpp_times)






