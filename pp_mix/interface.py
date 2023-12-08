import logging
import joblib
import os
import sys
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from itertools import combinations, product
from scipy.stats import multivariate_normal, norm
from .params_helper import make_params
from .src import python_exports as pp_mix_py


class ConditionalMCMC(object):
    def __init__(self, pp_params=None, prec_params=None,
                 jump_params=None, init_n_clus=4):
        
        self.params = make_params(pp_params,prec_params,jump_params)
        self.params.init_n_clus = init_n_clus

    def run(self, nburn, niter, thin, data, log_every=20, bernoulli=False):
        #check_params(self.params, data, bernoulli)
        if data.ndim == 1:
            self.dim = 1
        else:
            self.dim = data.shape[1]
        
        #out = pp_mix_py._run_pp_mix(
        #    nburn, niter, thin, data, self.serialized_params, bernoulli, log_every)
        out = pp_mix_py._run_pp_mix(
            nburn, niter, thin, data, self.params, bernoulli, log_every)
        
        self.chains = out
    
    def sample_predictive(self):
        if self.dim == 1:
            out = pp_mix_py._sample_predictive_univ(self.chains)
        else:
            out = pp_mix_py._sample_predictive_multi(self.chain,self.dim)
        return out 