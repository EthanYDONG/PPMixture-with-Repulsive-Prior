from .src.proto.Params import *
import logging

import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

def make_params(pp_params, prec_params, jump_params=None):
    params = Params()
    if isinstance(pp_params, StraussParams):
        params.strauss_param = pp_params
        params.has_strauss = True
    elif isinstance(pp_params, NrepParams):
        params.nrep_param = pp_params
        params.has_nrep = True
    elif isinstance(pp_params, DPPParams):
        params.dpp_param = pp_params
        params.has_dpp = True
    else:
        #logging.error("pp_params type not valid")
        print('pp_params type not valid')

    if isinstance(prec_params, WishartParams):
        params.wishart_param = prec_params
        params.has_wishart = True
    elif isinstance(prec_params, FixedMultiPrecParams):
        params.fixed_multi_prec = prec_params
        params.has_fixed_multi_prec = True
    elif isinstance(prec_params, GammaParams):
        params.gamma_prec = prec_params
        params.has_gamma_prec = True
    elif isinstance(prec_params, FixedUnivPrecParams):
        params.fixed_univ_prec = prec_params
        params.has_fixed_multi_prec = True
    else:
        #logging.error("prec_params type not valid")
        print('prec_params type not valid')

    if isinstance(jump_params, GammaParams):
        params.gamma_jump_param = jump_params 
        params.has_gamma_jump = True
    else:
        #logging.error("jump_params not recognized")
        print('jump_params not recognized')

    return params

def make_default_strauss(data, nstar=10, m_max=30, prior_m=None):
    if data.ndim == 1:
        data = data.reshape(-1, 1) 
        
    params = StraussParams()
    pdist = pairwise_distances(data).reshape((-1, ))
    grid = np.linspace(np.min(pdist), np.max(pdist), 200)
    
    if len(pdist) > 10000:
        pdist = np.random.choice(pdist, 10000, False)
    dens_estimate = gaussian_kde(pdist).evaluate(grid)

    """plt.plot(grid, dens_estimate, label='Density Estimate')
    plt.show()
    plt.savefig('density_plot_mydata.png')"""

    params_init = StraussInit()
    # params_init.R = grid[argrelextrema(dens_estimate, np.less)[0][0]]
    params_init.R = 0.01

    params_init.gamma = np.exp(-nstar)

    params_prior = StraussPrior()
    ranges = np.vstack([np.min(data, axis=0), np.max(data, axis=0)]) * 2
    vol = np.prod(np.diff(ranges, axis=0))
    params_prior.beta_l = 1.0 / vol
    params_prior.beta_u = m_max / vol

    params.init = params_init
    params.prior = params_prior
    params.has_init = True
    params.has_prior = True
    
    if prior_m is not None:
        params.init.beta = prior_m / vol
    else:
        params.init.beta = 0.5 * (params.prior.beta_l + params.prior.beta_u)
    return params
