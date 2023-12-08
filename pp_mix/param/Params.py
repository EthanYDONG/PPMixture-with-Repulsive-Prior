import numpy as np
import pandas as pd

class WishartParams():
    def __init__(self, u, identity, sigma, dim):
        self.u = u
        self.identity = identity
        self.sigma = sigma
        self.dim = dim

class FixedMultiPrecParams():
    def __init__(self, sigma, dim):
        self.sigma = sigma
        self.dim = dim

class FixedUnivPrecParams():
    def __init__(self, sigma):
        self.sigma = sigma

class GammaParams():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

class StraussPrior():
    def __init__(self,beta_l,beta_u,gamma_l,gamma_u,
                 r_l,r_u) -> None:
        self.beta_l = beta_l
        self.beta_u = beta_u
        self.gamma_l = gamma_l
        self.gamma_u = gamma_u
        self.r_l = r_l
        self.r_u = r_u

def StraussInit():
    def __init__(self,beta,gamma,R):
        self.beta = beta
        self.gamma = gamma
        self.R = R 

class StraussParams():
    def __init__(self,fixed_param, prior=None, init=None):
        self.fixed_param
        self.prior = prior
        self.init = init 
        self.has_prior = False
        self.has_init = False
        if self.prior:
            self.has_prior = True
        if self.has_init:
            self.has_init = True

class NrepParams():
    def __init__(self,u ,p):
        self.u = u
        self.p = p

class DPPParams():
    def __init__(self, nu, rho, N, s, fixed_params):
        self.nu = nu
        self.rho = rho
        self.N = N
        self.s = s
        self.fixed_param = fixed_params



class Params():
    def __init__(self):
        self.has_strauss = False
        self.has_nrep = False
        self.has_dpp = False

        self.strauss_param = None
        self.nrep_param = None
        self.dpp_param = None
        
        self.has_gamma_jump = False
        self.gamma_jump_param = None

        self.has_fixed_multi_prec = False
        self.fixed_multi_prec = None

        self.has_wishart = False
        self.wishart_param = None

        self.has_fixed_univ_prec = False
        self.fixed_univ_prec = None

        self.has_gamma_prec = False
        self.gamma_prec = None
        
    def Initialize(self):
        pass