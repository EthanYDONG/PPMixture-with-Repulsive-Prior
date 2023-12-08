import numpy as np
import pandas as pd

class WishartParams():
    #def __init__(self, u, identity, sigma, dim):
    def __init__(self, *args):
        if len(args)==0:
            #self.u = 0
            self.nu = 0
            self.identity = 0
            self.sigma = 0
            self.dim = 0
        else:
            #self.u = args[0]
            self.nu = args[0]
            self.identity = args[1]
            self.sigma = args[2]
            self.dim = args[3]

class FixedMultiPrecParams():
    #def __init__(self, sigma, dim):
    def __init__(self,*args):
        if len(args) == 0:
            self.sigma = 0
            self.dim = 0
        else:
            self.sigma = args[0]
            self.dim = args[1]

class FixedUnivPrecParams():
    def __init__(self, *args):
        if len(args) == 0:
            self.sigma = 0
        else:
            self.sigma = args[0]

class GammaParams():
    #def __init__(self, alpha, beta):
    def __init__(self, *args):
        if len(args)==0:
            self.alpha = 0
            self.beta = 0
        else:
            self.alpha = args[0]
            self.beta = args[1]

class StraussPrior():
    # def __init__(self,beta_l,beta_u,gamma_l,gamma_u,
    #              r_l,r_u) -> None:
    def __init__(self, *args):
        if len(args)==0:
            self.beta_u = 0
            self.gamma_l = 0
            self.gamma_u = 0
            self.r_l = 0
            self.r_u = 0
        else:
            self.beta_u = args[0]
            self.gamma_l = args[1]
            self.gamma_u = args[2]
            self.r_l = args[3]
            self.r_u = args[4]

class StraussInit():
    # def __init__(self,beta,gamma,R):
    def __init__(self, *args):
        if len(args)==0:
            self.beta = 0
            self.gamma = 0
            self.R = 0
        else:
            self.beta = args[0]
            self.gamma = args[1]
            self.R = args[2] 

class StraussParams():
    # def __init__(self,fixed_param, prior=None, init=None):
    def __init__(self, *args):
        if len(args) == 0:
            self.fixed_param = False
            self.prior = None
            self.init = None
        else:
            self.fixed_param = args[0]
            self.prior = args[1]
            self.init = args[2]
        self.has_init = False
        self.has_prior = False
        if self.prior:
            self.has_prior = True
        if self.init:
            self.has_init = True

class NrepParams():
    def __init__(self,u ,p):
        self.u = u
        self.p = p

class DPPParams():
    def __init__(self):
        self.nu = 0
        self.rho = 0
        self.N = 0
        self.s = 0
        self.fixed_param = False



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
        
        self.init_n_clus = 10
    
        