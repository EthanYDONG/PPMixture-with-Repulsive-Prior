import numpy as np
import pandas as pd
from .point_process.strauss_pp import StraussPP
from .point_process.nrep_pp import NrepPP
from .point_process.determinantal_pp import DeterminantalPP
from .jump.gamma import  GammaJump
from .precs.gamma import GammaPrec
from .precs.fixed_prec import FixedPrec,FixedUnivPrec
from .precs.wishart import Wishart

def make_pp(params):

    if params.has_strauss:
        return make_strauss(params.strauss_param)
    elif params.has_nrep:
        return make_nrep(params.nrep_param)
    elif params.has_dpp:
        return make_dpp(params.dpp_param)

def make_strauss(strauss_param):
    strauss_pp = None
    if strauss_param.fixed_param:
        strauss_pp = StraussPP(strauss_param.beta,strauss_param.gamma,
                                strauss_param.r)
    elif strauss_param.has_init and strauss_param.has_prior:
        strauss_pp = StraussPP(strauss_param.prior,strauss_param.init.beta,
                                strauss_param.init.gamma,strauss_param.init.R)
    else:
        strauss_pp = StraussPP(strauss_param.prior)
    
    return strauss_pp

def make_nrep(nrep_param):
    nrep_pp = NrepPP(nrep_param.u, nrep_param.p)
    return nrep_pp

def make_dpp(dpp_param):
    dpp = DeterminantalPP(dpp_param.N, dpp_param.rho, dpp_param.nu, dpp_param.s,
                             dpp_param.fixed_param)
    return dpp

def make_jump(params):
    gamma_jump = None
    if (params.has_gamma_jump):
        gamma_jump = make_gamma_jump(params.gamma_jump_param)
    return gamma_jump

def make_gamma_jump(gamma_param):
    gamma_jump = GammaJump(gamma_param.alpha, gamma_param.beta)
    return gamma_jump

def make_prec(params):
    prec = None
    if (params.has_fixed_multi_prec):
        prec = make_fixed_prec(params.fixed_multi_prec)
    elif (params.has_wishart):
        prec = make_wishart(params.wishart_param)
    elif (params.has_fixed_univ_prec):
        prec = make_fixed_prec(params.fixed_univ_pre)
    elif (params.has_gamma_prec):
        prec = make_gamma_prec(params.gamma_prec)
    
    return prec

def make_fixed_prec(fixed_multi_prec_params):
  return  FixedPrec(fixed_multi_prec_params.dim, fixed_multi_prec_params.sigma)


def make_wishart(wishart_params):
   #wishart_params.PrintDebugString()
  sigma = 1.0
  if (wishart_params.sigma > 0) :
    sigma = wishart_params.sigma
  return Wishart(wishart_params.nu, wishart_params.dim, sigma)


def make_fixed_prec(fixed_univ_prec_params):
  return FixedUnivPrec(fixed_univ_prec_params.sigma)


def make_gamma_prec(gamma_param):
  return GammaPrec(gamma_param.alpha, gamma_param.beta)

