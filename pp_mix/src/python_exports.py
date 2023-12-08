#import pybind11
import numpy as np
import random
#from pybind11 import Eigen
from .factory import make_pp,make_jump,make_prec
from .conditional_mcmc import *
from .proto.proto import EigenVector,EigenMatrix
import datetime
now = datetime.datetime.now()

"""
# Mocking necessary classes and functions for illustration
class Params:
    def ParseFromString(self, s):
        pass

class BasePP:
    def set_ranges(self, ranges):
        pass

class BaseJump:
    pass

class BasePrec:
    pass

class UnivariateConditionalMCMC:
    def __init__(self, pp_mix, h, g, params):
        pass

    def initialize(self, datavec):
        pass

    def run_one(self):
        pass

    def get_state_as_proto(self, curr):
        pass

class UnivariateMixtureState:
    def SerializeToString(self, s):
        pass
"""
def run_pp_mix_univ(burnin, niter, thin, data, params, log_every):
    ranges = np.vstack([np.min(data, axis=0), np.max(data, axis=0)]) * 2
    print("ranges: \n", ranges)

    out = []
    pp_mix = make_pp(params)
    h = make_jump(params)
    g = make_prec(params)
    pp_mix.set_ranges(ranges)

    sampler = UnivariateConditionalMCMC(pp_mix, h, g, params)
    #datavec = data.flatten().tolist()
    sampler.initialize(data)

    for i in range(burnin):
        sampler.run_one()
        if (i + 1) % log_every == 0:
            print("Burnin, iter #", i + 1, " / ", burnin)

    for i in range(niter):
        sampler.run_one()
        if i % thin == 0:
            s = ""
            # curr = UnivariateMixtureState() #获取当前的state
            # sampler.get_state_as_proto(curr) 
            # curr.SerializeToString(s)
            # out.append(bytes(s, 'utf-8'))
            curr = sampler.get_state_as_proto()
            out.append(curr)
        if (i + 1) % log_every == 0:
            print("Running, iter #", i + 1, " / ", niter)

    return out

# TODO: Define other functions and classes similarly

# Mocked functions for illustration


# For the Pybind11 module definition, you can use `pybind11_module` from the `cppimport` library 
# or compile it as a separate C++ extension module using a C++ compiler.
def run_pp_mix_multi(burnin, niter, thin, data, params, log_every):
    random_seed = random.random()

    random.seed(42)

    ranges = np.vstack([np.min(data, axis=0), np.max(data, axis=0)]) * 2
    print("ranges: \n", ranges)

    out = []
    pp_mix = make_pp(params)
    h = make_jump(params)
    g = make_prec(params)
    pp_mix.set_ranges(ranges)

    sampler = MultivariateConditionalMCMC(pp_mix, h, g, params)
    #datavec = [row_vector for row_vector in data]
    sampler.initialize(data)

#####保存为字典########
    history = {'i': [], 'ma': [], 'mna': [], 'a_means': [], 'na_means': [], 'a_precs': [], 'na_precs': [], 'a_jumps': [], 'na_jumps': [], 'clus_alloc': [], 'u': [], 'beta': [], 'gamma': [], 'r': [],'ppstate':[]}

    for i in range(burnin):
        sampler.run_one()
        if (i + 1) % log_every == 0:
            print("Burnin, iter #", i + 1, " / ", burnin)
    
    for i in range(niter):
        sampler.run_one()
        if i % thin == 0:
            s = ""
            # curr = UnivariateMixtureState() #获取当前的state
            # sampler.get_state_as_proto(curr) 
            # curr.SerializeToString(s)
            # out.append(bytes(s, 'utf-8'))
            curr = None
            curr = sampler.get_state_as_proto()
            #print(curr.clus_alloc)
            out.append(curr)
        if (i + 1) % log_every == 0:
            print("Running, iter #", i + 1, " / ", niter)
            history['i'].append(i)
            history['ma'].append(curr.ma)
            history['mna'].append(curr.mna)
            history['a_means'].append(curr.a_means)
            history['na_means'].append(curr.na_means)
            history['a_precs'].append(curr.a_precs)
            history['na_precs'].append(curr.na_precs)
            history['a_jumps'].append(curr.a_jumps)
            history['na_jumps'].append(curr.na_jumps)
            history['clus_alloc'].append(curr.clus_alloc)
            history['u'].append(curr.u)
            history['ppstate'].append(curr.pp_state)
            #history['gamma'].append(curr.pp_state.gamma)
            #history['beta'].append(curr.pp_state.beta)
            #history['r'].append(curr.pp_state.r)
            # print(curr.clus_alloc)

    np.save('20231207_dpp_history1.npy', history)
    return out

def run_pp_mix_bernoulli(burnin, niter, thin, data, params, log_every):
    ranges = np.zeros((2, data.shape[1])) 

    ranges[0, :] = 0.0
    ranges[1, :] = 1.0

    out = []
    pp_mix = make_pp(params)
    h = make_jump(params)
    g = make_prec(params)
    pp_mix.set_ranges(ranges)

    sampler = BernoulliConditionalMCMC(pp_mix, h, g, params)
    datavec = [row_vector for row_vector in data]
    sampler.initialize(datavec)

    for i in range(burnin):
        sampler.run_one()
        if (i + 1) % log_every == 0:
            print("Burnin, iter #", i + 1, " / ", burnin)
    
    for i in range(niter):
        print("start running***\n")
        sampler.run_one()
        if i % thin == 0:
            s = ""
            # curr = UnivariateMixtureState() #获取当前的state
            # sampler.get_state_as_proto(curr) 
            # curr.SerializeToString(s)
            # out.append(bytes(s, 'utf-8'))
            curr = sampler.get_state_as_proto()
            print(curr.clus_alloc)
            print(i)
            out.append(curr)
        if (i + 1) % log_every == 0:
            print("Running, iter #", i + 1, " / ", niter)
    return out

def _run_pp_mix(burnin, niter, thin, data, params,bernoulli=False, log_every = 200):

    if (bernoulli):
        return run_pp_mix_bernoulli(burnin, niter, thin, data, params, log_every)
    elif data.shape[0] == 1 or data.shape[1] == 1:
        return run_pp_mix_univ(burnin, niter, thin, data, params, log_every)
    else:
        return run_pp_mix_multi(burnin, niter, thin, data, params, log_every)


def _sample_predictive_univ(chain):

    niter = len(chain)

    out = np.zeros(niter)

    for i in range(niter):
        state = out[i]

        a_means = np.array(state.a_means.data)
        a_precs = np.array(state.a_precs)
        na_means = np.array(state.na_means)
        na_precs = np.array(state.na_precs)
        a_jumps = np.array(state.a_jumps)
        na_jumps = np.array(state.na_jumps)

        probas = np.concatenate([a_jumps, na_jumps])
        probas /= probas.sum()

        k = np.random.choice(np.arange(len(probas)), p=probas)

        if k < len(state.ma):
            mu = a_means[k]
            sig = 1.0 / np.sqrt(a_precs[k])
        else:
            mu = na_means[k - state.ma]
            sig = 1.0 / np.sqrt(na_precs[k - state.ma])

        out[i] = np.random.normal(mu, sig)
    return out

def _sample_predictive_multi(chain, dim):

    niter = len(chain)
    out = np.zeros((niter, dim))
    for i in range(niter):

        state = chain[i]
        # state.a_jumps state.na_jumps 数据结构为 EigenVector
        a_jumps = np.array(state.a_jumps)
        na_jumps = np.array(state.na_jumps)
        probas = np.concatenate([a_jumps, na_jumps])
        probas /= probas.sum()

        k = np.random.choice(np.arange(len(probas)), p=probas)

        if k < len(state.a_means):
            mu = state.a_means[k]
            prec = state.a_precs[k]
        else:
            mu = state.na_means[k - len(state.a_means)]
            prec = state.na_precs()[k - len(state.a_means)]
        
        out[i] = np.random.multivariate_normal(mu, prec).T
    return out