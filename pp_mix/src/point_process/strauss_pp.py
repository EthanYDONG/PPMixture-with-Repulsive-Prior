import numpy as np
from scipy.stats import uniform, truncnorm
from scipy.special import logsumexp
from .basepp import BasePP
from ..adaptive_metropolis import AdaptiveMetropolis
from ..utils import pairwise_dist_sq
from .perfect_sampler import PerfectSampler
from ..proto import proto as proto

class StraussPP(BasePP):
    # def __init__(self, dim, ranges, priors=None, beta=None, gamma=None, R=None):
    #     super().__init__(dim, ranges)
    #     self.beta = beta
    #     self.gamma = gamma
    #     self.R = R
    #     self.fixed_params = False
    #     self.priors = priors
    #     self.am_beta = None
    #     self.sqrt_chisq_quantile = None
    #     self.vol_range = None

    def __init__(self,*args):
        super().__init__()
        if len(args) == 1:
            priors = args[0]
            self.priors = priors
            self.beta = (priors.beta_u + priors.beta_l) / 2.0
            self.gamma = (priors.gamma_u + priors.gamma_l) / 2.0
            self.R = (priors.r_u + priors.r_l) / 2.0
            self.fixed_params = False
        elif len(args) == 3:
            self.beta = args[0]
            self.gamma = args[1]
            self.R = args[2]
            self.fixed_params = True
        elif len(args) == 4:
            self.priors = args[0]
            self.beta = args[1]
            self.gamma = args[2]
            self.R = args[3]
            self.fixed_params = False
        
        self.am_beta = None
        self.sqrt_chisq_quantile = None

    def initialize(self):
        # if self.priors:
        #     self.beta = (self.priors.beta_u() + self.priors.beta_l()) / 2.0
        #     self.gamma = (self.priors.gamma_u() + self.priors.gamma_l()) / 2.0
        #     self.R = (self.priors.r_u() + self.priors.r_l()) / 2.0
        #     self.fixed_params = False
        # self.am_beta = AdaptiveMetropolis(1)
        # self.am_beta.init()
        # self.c_star = self.beta * self.vol_range
        # print(f"initialize\n beta: {self.beta}, vol_range: {self.vol_range}, c_star: {self.c_star}")
        # #self.sqrt_chisq_quantile = np.sqrt(quantile(chi2(dim), 0.1))
        # chisq = np.random.chisquare(df = self.dim)
        # self.sqrt_chisq_quantile = np.sqrt(np.quantile(chisq, 0.1))

        self.am_beta = AdaptiveMetropolis(1)
        self.am_beta.init()
        self.c_star = self.beta*self.vol_range
        print(f"initialize\n beta: {self.beta}, vol_range: {self.vol_range}, c_star: {self.c_star}")
        #self.sqrt_chisq_quantile = np.sqrt(quantile(chi2(dim), 0.1))
        chisq = np.random.chisquare(df = self.dim)
        self.sqrt_chisq_quantile = np.sqrt(np.quantile(chisq, 0.1))
    def dens(self, x, log=True):
        out = 0.0
        if (x.size == 1 and self.dim == 1) or (x.shape[0] == 1 and self.dim > 1) or (x.shape[1] == 1 and self.dim > 1):
            out = np.log(self.beta)
        else:
            npoints = x.shape[0]
            out = np.log(self.beta) * npoints
            pdist = pairwise_dist_sq(x)
            pdist[np.diag_indices(npoints)] = np.ones(npoints) * self.R * self.R * 10
            out += np.log(self.gamma) * (pdist < self.R * self.R).sum() / 2
        if not log:
            out = np.exp(out)
        return out

    def dens_from_pdist(self, dists, beta_, gamma_, R_, log=True):
        out = 0.0
        if dists.shape[0] == 1:
            out = beta_
        else:
            npoints = dists.shape[0]
            out = np.log(beta_) * npoints
            dists[np.diag_indices(npoints)] = np.ones(npoints) * R_ * R_ * 10.0
            out += np.log(gamma_) * (dists < R_ * R_).sum() / 2
        if not log:
            out = np.exp(out)
        return out

    def papangelou(self, xi, x, log=True):
        out = 0.0
        if xi.shape[1] != self.dim:
            xi = xi.T

        for i in range(xi.shape[0]):
            for j in range(self.dim):
                if (xi[i, j] < self.ranges[0, j]) or (xi[i, j] > self.ranges[1, j]):
                    print("SOMETHING WENT WRONG !!")
                    if log:
                        out = -np.inf
                    else:
                        out = 0.0
                    return out

        dists = pairwise_dist_sq(xi, x)
        out = np.log(self.gamma) * (dists < self.R * self.R).sum()
        if not log:
            out = np.exp(out)
        return out

    def papangelou_point(self, xi, x, log=True):
        exp = sum(1.0 * (np.sum((p.coords - xi.coords)**2) < self.R * self.R) for p in x)
        out = np.log(self.gamma) * exp
        if not log:
            out = np.exp(out)
        return out

    def phi_star_rng(self):
        out = np.zeros(self.dim)
        for i in range(self.dim):
            out[i] = uniform.rvs(self.ranges[0, i], self.ranges[1, i])
        return out.reshape(-1,self.dim)

    def phi_star_dens(self, xi, log=True):
        out = self.beta
        if log:
            out = np.log(out)
        return out

    def estimate_mean_proposal_sigma(self):
        return self.R / self.sqrt_chisq_quantile

    def update_hypers(self, active, non_active):
        if self.fixed_params:
            return

        Ma = active.shape[0]
        Mna = non_active.shape[0]
        dim = active.shape[1]
        all_points = np.vstack((active, non_active))

        dists = pairwise_dist_sq(all_points)
        # aux_var, aux_dists = simulate(self.vol_range, self.beta, self.gamma, self.R)
        # aux_lik_rate = self.dens_from_pdist(aux_dists, self.beta, self.gamma, self.R) - self.dens_from_pdist(aux_dists, self.beta, self.gamma, self.R)
        # 有待修复
        arate = 0.0

        # UPDATE BETA
        upper = self.priors.beta_u
        lower = self.priors.beta_l
        scale = (upper - lower) / 15

        prop = truncnorm.rvs((lower - self.beta) / scale, (upper - self.beta) / scale, loc=self.beta, scale=scale)

        proprate = truncnorm.logpdf(prop, (lower - self.beta) / scale, (upper - self.beta) / scale, loc=self.beta, scale=scale) - \
                   truncnorm.logpdf(self.beta, (lower - prop) / scale, (upper - prop) / scale, loc=prop, scale=scale)

        arate += proprate

        likrate = self.dens_from_pdist(dists, prop, self.gamma, self.R) - self.dens_from_pdist(dists, self.beta, self.gamma, self.R)
        arate += likrate

        # trick just for perfect simulation
        old_beta = self.beta
        self.beta = prop
        sampler = PerfectSampler(self)
        #aux_var, aux_dists = sampler.simulate(self.vol_range, self.beta, self.gamma, self.R)
        aux_var = sampler.simulate()
        self.beta = old_beta

        aux_dists = pairwise_dist_sq(aux_var)

        aux_lik_rate = self.dens_from_pdist(aux_dists, self.beta, self.gamma, self.R) - \
                       self.dens_from_pdist(aux_dists, prop, self.gamma, self.R)
        arate += aux_lik_rate

        if np.log(uniform.rvs(0, 1)) < arate:
            self.beta = prop
            self.c_star = self.beta * self.vol_range

    def get_state_as_proto(self):
        state = proto.StraussState()
        state.beta = self.beta
        state.gamma = self.gamma
        state.r = self.R
        # state.birth_prob = self.birth_prob
        # state.birth_arate = self.birth_arate
        return state
