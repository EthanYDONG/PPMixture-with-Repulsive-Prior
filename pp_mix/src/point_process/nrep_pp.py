import numpy as np
from scipy.stats import truncnorm
from .basepp import BasePP
from ..proto import proto as proto
class NrepPP(BasePP): 
    def __init__(self, u, p):
        self.u = u
        self.p = p
        self.tau = None
        #self.dim = None  # Assuming dim is set somewhere in your code
        #self.c_star = 1.0
        #self.ranges = None  # Assuming ranges is set somewhere in your code

    def initialize(self):
        self.c_star = 1.0
        self.calibrate()

    def calibrate(self):
        shape = (1.0 * self.dim) / 2
        scale = 0.5
        q = truncnorm.ppf(1.0 - self.p, shape, scale)
        self.tau = q / (-np.log(1.0 - self.u))
        print("tau:", self.tau)

    def multi_trunc_normal_lpdf(self, x):
        out = 0.0
        for i in range(self.dim):
            out += truncnorm.logpdf(x[i], loc=0.0, scale=1.0,
                                     a=self.ranges[0, i], b=self.ranges[1, i])
        return out

    def dens(self, x, log=True):
        out = 0.0
        if ((x.size == 1 and self.dim == 1) or
                (x.shape[0] == 1 and self.dim > 1) or
                (x.shape[1] == 1 and self.dim > 1)):
            out = self.multi_trunc_normal_lpdf(x.flatten())
        else:
            for i in range(x.shape[0]):
                out += self.multi_trunc_normal_lpdf(x[i, :])

            pdist = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
            out += np.sum(np.log(1.0 - np.exp(-0.5 / self.tau * pdist)))

        if not log:
            out = np.exp(out)

        return out

    def papangelou(self, xi, x, log=True):
        out = 0.0
        if xi.shape[1] == 1:
            xi = xi.T

        for i in range(xi.shape[0]):
            out += self.multi_trunc_normal_lpdf(xi[i, :])

        dists = np.sum((xi[:, None, :] - x[None, :, :]) ** 2, axis=2)
        id_matrix = np.ones_like(dists)
        out += np.sum(np.log(1.0 - np.exp(-0.5 / self.tau * dists)))

        if not log:
            out = np.exp(out)

        return out

    def phi_star_rng(self):
        out = np.zeros(self.dim)
        for i in range(self.dim):
            out[i] = truncnorm.rvs(
                self.ranges[0, i], self.ranges[1, i], loc=0.0, scale=1.0)

        return out

    def phi_star_dens(self, xi, log=True):
        out = self.multi_trunc_normal_lpdf(xi)
        if not log:
            out = np.exp(out)
        return out

    def update_hypers(self, active, non_active):
        return

    def get_state_as_proto(self):
        # state = proto.NrepState()
        # state.u = self.u
        # state.p = self.p
        # state.tau = self.tau
        # return state
        return

    def estimate_mean_proposal_sigma(self):
        return 0.25

    def rejection_sampling_M(self, npoints):
        return 1.0


