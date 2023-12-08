import numpy as np
from numpy.linalg import inv, cholesky
from numpy.random import normal, uniform

class AdaptiveMetropolis:
    def __init__(self, dim):
        self.beta = 0.05
        self.sigma_n = None
        self.sum_sq_val = None
        self.sum_val = None
        self.mean_n = None
        self.dim = dim
        self.niter = 0

    def init(self):
        print("AdaptiveMetropolis::init()")
        self.sigma_n = np.eye(self.dim)
        self.mean_n = np.zeros(self.dim)
        self.sum_sq_val = np.zeros(self.dim)
        self.sum_val = np.zeros(self.dim)

    def update_sigma(self, curr):
        self.niter += 1

        self.mean_n = (self.mean_n * (self.niter - 1) + curr) / self.niter
        self.sum_val += curr
        self.sum_sq_val += np.outer(curr, curr)

        print("update")
        print("mean_n:", self.mean_n)
        print("sum_val:", self.sum_val)
        print("sum_sq_val:", self.sum_sq_val)

        if self.niter >= 2:
            self.sigma_n = (self.sum_sq_val - 2 * np.outer(self.mean_n, self.sum_val) + np.outer(self.mean_n, self.mean_n)) / self.niter
        else:
            self.sigma_n = np.eye(self.dim)

        print("sigma_n:", self.sigma_n)

    def propose(self, curr):
        print("propose")
        if uniform(0, 1) < self.beta:
            var = 0.1**2
        else:
            var = 2.38**2 * np.trace(self.sigma_n) / self.dim

        print("sigma_n:", self.sigma_n)
        print("var:", var)
        out = normal(curr, np.sqrt(var))
        self.update_sigma(out)

        return out


