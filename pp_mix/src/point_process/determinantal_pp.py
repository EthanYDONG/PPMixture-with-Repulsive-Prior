import numpy as np
from scipy.spatial.distance import cdist
from itertools import product
from scipy.linalg import sqrtm
from .basepp import BasePP

class DeterminantalPP(BasePP):
    def __init__(self, N, rho, nu, s, fixed_params=True):
        self.N = N
        self.rho = rho
        self.nu = nu
        self.s = s
        self.fixed_params = fixed_params
        if s == 0:
            self.s = 0.5

    def initialize(self):
        self.eigen_decomposition()
        self.c_star = np.sum(self.phi_tildes)
        self.A = np.zeros((self.dim, self.dim))
        self.b = np.zeros(self.dim)
        for i in range(self.dim):
            self.A[i, i] = 1.0 / (self.ranges[1, i] - self.ranges[0, i])
            self.b[i] = -self.A[i, i] * (self.ranges[1, i] + self.ranges[0, i]) / 2.0

    def dens(self, x, log=True):
        n = x.shape[0]

        if n == 1 and self.dim == 1:
            out = -1.0 * n * np.log(self.vol_range) + self.vol_range + np.log(np.sum(self.phi_tildes))
        else:
            check_range = np.all((x >= self.ranges[0]) & (x <= self.ranges[1]))
            if check_range:
                out = -1.0 * n * np.log(self.vol_range) + self.vol_range
                xtrans = np.dot(x, self.A.T) + self.b
                out += self.log_det_Ctilde(xtrans) - self.Ds
            else:
                out = -np.inf

        if not log:
            out = np.exp(out)

        return out

    def papangelou(self, xi, x, log=True):
        if xi.shape[1] != self.dim:
            xi = xi.T

        all_points = np.vstack([xi, x])
        out = self.dens(all_points) - self.dens(x)

        if not log:
            out = np.exp(out)

        return out

    def phi_star_rng(self):
        out = np.zeros(self.dim)
        for i in range(self.dim):
            out[i] = np.random.uniform(self.ranges[0, i], self.ranges[1, i])
        return out.reshape(-1, self.dim)

    def phi_star_dens(self, xi, log=True):
        out = np.sum(self.phi_tildes)
        if log:
            out = np.log(out)

        return out

    def update_hypers(self, active, non_active):
        if not self.fixed_params:
            self.eigen_decomposition()

    def get_Ctilde(self, x):
        Ctilde = np.zeros((x.shape[0], x.shape[0]))

        for l in range(x.shape[0]):
            for m in range(l + 1):
                aux = 0.0
                diff = x[l, :] - x[m, :]
                for kind in range(self.Kappas.shape[0]):
                    dotprod = np.dot(self.Kappas[kind, :], diff)
                    aux += self.phi_tildes[kind] * np.cos(2.0 * np.pi * dotprod)
                Ctilde[l, m] = aux
                Ctilde[m, l] = aux

        return Ctilde

    def log_det_Ctilde(self, x):
        Ctilde = self.get_Ctilde(x)
        return 2.0 * np.log(np.linalg.cholesky(Ctilde).diagonal().prod())

    def eigen_decomposition(self):
        k = np.arange(-self.N, self.N + 1)
        kappas = list(product(k, repeat=self.dim))

        self.Kappas = np.array(kappas)
        self.phis = np.zeros(len(kappas))
        self.phi_tildes = np.zeros(len(kappas))
        self.Ds = 0.0

        dim_ = 1.0 * self.dim
        log_alpha_max = 1.0 / dim_ * (
            np.log(np.math.gamma(dim_ / self.nu + 1))
            - np.log(self.rho)
            - np.log(np.math.gamma(dim_ / 2 + 1))
            - 0.5 * np.log(np.sqrt(np.pi)) #//using stan::math::LOG_SQRT_PI
        )
        alpha_max = np.exp(log_alpha_max)

        for i in range(len(kappas)):
            self.phis[i] = (
                np.power(self.s, self.dim)
                * np.exp(-(self.s * alpha_max * np.linalg.norm(self.Kappas[i, :])))
            )
            self.phi_tildes[i] = self.phis[i] / (1 - self.phis[i])
            self.Ds += np.log(1 + self.phi_tildes[i])
