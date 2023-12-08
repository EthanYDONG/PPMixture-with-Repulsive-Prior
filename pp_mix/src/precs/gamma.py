import numpy as np
from .base_prec import BaseUnivPrec
from scipy.stats import gamma

class GammaPrec(BaseUnivPrec):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_prior(self):
        return np.random.gamma(self.alpha, 1 / self.beta)

    def sample_given_data(self, data, curr, mean):
        m = mean[0]
        x_min_mu = [(x - m) ** 2 for x in data]
        sum_squares = np.sum(x_min_mu)
        return np.random.gamma(self.alpha + 0.5 * len(data), 1 / (self.beta + 0.5 * sum_squares))

    def mean(self):
        return self.alpha / self.beta

    def lpdf(self, val):
        return (self.alpha - 1) * np.log(val) - self.beta * val - np.log(gamma(self.alpha)) + self.alpha * np.log(self.beta)

