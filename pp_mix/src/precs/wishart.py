import numpy as np
import random
from .precmat import PrecMat
from .base_prec import BaseMultiPrec
from scipy.stats import wishart

class Wishart(BaseMultiPrec):
    def __init__(self, df, dim, sigma):
        self.df = df
        self.psi = np.eye(dim) * sigma
        self.inv_psi = np.eye(dim) / sigma

    def sample_prior(self):
        out = wishart.rvs(self.df, self.psi)
        #return out
        return PrecMat(out)

    def sample_given_data(self, data, curr, mean):
        data_mat = np.vstack(data)
        data_mat = data_mat - mean

        out = wishart.rvs(self.df + len(data), np.linalg.inv(self.inv_psi + data_mat.T @ data_mat))
        #return out
        return PrecMat(out)

    def mean(self):
        return self.psi * self.df
    
    def get_df(self):
        return self.df
    
    def get_psi(self):
        return self.psi
    
    def lpdf(self, val):
        precision_matrix = val.get_prec()
        return wishart(df = self.df, scale = self.psi).pdf(precision_matrix)

