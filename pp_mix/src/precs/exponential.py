import numpy as np
import random
from .precmat import PrecMat
from .base_prec import BaseMultiPrec
from scipy.stats import expon


class Expon(BaseMultiPrec):
    def __init__(self, scale, C, D):
        self.scale = scale  # scale = 1/E(x)
        self.C = C
        self.D = D
    def sample_prior(self):
        out = np.random.exponential(self.scale, size=(self.C, self.D, self.C))
        return out
    
    def sample_given_data(self, data):
        pass
# D M=kernel K D 
    def mean(self):
        out = np.full(shape=(self.C, self.D, self.C), fill_value = 1/self.scale)
        return out
    
    def lpdf(self, val):
        # 初始化一个与矩阵相同形状的数组来存储每个元素的PDF值
        pdf_values = np.empty_like(val, dtype=float)

        # 计算每个元素的PDF值
        for i in range(val.shape[0]):
            for j in range(val.shape[1]):
                for k in range(val.shape[2]):
                    pdf_values[i, j, k] = expon.pdf(val[i, j, k], scale=self.scale)  # 假设每个元素的指数分布参数为2.0

        # 计算整个矩阵的PDF，将每个元素的PDF相乘
        matrix_pdf = np.prod(pdf_values)

        return matrix_pdf