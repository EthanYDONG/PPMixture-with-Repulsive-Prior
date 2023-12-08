import numpy as np
from scipy.stats import gamma, uniform

class BaseJump:
    def __init__(self):
        pass

    def sample_tilted(self, u):
        pass

    def sample_given_data(self, ndata, curr, u):
        pass

    def laplace(self, u):
        pass

class GammaJump(BaseJump):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_tilted(self, u):
        # s, v = None, None
        # while True:
        #     s = gamma.rvs(self.alpha, scale=1 / self.beta)
        #     v = uniform.rvs(0, 1)
        #     if np.log(v) < -s * u:
        #         return s
        return gamma.rvs(self.alpha, scale=1 / (self.beta + u))

    def sample_given_data(self, ndata, curr: float, u):
        out = None
        nh = ndata
        temp = curr
        prop = curr + uniform.rvs(-0.1, 0.1)
        # if (prop <=0):
        #     print('erro ',temp,prop)
        num = np.log(prop) * nh - u * prop + gamma.logpdf(prop, a=self.alpha, scale=1 / self.beta)
        den = np.log(curr) * nh - u * curr + gamma.logpdf(curr, a=self.alpha, scale=1 / self.beta)

        if np.log(uniform.rvs(0, 1)) < min(num - den,0):
            out = prop
        else:
            out = curr
        if (out <=0):
            print("out:",out)
        return out

    def laplace(self, u):
        return self.beta**self.alpha / (self.beta + u)**self.alpha
