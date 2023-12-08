from abc import ABC, abstractmethod
import numpy as np

class BasePrec(ABC):
    @abstractmethod
    def __init__(self):
        pass

class BaseUnivPrec(BasePrec):
    @abstractmethod
    def sample_prior(self):
        pass
    
    @abstractmethod
    def sample_given_data(self, data, curr, mean):
        pass

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def lpdf(self, val):
        pass

class BaseMultiPrec(BasePrec):
    @abstractmethod
    def sample_prior(self):
        pass
    
    @abstractmethod
    def sample_given_data(self, data, curr, mean):
        pass

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def lpdf(self, val):
        pass

"""
class FixedUnivPrec(BaseUnivPrec):
    def __init__(self, sigma):
        self.sigma = sigma

    def sample_prior(self):
        return self.sigma

    def sample_given_data(self, data, curr, mean):
        return self.sigma

    def mean(self):
        return self.sigma

    def lpdf(self, val):
        return 0.0

class FixedPrec(BaseMultiPrec):
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma

    def sample_prior(self):
        return self.sigma * np.eye(self.dim)

    def sample_given_data(self, data, curr, mean):
        return self.sample_prior()

    def mean(self):
        return self.sigma * np.eye(self.dim)

    def lpdf(self, val):
        return 0.0
"""