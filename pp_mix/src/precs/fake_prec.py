from .precmat import PrecMat
from .base_prec import BaseMultiPrec
from typing import List
import numpy as np

class FakePrec(BaseMultiPrec):
    def __del__(self):
        pass

    def sample_prior(self) -> PrecMat:
        return PrecMat()

    def sample_given_data(self, data: List[np.ndarray], curr: PrecMat, mean: np.ndarray) -> PrecMat:
        return PrecMat()

    def mean(self) -> PrecMat:
        return PrecMat()

    def lpdf(self, val: PrecMat) -> float:
        return 0.0
