import numpy as np

from typing import Callable
from Types import np_ndarray
from abc import abstractmethod


class IPDETerms(object):

    @abstractmethod
    def source(self, t: float, x: np_ndarray):
        pass

    @abstractmethod
    def convection(self, t: float, x: np_ndarray):
        pass

    @abstractmethod
    def diffusion(self, t: float, x: np_ndarray):
        pass


class BS_forward_PDE(IPDETerms):
    def __init__(self, sigma):
        self._sigma = sigma

    def source(self, t: float, x: np_ndarray):
        return np.zeros(x.size)

    def convection(self, t: float, x: np_ndarray):
        return np.zeros(x.size)

    def diffusion(self, t: float, x: np_ndarray):
        return 0.5 * np.power(self._sigma * x, 2.0) * np.ones(x.size)


class LN_BS_PDE(IPDETerms):
    def __init__(self, r: float, q: float, sigma: float):
        self._r = r
        self._q = q
        self._sigma = sigma

    def source(self, t: float, x: np_ndarray):
        return -self._r * np.ones(x.size)

    def convection(self, t: float, x: np_ndarray):
        return (self._r - self._q - 0.5 * np.power(self._sigma, 2.0)) * np.ones(x.size)

    def diffusion(self, t: float, x: np_ndarray):
        return 0.5 * np.power(self._sigma, 2.0) * np.ones(x.size)


class CEV_forward_PDE(IPDETerms):
    def __init__(self, sigma: float, beta: float):
        self._sigma = sigma
        self._beta = beta

    def source(self, t: float, x: np_ndarray):
        return np.zeros(x.size)

    def convection(self, t: float, x: np_ndarray):
        return np.zeros(x.size)

    def diffusion(self, t: float, x: np_ndarray):
        return 0.5 * self._sigma * np.power(x, self._beta)


class PDE(object):
    def __init__(self,
                 source: Callable[[float, np_ndarray], np_ndarray],
                 convection: Callable[[float, np_ndarray], np_ndarray],
                 diffusion: Callable[[float, np_ndarray], np_ndarray]):
        self._source = source
        self._convection = convection
        self._diffusion = diffusion

    def source(self, t: float, x: np_ndarray) -> np_ndarray:
        return self._source(t, x)

    def convection(self, t: float, x: np_ndarray) -> np_ndarray:
        return self._convection(t, x)

    def diffusion(self, t: float, x: np_ndarray) -> np_ndarray:
        return self._diffusion(t, x)

    @classmethod
    def from_ipde_terms(cls, ipde: IPDETerms):
        return cls(ipde.source,
                   ipde.convection,
                   ipde.diffusion)

