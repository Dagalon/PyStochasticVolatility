import numpy as np

from typing import Callable
from Tools.Types import ndarray
from abc import abstractmethod


class IPDETerms(object):

    @abstractmethod
    def source(self, t: float, x: ndarray):
        pass

    @abstractmethod
    def convection(self, t: float, x: ndarray, v: ndarray):
        pass

    @abstractmethod
    def diffusion(self, t: float, x: ndarray):
        pass


class BS_forward_PDE(IPDETerms):
    def __init__(self, sigma):
        self._sigma = sigma

    def source(self, t: float, x: ndarray):
        return np.zeros(x.size)

    def convection(self, t: float, x: ndarray, v: ndarray):
        return np.zeros(x.size)

    def diffusion(self, t: float, x: ndarray):
        return 0.5 * np.power(self._sigma * x, 2.0) * np.ones(x.size)


class LN_BS_PDE(IPDETerms):
    def __init__(self, r: float, q: float, sigma: float):
        self._r = r
        self._q = q
        self._sigma = sigma

    def source(self, t: float, x: ndarray):
        return -self._r * np.ones(x.size)

    def convection(self, t: float, x: ndarray, v: ndarray):
        return (self._r - self._q - 0.5 * np.power(self._sigma, 2.0)) * np.ones(x.size)

    def diffusion(self, t: float, x: ndarray):
        return 0.5 * np.power(self._sigma, 2.0) * np.ones(x.size)


class LN_FORWARD_LOCAL_VOL_PDE(IPDETerms):
    def __init__(self, f_local_vol: Callable[[float, ndarray], ndarray]):
        self._sigma = f_local_vol

    def source(self, t: float, x: ndarray):
        return np.zeros(x.size)

    def diffusion(self, t: float, x: ndarray):
        return 0.5 * np.power(self._sigma(t, x), 2.0)

    def convection(self, t: float, x: ndarray, v: ndarray):
        return - v


class NORMAL_LOCAL_VOL_PDE(IPDETerms):
    def __init__(self, f_local_vol: Callable[[float, ndarray], ndarray]):
        self._sigma = f_local_vol

    def source(self, t: float, x: ndarray):
        return np.zeros(x.size)

    def diffusion(self, t: float, x: ndarray):
        return  0.5 * np.power(self._sigma(t, x), 2.0)

    def convection(self, t: float, x: ndarray, v: ndarray):
        return np.zeros(x.size)


class PDE(object):
    def __init__(self,
                 source: Callable[[float, ndarray], ndarray],
                 convection: Callable[[float, ndarray, ndarray], ndarray],
                 diffusion: Callable[[float, ndarray], ndarray]):
        self._source = source
        self._convection = convection
        self._diffusion = diffusion

    def source(self, t: float, x: ndarray) -> ndarray:
        return self._source(t, x)

    def convection(self, t: float, x: ndarray, v: ndarray) -> ndarray:
        return self._convection(t, x, v)

    def diffusion(self, t: float, x: ndarray) -> ndarray:
        return self._diffusion(t, x)

    @classmethod
    def from_ipde_terms(cls, ipde: IPDETerms):
        return cls(ipde.source,
                   ipde.convection,
                   ipde.diffusion)

