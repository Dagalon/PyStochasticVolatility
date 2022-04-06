import numpy as np

from typing import Callable
from Tools.Types import ndarray
from scipy.stats import norm


def uniform_mesh(no_points: int, T0: float, T1: float):
    return np.linspace(T0, T1, no_points)


def finite_volume_mesh(no_points: int, T0: float, T1: float):
    no_points_total = 2 * no_points - 1
    aux_nodes = np.linspace(T0, T1, no_points)
    mesh = np.empty(no_points_total)

    for i in range(0, no_points):
        mesh[2 * i] = aux_nodes[i]

    for i in range(0, no_points-1):
        mesh[2 * i + 1] = 0.5 * (mesh[2 * i] + mesh[2 * i + 2])

    return mesh


class Mesh(object):
    def __init__(self,
                 generator: Callable[[int, float, float], ndarray],
                 no_points: int,
                 T0=0.0,
                 T1=0.0,
                 ):
        self._generator = generator

        self._T0 = T0
        self._T1 = T1

        self._no_points = no_points

        if T0 != 0 or T1 != 0:
            self._points = generator(no_points, self._T0, self._T1)
            self._shift = np.diff(self._points, n=1)

    @property
    def left_boundary(self):
        return self._T0

    @property
    def right_boundary(self):
        return self._T1

    @property
    def nodes(self):
        return self._points

    def update(self, no_points: int):
        self._points = self._generator(no_points, self.left_boundary, self.right_boundary)

    def get_size(self):
        return len(self._points)

    def get_lower_bound(self):
        return self._T0

    def get_upper_bound(self):
        return self._T1

    def get_points(self):
        return self._points

    def get_point(self, i: int):
        return self._points[i]

    def get_shift(self):
        return self._shift

    def update_mesh(self, T0: float, T1: float, no_points: int, generator: Callable[[int, float, float], ndarray]):
        self._T0 = T0
        self._T1 = T1
        self._no_points = no_points
        self._points = generator(no_points, T0, T1)
        self._shift = np.diff(self._points)

    def get_bounds(self):
        return self._T0, self._T1


class LnUnderlyingMesh(Mesh):
    def __init__(self,
                 r: float,
                 q: float,
                 sigma: float,
                 s0: float,
                 T: float,
                 alpha: float,
                 generator: Callable[[int, float, float], ndarray],
                 no_points: int):
        Mesh.__init__(self, generator, no_points)

        self._r = r
        self._q = q
        self._sigma = sigma
        self._ln_s0 = np.log(s0)
        self._T = T
        self._alpha = alpha

        self._T0, self._T1 = self.get_bounds()

        self._points = generator(no_points, self._T0, self._T1)
        self._shift = np.diff(self._points, n=1)

    def get_bounds(self):
        alpha_q = - norm.isf(self._alpha)
        mu = (self._r - self._q - 0.5 * np.power(self._sigma, 2.0))
        s_u = self._ln_s0 + mu * self._T + alpha_q * self._sigma * np.sqrt(self._T)
        s_l = self._ln_s0 + mu * self._T - alpha_q * self._sigma * np.sqrt(self._T)
        return s_l, s_u


class BachelierUnderlyingMesh(Mesh):
    def __init__(self,
                 sigma: float,
                 x0: float,
                 T: float,
                 alpha: float,
                 generator: Callable[[int, float, float], ndarray],
                 no_points: int):
        Mesh.__init__(self, generator, no_points)

        self._sigma = sigma
        self._x0 = x0
        self._T = T
        self._alpha = alpha

        self._T0, self._T1 = self.get_bounds()

        self._points = generator(no_points, self._T0, self._T1)
        self._shift = np.diff(self._points, n=1)

    def get_bounds(self):
        alpha_q = - norm.isf(self._alpha)
        s_u = self._x0 + alpha_q * self._sigma * np.sqrt(self._T)
        s_l = self._x0 - alpha_q * self._sigma * np.sqrt(self._T)
        return s_l, s_u
