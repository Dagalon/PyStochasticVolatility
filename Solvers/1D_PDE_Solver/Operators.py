import numpy as np
from Meshes import Mesh
from Types import np_ndarray
from BoundariesConditions import BoundaryCondition
from typing import Callable


class Operator(object):
    def __init__(self, mesh: Mesh, bc: BoundaryCondition):
        self._diagonal_upper = np.zeros(mesh.get_size() - 1)
        self._diagonal_lower = np.zeros(mesh.get_size() - 1)
        self._diagonal = np.zeros(mesh.get_size())
        self._mesh = mesh
        self._bc = bc

    def diagonal_upper(self):
        return self._diagonal_upper

    def diagonal_lower(self):
        return self._diagonal_lower

    def diagonal(self):
        return self._diagonal

    def get_mesh(self):
        return self._mesh

    def modify_operator(self, functional: Callable[[np_ndarray], None]):
        functional(self._diagonal)
        functional(self._diagonal_lower)
        functional(self._diagonal_upper)
        pass

    def apply_implicit_operator(self, delta: float, u_i: np_ndarray) -> np_ndarray:
        pass

    def apply_explicit_operator(self, delta: float, u_i: np_ndarray) -> np_ndarray:
        pass

    def update_operator(self, t: float, mesh: Mesh):
        pass

    def apply_boundary_condition(self, **kwargs):
        pass

    def apply_boundary_condition_after_update(self, **kwargs):
        pass


class Gradient(Operator):

    def __init__(self, mesh: Mesh, bc: BoundaryCondition):
        super().__init__(mesh, bc)

    def update_operator(self, t: float, mesh: Mesh):
        for i in range(1, self._mesh.get_size() - 1):
            delta_r = self._mesh.get_shift()[i]
            delta_l = self._mesh.get_shift()[i - 1]

            self._diagonal_lower[i-1] = - delta_r / (delta_l * (delta_r + delta_l))
            self._diagonal[i] = - (delta_l - delta_r) / (delta_r * delta_l)
            self._diagonal_upper[i] = delta_l / (delta_r * (delta_r + delta_l))


class Laplacian(Operator):
    def __init__(self, mesh: Mesh, bc: BoundaryCondition):
        super().__init__(mesh, bc)

    def update_operator(self, t: float, mesh: Mesh):
        for i in range(1, self._mesh.get_size() - 1):
            delta_r = self._mesh.get_shift()[i]
            delta_l = self._mesh.get_shift()[i - 1]
            self._diagonal_lower[i - 1] = 2.0 / (delta_l * (delta_r + delta_l))
            self._diagonal[i] = - 2.0 / (delta_l * delta_r)
            self._diagonal_upper[i] = 2.0 / (delta_r * (delta_r + delta_l))



