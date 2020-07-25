import Tools
import numpy as np

from Operators import Operator, Laplacian, Gradient
from PDEs import PDE
from Types import BoundaryConditionType, np_ndarray
from Meshes import Mesh
from BoundariesConditions import BoundaryCondition


class LinearPDEOperator(Operator):

    def __init__(self, mesh: Mesh, pde: PDE, bc: BoundaryCondition):
        Operator.__init__(self, mesh, bc)
        self._pde = pde
        self._laplacian = Laplacian(mesh, BoundaryCondition(BoundaryConditionType.ZERO_LAPLACIAN))
        self._gradient = Gradient(mesh, BoundaryCondition(BoundaryConditionType.ZERO_GRADIENT))

    def apply_implicit_operator(self, delta: float, u_i: np_ndarray) -> np_ndarray:
        return Tools.tdr_system_solver((1.0 - delta * self.diagonal()),
                                       - delta * self.diagonal_lower(),
                                       - delta * self.diagonal_upper(),
                                       u_i)

    def apply_explicit_operator(self, delta: float, u_i: np_ndarray) -> np_ndarray:
        return Tools.apply_tdr((1.0 + delta * self.diagonal()),
                               delta * self.diagonal_lower(),
                               delta * self.diagonal_upper(),
                               u_i)

    def get_pde(self):
        return self._pde

    def apply_boundary_condition(self, **kwargs):
        self._bc.apply_boundary_condition(**kwargs)

    def apply_boundary_condition_after_update(self, **kwargs):
        self._bc.apply_boundary_condition_after_update(**kwargs)

    def update_operator(self, t: float, mesh: Mesh):
        diffusion = self._pde.diffusion(t, mesh.get_points())
        convection = self._pde.convection(t, mesh.get_points())
        source = self._pde.source(t, mesh.get_points())

        self._laplacian.update_operator(t, mesh)
        self._gradient.update_operator(t, mesh)

        Tools.update_diagonal(diffusion,
                              convection,
                              source,
                              self._gradient.diagonal(),
                              self._laplacian.diagonal(),
                              self.diagonal())

        Tools.update_diagonal_upper(diffusion,
                                    convection,
                                    self._gradient.diagonal_upper(),
                                    self._laplacian.diagonal_upper(),
                                    self.diagonal_upper())

        Tools.update_diagonal_lower(diffusion,
                                    convection,
                                    self._gradient.diagonal_lower(),
                                    self._laplacian.diagonal_lower(),
                                    self.diagonal_lower())

