import numpy as np

from Solvers.PDE_Solver.Operators import Operator
from Solvers.PDE_Solver.Types import SchemeType
from Tools.Types import ndarray
from typing import List
from abc import abstractmethod
from Solvers.PDE_Solver.Meshes import Mesh
from functools import partial


class Scheme(object):
    def __init__(self, operator: List[Operator]):
        self._operator = operator

    @abstractmethod
    def step_solver(self, mesh: Mesh, t_i_1: float, t_i: float, u_i_1: ndarray, u_i: ndarray):
        pass

    def modify_operators(self, **kwargs):
        pass

    def set_parameters(self, **kwargs):
        pass


class ImplicitScheme(Scheme):
    def __init__(self, operator: Operator):
        Scheme.__init__(self, [operator])

    def step_solver(self, mesh: Mesh, t_i_1: float, t_i: float, u_i_1: ndarray, u_i: ndarray):
        delta_i_1_i = (t_i - t_i_1)
        self._operator[0].update_operator(t_i_1, mesh)
        self._operator[0].apply_boundary_condition(t_i_1=t_i_1, t_i=t_i, operator=self._operator[0],
                                                   scheme_type=SchemeType.IMPLICIT)

        np.copyto(u_i_1, self._operator[0].apply_implicit_operator(delta_i_1_i, u_i))
        self._operator[0].apply_boundary_condition_after_update(u_t=u_i_1, operator=self._operator[0])


class ExplicitScheme(Scheme):
    def __init__(self, operator: Operator):
        Scheme.__init__(self, [operator])

    def step_solver(self, mesh: Mesh, t_i_1: float, t_i: float, u_i_1: ndarray, u_i: ndarray):
        delta_i_1_i = (t_i - t_i_1)
        self._operator[0].update_operator(t_i, mesh)
        self._operator[0].apply_boundary_condition(t_i_1=t_i_1, t_i=t_i, operator=self._operator[0],
                                                   scheme_type=SchemeType.IMPLICIT)

        np.copyto(u_i_1, self._operator[0].apply_explicit_operator(delta_i_1_i, u_i))
        self._operator[0].apply_boundary_condition_after_update(u_t=u_i_1, operator=self._operator[0])


class ThetaScheme(Scheme):
    def __init__(self, operator: List[Operator], theta: float):
        Scheme.__init__(self, operator)
        self._theta = theta

    def get_theta(self):
        return self._theta

    def modify_operators(self, **kwargs):
        def f(x: ndarray, alpha: float):
            np.multiply(x, alpha, out=x)

        self._operator[0].modify_operator(partial(f, alpha=self._theta))
        self._operator[1].modify_operator(partial(f, alpha=1.0 - self._theta))

    def set_parameters(self, **kwargs):
        self._theta = kwargs['theta']

    def step_solver(self, mesh: Mesh, t_i_1: float, t_i: float, u_i_1: ndarray, u_i: ndarray):
        delta_i_1_i = (t_i - t_i_1)
        u_i_1_explicit = np.zeros(mesh.get_size())

        self._operator[0].update_operator(t_i, mesh)
        self._operator[1].update_operator(t_i, mesh)

        self.modify_operators()

        self._operator[0].apply_boundary_condition(t_i_1=t_i_1, t_i=t_i, operator=self._operator[0],
                                                   scheme_type=SchemeType.EXPLICIT)

        self._operator[1].apply_boundary_condition(t_i_1=t_i_1, t_i=t_i, operator=self._operator[1],
                                                   scheme_type=SchemeType.IMPLICIT)

        np.copyto(u_i_1_explicit, self._operator[0].apply_explicit_operator(delta_i_1_i, u_i))
        np.copyto(u_i_1, self._operator[1].apply_implicit_operator(delta_i_1_i, u_i_1_explicit))

        self._operator[0].apply_boundary_condition_after_update(u_t=u_i_1, operator=self._operator[0])
        self._operator[1].apply_boundary_condition_after_update(u_t=u_i_1, operator=self._operator[1])
