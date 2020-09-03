import numpy as np

from Solvers.PDE_Solver.PDEOperators import LinearPDEOperator
from Solvers.PDE_Solver.BoundariesConditions import BoundaryCondition
from Solvers.PDE_Solver.TerminalConditions import TerminalCondition
from Solvers.PDE_Solver.Types import SchemeType, BoundaryConditionType
from Solvers.PDE_Solver.Meshes import Mesh
from Solvers.PDE_Solver import Schemes
from typing import List


class FDSolver(object):

    def __init__(self,
                 mesh_t: Mesh,
                 mesh_x: Mesh,
                 operators: List[LinearPDEOperator],
                 scheme_type: SchemeType,
                 bc_type: BoundaryConditionType,
                 tc: TerminalCondition):

        self._operators = operators
        self._mesh_t = mesh_t
        self._mesh_x = mesh_x
        self._u_grid = np.zeros(shape=(mesh_x.get_size(), mesh_t.get_size()))

        if scheme_type == SchemeType.EXPLICIT:
            self._scheme = Schemes.ExplicitScheme(operators[0])
        elif scheme_type == SchemeType.IMPLICIT:
            self._scheme = Schemes.ImplicitScheme(operators[0])
        elif scheme_type == SchemeType.CRANK_NICOLSON:
            self._scheme = Schemes.ThetaScheme(operators, 0.5)
        else:
            raise ValueError("The operator type " + str(scheme_type))

        self._bc = BoundaryCondition(bc_type)
        self._tc = tc

    def get_solution_grid(self):
        return self._u_grid

    def solver(self):
        u_i = np.zeros(self._mesh_x.get_size())
        u_i_1 = np.zeros(self._mesh_x.get_size())

        u_i = self._tc.get_value(self._mesh_x)
        np.copyto(self._u_grid[:, -1], u_i)
        no_t_i = self._mesh_t.get_size()

        for i in range(no_t_i-2, -1, -1):
            self._scheme.step_solver(self._mesh_x,
                                     self._mesh_t.get_point(i),
                                     self._mesh_t.get_point(i+1),
                                     u_i_1,
                                     u_i)

            np.copyto(u_i, u_i_1)
            np.copyto(self._u_grid[:, i], u_i_1)








