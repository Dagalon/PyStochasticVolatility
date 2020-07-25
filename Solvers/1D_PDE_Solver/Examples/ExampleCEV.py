import numpy as np
import PDESolvers
import PDEOperators

from Types import ndarray
from Meshes import uniform_mesh, Mesh
from PDEs import CEV_forward_PDE, PDE
from TerminalConditions import TerminalCondition


def delta_terminal_condition(x: ndarray, x0: float, h: float):
    return np.exp(-0.5 * np.power(x - x0, 2.0) / h) / np.sqrt(2.0 * np.pi * h)


T = 1.0
mesh_t = Mesh(uniform_mesh, 50, 0.0, T)
mesh_x = Mesh(uniform_mesh, 100, 0.0, 5.0)

sigma = 0.3
beta = 0.2
cev_forward_pde = PDE.from_ipde_terms(CEV_forward_PDE(sigma, beta))
bc = PDEOperators.ZeroLaplacianBC()

operator_exp = PDEOperators.LinearPDEOperator(mesh_x, cev_forward_pde, bc)
operator_impl = PDEOperators.LinearPDEOperator(mesh_x, cev_forward_pde, bc)
operators = [operator_exp, operator_impl]

tc = TerminalCondition(delta_terminal_condition)
