import numpy as np
import time

from Solvers.PDE_Solver import PDESolvers
from Solvers.PDE_Solver import PDEOperators
from scipy.interpolate import interp1d
from Solvers.PDE_Solver.Meshes import uniform_mesh, Mesh, LnUnderlyingMesh
from Solvers.PDE_Solver.PDEs import LN_BS_PDE, PDE
from Solvers.PDE_Solver.Types import BoundaryConditionType, np_ndarray, SchemeType
from Solvers.PDE_Solver.TerminalConditions import TerminalCondition
from Solvers.PDE_Solver.BoundariesConditions import Zero_Laplacian_BC
from py_vollib.black_scholes_merton import black_scholes_merton

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
except ImportError:
    print('The import of matplotlib is not working.')

T = 1.0
mesh_t = Mesh(uniform_mesh, 10, 0.0, T)

r = 0.0
q = 0.0
sigma = 0.3

S0 = 100.0
# K = np.exp((r - q) * T) * S0 + 10
K = 90.0
log_K = np.log(K)

f = np.exp((r - q) * T) * S0
df = np.exp(-r * T)

start_time = time.time()
analytic_price = df * black_scholes_merton('c', f, K, T, 0.0, sigma, 0.0)
end_time = time.time()
print(end_time - start_time)
print(analytic_price)

mesh_x = LnUnderlyingMesh(r, q, sigma, S0, T, 0.9999999, uniform_mesh, 100)

bs_pde = PDE.from_ipde_terms(LN_BS_PDE(r, q, sigma))
bc = Zero_Laplacian_BC()

operator_exp = PDEOperators.LinearPDEOperator(mesh_x, bs_pde, bc)
operator_impl = PDEOperators.LinearPDEOperator(mesh_x, bs_pde, bc)
operators = [operator_exp, operator_impl]


def f_ln_payoff(mesh: Mesh) -> np_ndarray:
    return np.maximum(np.exp(mesh.get_points()) - K, 0.0)


tc = TerminalCondition(f_ln_payoff)

pd_solver = PDESolvers.FDSolver(mesh_t,
                                mesh_x,
                                operators,
                                SchemeType.CRANK_NICOLSON,
                                BoundaryConditionType.ZERO_DIFFUSION,
                                tc)
start_time = time.time()
pd_solver.solver()
end_time = time.time()
f = interp1d(mesh_x.get_points(), pd_solver._u_grid[:, 0], kind='linear', fill_value='extrapolate')
pde_price = float(f(np.log(S0)))
print(end_time - start_time)
print(pde_price)




