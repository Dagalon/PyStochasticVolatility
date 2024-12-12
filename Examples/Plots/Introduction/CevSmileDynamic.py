import numpy as np
import matplotlib.pylab as plt

from functools import partial
from MC_Engines.MC_LocalVol import LocalVolFunctionals
from scipy.interpolate import interp1d
from Solvers.PDE_Solver import PDESolvers
from Solvers.PDE_Solver import PDEOperators
from Solvers.PDE_Solver.Meshes import BachelierUnderlyingMesh, uniform_mesh, Mesh
from Solvers.PDE_Solver.PDEs import NORMAL_LOCAL_VOL_PDE, PDE
from Solvers.PDE_Solver.Types import BoundaryConditionType, np_ndarray, SchemeType
from Solvers.PDE_Solver.TerminalConditions import TerminalCondition
from Solvers.PDE_Solver.BoundariesConditions import Zero_Laplacian_BC
from Tools.Bachelier import implied_volatility


def f_payoff(mesh: Mesh, k: float) -> np_ndarray:
    return np.maximum(mesh.get_points() - k, 0.0)


def get_vetor_iv_cev(nu: float, alpha: float, T: float, f0: float):
    # Smile curve with cev
    mesh_t = Mesh(uniform_mesh, 200, 0.0, T)
    mesh_x = BachelierUnderlyingMesh(alpha, f0, T, 0.99, uniform_mesh, 200)
    diffusion = partial(LocalVolFunctionals.cev_diffusion, beta=nu, sigma=alpha, shift=0.035)
    cev_pde = PDE.from_ipde_terms(NORMAL_LOCAL_VOL_PDE(diffusion))

    k_s = np.arange(f0 - 0.02, f0 + 0.025, 0.005)
    # k_s = [f0]
    tc_s = [TerminalCondition(partial(f_payoff, k=k_i)) for k_i in k_s]
    bc = Zero_Laplacian_BC()
    operator_exp = PDEOperators.LinearPDEOperator(mesh_x, cev_pde, bc)
    operator_impl = PDEOperators.LinearPDEOperator(mesh_x, cev_pde, bc)
    operators = [operator_exp, operator_impl]

    pde_price = []

    for tc_i in tc_s:
        pd_solver = PDESolvers.FDSolver(mesh_t,
                                        mesh_x,
                                        operators,
                                        SchemeType.CRANK_NICOLSON,
                                        BoundaryConditionType.ZERO_DIFFUSION,
                                        tc_i)

        pd_solver.solver()
        f = interp1d(mesh_x.get_points(), pd_solver._u_grid[:, 0], kind='linear', fill_value='extrapolate')
        pde_price.append(float(f(f0)))

    # Compute the iv
    no_elements = len(pde_price)

    # From hagan
    iv_fd = []
    z_s = []
    for i in range(0, no_elements):
        z_s.append(k_s[i] - f0)
        iv_fd.append(implied_volatility(pde_price[i], f0, k_s[i], T, 'c'))

    return z_s, iv_fd


T = 2

alpha = 0.075
nu = 0.6

# CEV parameter
styles = ['o', '*', 'x', '^', '.']
f0_s = [0.01, 0.015, 0.020, 0.03]
colors = ['green', 'black', 'blue', 'orange', 'olive']
no_f0 = len(f0_s)
plt.figure()
plt.title("CEV smile dynamic for gamma = %s and sigma = %s " % (nu, alpha))
for i in range(0, no_f0):
    z_s, iv_fd = get_vetor_iv_cev(nu, alpha, T, f0_s[i])
    plt.plot(z_s, iv_fd, label="S0=" + str(f0_s[i]), linestyle='--', marker=styles[i], color=colors[i])

plt.xlabel("k-f")
plt.legend()
plt.show()
