import numpy as np
import matplotlib.pylab as plt

from functools import partial
from MC_Engines.MC_LocalVolEngine import LocalVolFunctionals, LocalVolEngine
from scipy.interpolate import interp1d
from Solvers.PDE_Solver import PDESolvers
from Solvers.PDE_Solver import PDEOperators
from Solvers.PDE_Solver.Meshes import uniform_mesh, Mesh, LnUnderlyingMesh
from Solvers.PDE_Solver.PDEs import LN_FORWARD_LOCAL_VOL_PDE, PDE
from Solvers.PDE_Solver.Types import BoundaryConditionType, np_ndarray, SchemeType
from Solvers.PDE_Solver.TerminalConditions import TerminalCondition
from Solvers.PDE_Solver.BoundariesConditions import Zero_Laplacian_BC
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from AnalyticEngines.LocalVolatility.Hagan import ExpansionLocVol


def f_ln_payoff(mesh: Mesh, k: float) -> np_ndarray:
    return np.maximum(np.exp(mesh.get_points()) - k, 0.0)


def get_vetor_iv_cev(nu: float, alpha: float, T: float, f0: float):
    # Smile curve with cev
    mesh_t = Mesh(uniform_mesh, 100, 0.0, T)
    mesh_x = LnUnderlyingMesh(0.0, 0.0, nu, f0, T, 0.999, uniform_mesh, 200)
    log_diffusion = partial(LocalVolFunctionals.log_cev_diffusion, beta=alpha - 1, sigma=nu)
    cev_pde = PDE.from_ipde_terms(LN_FORWARD_LOCAL_VOL_PDE(log_diffusion))

    k_s = np.arange(f0 - 3.0, f0 + 3.0, 0.5)
    # k_s = [f0]
    tc_s = [TerminalCondition(partial(f_ln_payoff, k=k_i)) for k_i in k_s]
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
        pde_price.append(float(f(np.log(f0))))

    # Compute the iv
    no_elements = len(pde_price)

    # From hagan
    iv_fd = []
    z_s = []
    for i in range(0, no_elements):
        z_s.append(np.log(k_s[i] / f0))
        iv_fd.append(implied_volatility(pde_price[i], f0, k_s[i], T, 0.0, 0.0, 'c'))

    return z_s, iv_fd


T = 1

alpha = 0.3
nu = 0.4

# CEV parameter
styles = ['o', '*', 'x', '^', '.']
f0_s = [10.0, 10.25, 10.5, 10.75, 11.0]
no_f0 = len(f0_s)
plt.figure()
plt.title("CEV smile dynamic for gamma = %s and sigma = %s " % (nu, alpha))
for i in range(0, no_f0):
    z_s, iv_fd = get_vetor_iv_cev(nu, alpha, T, f0_s[i])
    plt.plot(z_s, iv_fd, label="S0=" + str(f0_s[i]), linestyle='--', marker=styles[i], color='black')

plt.xlabel("ln(k/f)")
plt.legend()
plt.show()
