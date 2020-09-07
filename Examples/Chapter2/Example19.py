import numpy as np
import matplotlib.pylab as plt

from functools import partial
from MC_Engines.MC_LocalVolEngine import LocalVolFunctionals,LocalVolEngine
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

f0 = 10.0
T = 1.0


def f_ln_payoff(mesh: Mesh, k: float) -> np_ndarray:
    return np.maximum(np.exp(mesh.get_points()) - k, 0.0)


def get_vetor_iv_cev(nu: float, alpha: float, T: float):
    # Smile curve with cev
    mesh_t = Mesh(uniform_mesh, 100, 0.0, T)
    mesh_x = LnUnderlyingMesh(0.0, 0.0, nu, f0, T, 0.999, uniform_mesh, 200)
    log_diffusion = partial(LocalVolFunctionals.log_cev_diffusion, beta=alpha - 1, sigma=nu)
    cev_pde = PDE.from_ipde_terms(LN_FORWARD_LOCAL_VOL_PDE(log_diffusion))

    k_s = np.arange(f0 - 4.0, f0 + 4.0, 0.5)
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


    # Hagan approximation
    expansion_hagan = ExpansionLocVol.hagan_loc_vol(lambda t: nu,
                                                    lambda x: np.power(x, alpha),
                                                    lambda x: alpha * np.power(x, alpha - 1.0),
                                                    lambda x: alpha * (alpha - 1.0) * np.power(x, alpha - 2.0))
    # Compute the iv
    no_elements = len(pde_price)

    # From hagan
    iv_hagan = []
    iv_fd = []
    z_s = []

    for i in range(0, no_elements):
        z_s.append(np.log(k_s[i] / f0))
        iv_fd.append(implied_volatility(pde_price[i], f0, k_s[i], T, 0.0, 0.0, 'c'))

    for i in range(0, no_elements):
        iv_hagan.append(expansion_hagan.get_implied_vol(T, f0, k_s[i]))

    return z_s, iv_fd, iv_hagan


# CEV parameter
alpha_lower_s = [0.1, 0.3, 0.5, 0.7]
alpha_upper_s = [1.1, 1.2, 1.3, 1.4]
nu = 0.4
fig, axs = plt.subplots(2, 1)

styles = ['o', '*', 'x', '+']

for i in range(0, 4):
    z_s, iv_fd, iv_hagan = get_vetor_iv_cev(nu, alpha_lower_s[i], T)
    axs[0].plot(z_s, iv_fd, label="gamma = " + str(alpha_lower_s[i]), linestyle='--', marker=styles[i], color="black")

axs[0].legend(loc="upper left")

for i in range(0, 4):
    z_s, iv_fd, iv_hagan = get_vetor_iv_cev(nu, alpha_upper_s[i], T)
    axs[1].plot(z_s, iv_fd, label="gamma = " + str(alpha_upper_s[i]), linestyle='--', marker=styles[i], color="black")

axs[1].legend(loc="upper left")
plt.show()
