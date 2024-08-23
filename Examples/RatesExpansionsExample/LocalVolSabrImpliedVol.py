import numpy as np
import matplotlib.pylab as plt

from scipy.interpolate import interp1d
from MC_Engines.MC_LocalVol import LocalVolFunctionals
from Solvers.PDE_Solver import PDESolvers
from Solvers.PDE_Solver import PDEOperators
from Solvers.PDE_Solver.Meshes import uniform_mesh, Mesh, BachelierUnderlyingMesh
from Solvers.PDE_Solver.PDEs import NORMAL_LOCAL_VOL_PDE, PDE
from Solvers.PDE_Solver.TerminalConditions import TerminalCondition
from Solvers.PDE_Solver.Types import BoundaryConditionType, np_ndarray, SchemeType
from Solvers.PDE_Solver.BoundariesConditions import Zero_Laplacian_BC
from VolatilitySurface.Tools import SABRTools
from AnalyticEngines.BetaZeroSabr import ExpansionTools
from functools import partial
from Tools.Bachelier import implied_volatility


def call(mesh: Mesh, k: float) -> np_ndarray:
    return np.maximum(mesh.get_points() - k, 0.0)


# option info
f0 = 0.01
t = 1.0
spreads = [-200.0, -175.0, -150.0, -100.0, -75.0, -50.0, -25.0, -10.0, 0.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 175.0, 200.0]
strikes = [f0 + si / 10000.0 for si in spreads]

# sabr parameters
alpha = 0.007
nu = 0.6
rho = -0.6
parameters = [alpha, nu, rho]

# meshes
mesh_t = Mesh(uniform_mesh, 100, 0.0, t)
mesh_x = BachelierUnderlyingMesh(alpha, f0, t, 0.99999, uniform_mesh, 400)

# local vol info
sabr_loc_vol = partial(LocalVolFunctionals.local_vol_normal_sabr, x0=f0, alpha=alpha, rho=rho, nu=nu)

# pde
pde = PDE.from_ipde_terms(NORMAL_LOCAL_VOL_PDE(sabr_loc_vol))

tc_s = [TerminalCondition(partial(call, k=k_i)) for k_i in strikes]

bc = Zero_Laplacian_BC()
operator_exp = PDEOperators.LinearPDEOperator(mesh_x, pde, bc)
operator_impl = PDEOperators.LinearPDEOperator(mesh_x, pde, bc)
operators = [operator_exp, operator_impl]


pde_iv = []
hagan_iv = []
watanabe_iv_lv = []
watanabe_iv_sv = []

for i in range(0, len(strikes)):
    pd_solver = PDESolvers.FDSolver(mesh_t,
                                    mesh_x,
                                    operators,
                                    SchemeType.CRANK_NICOLSON,
                                    BoundaryConditionType.ZERO_DIFFUSION,
                                    tc_s[i])

    pd_solver.solver()
    f = interp1d(mesh_x.get_points(), pd_solver._u_grid[:, 0], kind='linear', fill_value='extrapolate')
    price = float(f(f0))
    pde_iv.append(implied_volatility(price, f0, strikes[i], t, 'c'))

    #hagan price
    hagan_iv.append(SABRTools.sabr_normal_jit(f0, strikes[i], alpha, rho, nu, t))

    #watanabe price
    price_sv = ExpansionTools.get_option_normal_sabr_watanabe_expansion(f0, strikes[i], t, alpha, nu, rho, 'c')
    price_lv = ExpansionTools.get_option_normal_sabr_loc_vol_expansion(f0, strikes[i], t, alpha, nu, rho, 'c')
    iv = implied_volatility(price_lv, f0, strikes[i], t, 'c')
    watanabe_iv_lv.append(ExpansionTools.get_iv_normal_lv_sabr_watanabe_expansion(f0, strikes[i], t, alpha, nu, rho))
    watanabe_iv_sv.append(ExpansionTools.get_iv_normal_sabr_watanabe_expansion(f0, strikes[i], t, alpha, nu, rho))

plt.plot(strikes, hagan_iv, label='Hagan IV', linestyle='dotted')
# plt.plot(strikes, watanabe_iv_sv, label='Watanabe IV SV', linestyle='dashed')
plt.plot(strikes, watanabe_iv_lv, label='Watanabe IV LV', linestyle='-')
plt.plot(strikes, pde_iv, label='PDE IV', linestyle=':')

plt.legend()
plt.show()



