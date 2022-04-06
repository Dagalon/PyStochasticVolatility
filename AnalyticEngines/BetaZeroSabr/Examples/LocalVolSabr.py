import numpy as np
import matplotlib.pylab as plt

from scipy.interpolate import interp1d
from MC_Engines.MC_LocalVol import LocalVolFunctionals
from Solvers.PDE_Solver import PDESolvers
from Solvers.PDE_Solver import PDEOperators
from Solvers.PDE_Solver.Meshes import uniform_mesh, Mesh, BachelierUnderlyingMesh
from Solvers.PDE_Solver.PDEs import NORMAL_LOCAL_VOL_PDE, PDE
from Solvers.PDE_Solver.Types import BoundaryConditionType, np_ndarray, SchemeType
from Solvers.PDE_Solver.TerminalConditions import TerminalCondition
from Solvers.PDE_Solver.BoundariesConditions import Zero_Laplacian_BC
from VolatilitySurface.Tools import SABRTools
from AnalyticEngines.BetaZeroSabr import ExpansionTools
from functools import partial
from Tools.Bachelier import bachelier


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


pde_price = []
hagan_price = []
watanabe_price_lv = []
watanabe_price_sv = []

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
    pde_price.append(price)

    #hagan price
    iv_hagan = SABRTools.sabr_normal_jit(f0, strikes[i], alpha, rho, nu, t)
    hagan_price.append(bachelier(f0, strikes[i], t, iv_hagan, 'c'))

    #watanabe price
    price_sv = ExpansionTools.get_option_normal_sabr_watanabe_expansion(f0, strikes[i], t, alpha, nu, rho, 'c')
    price_lv = ExpansionTools.get_option_normal_sabr_loc_vol_expansion(f0, strikes[i], t, alpha, nu, rho, 'c')
    watanabe_price_lv.append(price_lv)
    watanabe_price_sv.append(price_sv)

plt.plot(strikes, hagan_price, label='Hagan price', linestyle='dotted')
# plt.plot(strikes, watanabe_price_sv, label='Watanabe price SV', linestyle='dashed')
plt.plot(strikes, watanabe_price_lv, label='Watanabe price LV', linestyle='-')
plt.plot(strikes, pde_price, label='PDE price', linestyle=':')

# iv_hagan = []
# iv_watanabe = []
# for i in range(0, len(strikes)):
#     iv = SABRTools.sabr_normal_jit(f0, strikes[i], alpha, rho, nu, t)
#     iv_hagan.append(iv)
#     iv = ExpansionTools.get_iv_normal_sabr_watanabe_expansion(f0, strikes[i], t, alpha, nu, rho)
#     iv_watanabe.append(iv)
#
#
# plt.plot(strikes, iv_hagan, label='IV Hagan', linestyle='dotted')
# plt.plot(strikes, iv_pde, label='IV Local Vol SABR', linestyle='dashed')
# plt.plot(strikes, iv_watanabe, label='IV Watanabe', linestyle='dashed')

plt.legend()
plt.show()



