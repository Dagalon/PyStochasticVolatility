import numpy as np
import matplotlib.pylab as plt
from VolatilitySurface.Tools.SABRTools import sabr_vol_jit

from scipy.interpolate import interp1d
from MC_Engines.MC_LocalVol import LocalVolFunctionals
from Solvers.PDE_Solver import PDESolvers
from Solvers.PDE_Solver import PDEOperators
from Solvers.PDE_Solver.Meshes import uniform_mesh, Mesh
from Solvers.PDE_Solver.PDEs import NORMAL_LOCAL_VOL_PDE, PDE
from Solvers.PDE_Solver.TerminalConditions import TerminalCondition
from Solvers.PDE_Solver.Types import BoundaryConditionType, np_ndarray, SchemeType
from Solvers.PDE_Solver.BoundariesConditions import Zero_Laplacian_BC
from functools import partial
from VolatilitySurface.Tools import SABRTools
from AnalyticEngines.BetaZeroSabr import ExpansionTools
from Tools.Bachelier import bachelier


def quadratic_call(mesh: Mesh, k: float) -> np_ndarray:
    return np.maximum(mesh.get_points() - k, 0.0)


def quadratic_put(mesh: Mesh, k: float) -> np_ndarray:
    return np.maximum(k - mesh.get_points(), 0.0)


# option info
f0 = 0.03
t = 10.0

spreads = [-300.0, -250.0, -200.0, -150.0, -100.0, -50.0, 0.0, 10.0, 25.0, 50.0, 75.0, 100.0,
           150.0, 175.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0]

# spreads = [700]


strikes = []
options = []
for si in spreads:
    strikes.append(si / 10000.0 + f0)

# sabr parameters
alpha = 75.5/10000.0
nu = 0.24
rho = 0.235
parameters = [alpha, nu, rho]

# meshes
mesh_t = Mesh(uniform_mesh, 1500, 0.0, t)
mesh_x = Mesh(uniform_mesh, 600, -6.0, 6.0)
# mesh_x = BachelierUnderlyingMesh(alpha, f0, t, 0.99999999999, uniform_mesh, 1000)

# local vol info
sabr_loc_vol = partial(LocalVolFunctionals.local_vol_normal_sabr, x0=f0, alpha=alpha, rho=rho, nu=nu)

# pde
pde = PDE.from_ipde_terms(NORMAL_LOCAL_VOL_PDE(sabr_loc_vol))

tc_s = [TerminalCondition(partial(quadratic_call, k=k_i)) for k_i in strikes]

bc = Zero_Laplacian_BC()
operator_exp = PDEOperators.LinearPDEOperator(mesh_x, pde, bc)
operator_impl = PDEOperators.LinearPDEOperator(mesh_x, pde, bc)
operators = [operator_exp, operator_impl]


pde_price = []
quadratic_hagan_price = []
quadratic_watanabe_price = []
price_hagan = []

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

    sigma_hagan = SABRTools.sabr_normal_jit(f0, strikes[i], alpha, rho, nu, t)
    price_hagan.append(bachelier(f0, strikes[i], t, sigma_hagan, 'c'))
    analytic_quadratic_price = SABRTools.quadratic_european_normal_sabr(f0, strikes[i], alpha, rho, nu, t, 'c')
    # quadratic_hagan_price.append(analytic_quadratic_price)
    watanabe_price = ExpansionTools.get_option_normal_sabr_loc_vol_expansion(f0, strikes[i], t, alpha, nu, rho, 'c')

    # get_option_normal_sabr_loc_vol_expansion(f0, k, t, alpha, nu, rho, option_type)

    pde_price.append(price)
    quadratic_watanabe_price.append(watanabe_price)


# plt.xlim([-0.02, 0.075])
# plt.plot(strikes, pde_price, label='PDE price quadratic', linestyle=':')
plt.plot(strikes, price_hagan, label='Hagan analytic price quadratic', linestyle=':')
plt.plot(strikes, quadratic_watanabe_price, label='Watanabe analytic price quadratic', linestyle=':')

plt.legend()
plt.show()



