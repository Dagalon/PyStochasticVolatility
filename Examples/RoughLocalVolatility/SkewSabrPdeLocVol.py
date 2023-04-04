import numpy as np
import matplotlib.pylab as plt

from scipy.interpolate import interp1d
from MC_Engines.MC_LocalVol import LocalVolFunctionals
from Solvers.PDE_Solver import PDESolvers
from Solvers.PDE_Solver import PDEOperators
from Solvers.PDE_Solver.Meshes import uniform_mesh, Mesh, LnUnderlyingMesh
from Solvers.PDE_Solver.PDEs import LN_FORWARD_LOCAL_VOL_PDE, PDE
from Solvers.PDE_Solver.TerminalConditions import TerminalCondition
from Solvers.PDE_Solver.Types import BoundaryConditionType, np_ndarray, SchemeType
from Solvers.PDE_Solver.BoundariesConditions import Zero_Laplacian_BC
from VolatilitySurface.Tools import SABRTools
from functools import partial
from py_vollib.black_scholes_merton import black_scholes_merton


def call(mesh: Mesh, k: float) -> np_ndarray:
    return np.maximum(np.exp(mesh.get_points()) - k, 0.0)


# option info
f0 = 100.0
t = 0.5
strikes = [f0 + i for i in range(-5, 5)]

# sabr parameters
alpha = 0.3
nu = 0.6
rho = -0.6
parameters = [alpha, nu, rho]

# meshes
mesh_t = Mesh(uniform_mesh, 100, 0.0, t)
mesh_x = LnUnderlyingMesh(0.0, 0.0, alpha, f0, t, 0.9999, uniform_mesh, 400)

# local vol info
sabr_loc_vol = partial(LocalVolFunctionals.local_vol_log_normal_sabr, x0=f0, alpha=alpha, rho=rho, nu=nu)
lv = lambda t, x: sabr_loc_vol(t, np.exp(x))
# pde
pde = PDE.from_ipde_terms(LN_FORWARD_LOCAL_VOL_PDE(lv))

tc_s = [TerminalCondition(partial(call, k=k_i)) for k_i in strikes]

bc = Zero_Laplacian_BC()
operator_exp = PDEOperators.LinearPDEOperator(mesh_x, pde, bc)
operator_impl = PDEOperators.LinearPDEOperator(mesh_x, pde, bc)
operators = [operator_exp, operator_impl]


pde_price = []
hagan_price = []
iv_hagan = []

for i in range(0, len(strikes)):
    pd_solver = PDESolvers.FDSolver(mesh_t,
                                    mesh_x,
                                    operators,
                                    SchemeType.CRANK_NICOLSON,
                                    BoundaryConditionType.ZERO_DIFFUSION,
                                    tc_s[i])

    pd_solver.solver()
    f = interp1d(mesh_x.get_points(), pd_solver._u_grid[:, 0], kind='linear', fill_value='extrapolate')
    price = float(f(np.log(f0)))
    pde_price.append(price)

    # hagan price
    z = np.log(f0 / strikes[i])
    iv_hagan.append(SABRTools.sabr_vol_jit(alpha, rho, nu, z, t))
    hagan_price.append(black_scholes_merton('c', f0, strikes[i], t, 0.0, iv_hagan[-1], 0.0))

plt.plot(strikes, hagan_price, label='Hagan price', linestyle='dotted')
plt.plot(strikes, pde_price, label='PDE price', linestyle=':')

plt.legend()
plt.show()



