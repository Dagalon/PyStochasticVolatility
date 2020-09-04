import time
import numpy as np
import matplotlib.pylab as plt

from functools import partial
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import RNG, Types
from MC_Engines.MC_LocalVolEngine import LocalVolFunctionals,LocalVolEngine
from scipy.interpolate import interp1d
from Solvers.PDE_Solver import PDESolvers
from Solvers.PDE_Solver import PDEOperators
from Solvers.PDE_Solver.Meshes import uniform_mesh, Mesh, LnUnderlyingMesh
from Solvers.PDE_Solver.PDEs import LN_FORWARD_LOCAL_VOL_PDE, PDE
from Solvers.PDE_Solver.Types import BoundaryConditionType, np_ndarray, SchemeType
from Solvers.PDE_Solver.TerminalConditions import TerminalCondition
from Solvers.PDE_Solver.BoundariesConditions import Zero_Laplacian_BC
from py_vollib.black_scholes_merton import black_scholes_merton
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

f0 = 100
seed = 123456789
no_paths = 300000
T = 2.0

# CEV parameter
alpha = 0.9
nu = 0.3
diffusion = partial(LocalVolFunctionals.log_cev_diffusion, beta=alpha - 1, sigma=nu)

epsilon = 1.0 / 32.0
no_time_steps = int(T / epsilon)

strike = 100.0
log_strike = np.log(strike)
notional = 1.0

european_option = EuropeanOption(strike, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)

rnd_generator = RNG.RndGenerator(seed)

# Smile curve with cev
mesh_t = Mesh(uniform_mesh, 100, 0.0, T)
mesh_x = LnUnderlyingMesh(0.0, 0.0, nu, f0, T, 0.999, uniform_mesh, 200)
log_diffusion = partial(LocalVolFunctionals.log_cev_diffusion, beta=alpha - 1, sigma=nu)
cev_pde = PDE.from_ipde_terms(LN_FORWARD_LOCAL_VOL_PDE(log_diffusion))


def f_ln_payoff(mesh: Mesh, k: float) -> np_ndarray:
    return np.maximum(np.exp(mesh.get_points()) - k, 0.0)


k_s = np.linspace(70, 130, 20)
tc_s = [TerminalCondition(partial(f_ln_payoff, k=k_i)) for k_i in k_s]

bc = Zero_Laplacian_BC()

operator_exp = PDEOperators.LinearPDEOperator(mesh_x, cev_pde, bc)
operator_impl = PDEOperators.LinearPDEOperator(mesh_x, cev_pde, bc)
operators = [operator_exp, operator_impl]

start_time = time.time()
pde_price = []
bs_price = []
mc_price = []

for k_i in k_s:
    bs_price.append(black_scholes_merton('c', f0, k_i, T, 0.0, nu, 0.0))

start_time = time.time()
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

end_time = time.time()
print(end_time - start_time)

# start_time = time.time()
# for k_i in k_s:
#     european_option = EuropeanOption(k_i, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)
#     map_output = LocalVolEngine.get_path_multi_step(0.0, T, f0, no_paths, no_time_steps,
#                                                     Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
#                                                     diffusion,
#                                                     rnd_generator)
#
#     result = european_option.get_price(map_output[Types.LOCAL_VOL_OUTPUT.PATHS])
#     mc_price.append(result[0])
#     std = result[1]
#
# end_time = time.time()
# print(end_time - start_time)

# Compute the iv
no_elements = len(pde_price)
iv_bs = []
iv_fd = []
iv_mc = []
z_s = []

for i in range(0, no_elements):
    z_s.append(np.log(k_s[i] / f0))
    # iv_bs.append(implied_volatility(bs_price[i], f0, k_s[i], T, 0.0, 0.0, 'c'))
    iv_fd.append(implied_volatility(pde_price[i], f0, k_s[i], T, 0.0, 0.0, 'c'))
    # iv_mc.append(round(implied_volatility(mc_price[i], f0, k_s[i], T, 0.0, 0.0, 'c'), 3))

plt.plot(z_s, iv_fd, label="FD implied vol", color="black", linestyle='dotted')
# plt.plot(k_s, iv_bs, label="BS implied vol", color="red", linestyle='dashed')
# plt.plot(z_s, iv_mc, label="MC implied vol", color="green", linestyle='dotted')

plt.title("B = " + str(alpha))
plt.legend()
plt.show()
