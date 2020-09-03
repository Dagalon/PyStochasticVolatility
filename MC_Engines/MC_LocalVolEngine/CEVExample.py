import time
import numpy as np
from functools import partial
from MC_Engines.MC_LocalVolEngine import LocalVolEngine
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import RNG, Types
from MC_Engines.MC_LocalVolEngine import LocalVolFunctionals
from py_vollib.black_scholes_merton import black_scholes_merton
from scipy.interpolate import interp1d
from Solvers.PDE_Solver import PDESolvers
from Solvers.PDE_Solver import PDEOperators
from Solvers.PDE_Solver.Meshes import uniform_mesh, Mesh, LnUnderlyingMesh
from Solvers.PDE_Solver.PDEs import LN_FORWARD_LOCAL_VOL_PDE, PDE
from Solvers.PDE_Solver.Types import BoundaryConditionType, np_ndarray, SchemeType
from Solvers.PDE_Solver.TerminalConditions import TerminalCondition
from Solvers.PDE_Solver.BoundariesConditions import Zero_Laplacian_BC

f0 = 100
seed = 123456789
no_paths = 50000
T = 2.0

# CEV parameter
alpha = 1.0
nu = 0.3
diffusion = partial(LocalVolFunctionals.cev_diffusion, beta=alpha - 1, sigma=nu)

epsilon = 1.0 / 32.0
no_time_steps = int(T / epsilon)

strike = 100.0
log_strike = np.log(strike)
notional = 1.0

european_option = EuropeanOption(strike, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)

rnd_generator = RNG.RndGenerator(seed)

# Compute the price of the option by MC
start_time = time.time()
map_output = LocalVolEngine.get_path_multi_step(0.0, T, f0, no_paths, no_time_steps,
                                                Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                diffusion,
                                                rnd_generator)
end_time = time.time()
delta_time = (end_time - start_time)

# BS price
bs_price = black_scholes_merton('c', f0, strike, T, 0.0, nu, 0.0)

# MC price
result = european_option.get_price(map_output[Types.LOCAL_VOL_OUTPUT.PATHS])
price = result[0]
std = result[1]

# PDE price
mesh_t = Mesh(uniform_mesh, 100, 0.0, T)
mesh_x = LnUnderlyingMesh(0.0, 0.0, nu, f0, T, 0.999, uniform_mesh, 200)
log_diffusion = partial(LocalVolFunctionals.log_cev_diffusion, beta=alpha - 1, sigma=nu)
cev_pde = PDE.from_ipde_terms(LN_FORWARD_LOCAL_VOL_PDE(log_diffusion))


def f_ln_payoff(mesh: Mesh) -> np_ndarray:
    return np.maximum(np.exp(mesh.get_points()) - strike, 0.0)


bc = Zero_Laplacian_BC()
tc = TerminalCondition(f_ln_payoff)

operator_exp = PDEOperators.LinearPDEOperator(mesh_x, cev_pde, bc)
operator_impl = PDEOperators.LinearPDEOperator(mesh_x, cev_pde, bc)
operators = [operator_exp, operator_impl]

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
pde_price = float(f(np.log(f0)))
print(end_time - start_time)
print(pde_price)
