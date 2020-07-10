import numpy as np
import numba as nb

from MC_Engines.MC_RBergomi import ToolsVariance
from Tools import Functionals
from Tools.Types import Vector, ndarray, SABR_OUTPUT, TYPE_STANDARD_NORMAL_SAMPLING


def get_v_t_sampling(t: float,
                     h: float,
                     nu: float,
                     z: ndarray):

    std_t = np.sqrt(ToolsVariance.get_volterra_covariance(t, h))
    return std_t * z


def get_path_multi_step(t0: float,
                        t1: float,
                        parameters: Vector,
                        f0: float,
                        no_paths: int,
                        no_time_steps: int,
                        type_random_number: TYPE_STANDARD_NORMAL_SAMPLING,
                        rnd_generator) -> map:

    nu = parameters[0]
    rho = parameters[1]
    rho_inv = np.sqrt(1.0 - rho * rho)

    no_paths = 2 * no_paths if type_random_number == TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC else no_paths

    t_i = np.linspace(t0, t1, no_time_steps)
    delta_t_i = np.diff(t_i)

    s_t = np.empty((no_paths, no_time_steps))
    s_t[:, 0] = f0







