import numpy as np

from MC_Engines.MC_RBergomi import ToolsVariance
from Tools.Types import Vector, ndarray, TYPE_STANDARD_NORMAL_SAMPLING


def get_v_t_sampling(t: float,
                     h: float,
                     z: ndarray):

    std_t = np.sqrt(ToolsVariance.get_volterra_covariance(t, h))
    return std_t * z


def get_path_multi_step(t0: float,
                        t1: float,
                        parameters: Vector,
                        f0: float,
                        v0: float,
                        no_paths: int,
                        no_time_steps: int,
                        type_random_number: TYPE_STANDARD_NORMAL_SAMPLING,
                        rnd_generator) -> map:

    nu = parameters[0]
    rho = parameters[1]
    h = parameters[2]

    no_paths = 2 * no_paths if type_random_number == TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC else no_paths

    t_i_s = np.linspace(t0, t1, no_time_steps)

    s_t = np.empty((no_paths, no_time_steps))
    s_t[:, 0] = f0

    z_i_s = rnd_generator.normal(mu=0.0, sigma=1.0, size=(2 * (no_time_steps - 1), no_paths))

    paths = ToolsVariance.generate_paths(f0,
                                         v0,
                                         nu,
                                         h,
                                         z_i_s,
                                         np.linalg.cholesky(ToolsVariance.get_covariance_matrix(t_i_s[1:], h, rho)),
                                         t_i_s,
                                         no_paths)

    return paths











