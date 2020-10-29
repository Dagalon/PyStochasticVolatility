import numpy as np

from MC_Engines.MC_RBergomi import ToolsVariance
from Tools.Types import Vector, ndarray, TYPE_STANDARD_NORMAL_SAMPLING, RBERGOMI_OUTPUT


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
                        rnd_generator):

    nu = parameters[0]
    rho = parameters[1]
    h = parameters[2]

    no_paths = 2 * no_paths if type_random_number == TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC else no_paths

    t_i_s = np.linspace(t0, t1, no_time_steps)

    s_t = np.empty((no_paths, no_time_steps))
    s_t[:, 0] = f0

    z_i_s = rnd_generator.normal(mu=0.0, sigma=1.0, size=(2 * (no_time_steps - 1), no_paths))
    map_out_put = {}
    # cov_matrix = ToolsVariance.get_covariance_matrix(t_i_s[1:], h, rho)
    outputs = ToolsVariance.generate_paths(f0,
                                           v0,
                                           nu,
                                           h,
                                           z_i_s,
                                           np.linalg.cholesky(ToolsVariance.get_covariance_matrix(t_i_s[1:], h, rho)),
                                           t_i_s,
                                           no_paths)

    map_out_put[RBERGOMI_OUTPUT.PATHS] = outputs[0]
    map_out_put[RBERGOMI_OUTPUT.SPOT_VARIANCE_PATHS] = outputs[1]
    map_out_put[RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS] = outputs[2]

    # test cov
    # variance = ToolsVariance.get_volterra_variance(t_i_s[1:no_time_steps], h)
    # for i in range(1, no_time_steps):
    #     for j in range(1, no_time_steps):
    #         w_i = np.log(outputs[1][:, i] / v0) + 0.5 * nu * nu * variance[i - 1]
    #         w_j = np.log(outputs[1][:, j] / v0) + 0.5 * nu * nu * variance[j - 1]
    #         rho_i_j = np.mean(w_i * w_j) - np.mean(w_i) * np.mean(w_j)
    #         rho_model = cov_matrix[no_time_steps + i - 2, no_time_steps + j - 2] * nu * nu

    return map_out_put











