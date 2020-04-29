import numpy as np
import numba as nb


from Tools.Types import ndarray, EULER_SCHEME_TYPE
from typing import Callable


def sde_euler_simulation(s: float,
                         t: float,
                         s0: float,
                         no_steps: int,
                         no_paths: int,
                         z: ndarray,
                         drift: Callable[[float, ndarray], ndarray],
                         sigma: Callable[[float, ndarray], ndarray],
                         euler_scheme: EULER_SCHEME_TYPE):

    t_i = np.linspace(s, t, no_steps)
    delta_t_i = np.diff(t_i)

    paths = np.empty(shape=(no_paths, no_steps))
    paths[:, 0] = s0

    if euler_scheme == EULER_SCHEME_TYPE.STANDARD:
        generator_step = get_euler_step
    elif euler_scheme == EULER_SCHEME_TYPE.LOG_NORMAL:
        generator_step = get_ln_euler_step
    else:
        raise Exception("The euler scheme " + str(euler_scheme) + "is " + str(EULER_SCHEME_TYPE.UNKNOWN))

    drift_i = np.empty(no_paths)
    sigma_i = np.empty(no_paths)

    for j in range(1, no_steps):
        drift_i = drift(t_i[j - 1], paths[:, j - 1])
        sigma_i = sigma(t_i[j - 1], paths[:, j - 1])
        paths[:, j] = generator_step(paths[:, j - 1], delta_t_i[j-1], z[:, j - 1], drift_i, sigma_i)

    return paths


@nb.jit("f8[:](f8[:], f8, f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def get_euler_step(s_i_1, delta_i, z_i, drift_i, sigma_i):
    no_paths = len(s_i_1)
    s_i = np.empty(no_paths)
    for i in range(0, no_paths):
        s_i[i] = np.maximum(s_i_1[i] + delta_i * drift_i[i] + np.sqrt(delta_i) * sigma_i[i] * z_i[i], 0.0)

    return s_i


@nb.jit("f8[:](f8[:], f8, f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def get_ln_euler_step(s_i_1, delta_i, z_i, drift_i, sigma_i):
    no_paths = len(s_i_1)
    s_i = np.empty(no_paths)
    for i in range(0, no_paths):
        s_i[i] = s_i_1[i] * np.exp(delta_i * (drift_i[i] / s_i_1[i]) - 0.5 * (sigma_i[i] * sigma_i[i]) / (s_i_1[i] * s_i_1[i]) * delta_i
                                   + np.sqrt(delta_i) * (sigma_i[i] / s_i_1[i]) * z_i[i])

    return s_i
