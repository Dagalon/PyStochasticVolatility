import numpy as np
from numba import jit

from Tools.Types import ndarray
from MC_Engines.MC_Heston import HestonTools
from ncephes import ndtri


@jit("f8[:](f8,f8,f8,f8,f8,f8,f8[:],f8[:], i8)", nopython=True, nogil=True)
def get_variance(k: float,
                 theta: float,
                 epsilon: float,
                 phi_switch_level: float,
                 t_i_1: float,
                 t_i: float,
                 v_t_i_1: ndarray,
                 u_i: ndarray,
                 no_paths: int):

    # no_paths = len(v_t_i_1)
    paths = np.zeros(no_paths)

    for i in range(0, no_paths):
        s_2_i = HestonTools.v_t_conditional_variance(k, theta, epsilon, v_t_i_1[i], t_i_1, t_i)
        m_i = HestonTools.v_t_conditional_mean(k, theta, v_t_i_1[i], t_i_1, t_i)
        phi = s_2_i / (m_i * m_i)

        if phi < phi_switch_level:
            parameters = HestonTools.matching_qe_moments_qg(m_i, s_2_i)
            z_i = ndtri(u_i[i])
            paths[i] = parameters[1] * np.power(parameters[0] + z_i, 2.0)
        else:
            parameters = HestonTools.matching_qe_moments_exp(m_i, s_2_i)
            paths[i] = HestonTools.inv_exp_heston(parameters[0], parameters[1], u_i[i])

    return paths
