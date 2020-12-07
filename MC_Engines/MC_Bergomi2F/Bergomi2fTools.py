import numpy as np
import numba as nb

from Tools import Types


@nb.jit("void(f8,f8,f8,f8,f8,f8,i8,f8[:,:],f8[:,:],f8[:],f8[:],f8[:],f8[:],f8[:])", nopython=True, nogil=True)
def get_log_spot(alpha_theta: float, nu_x: float, nu_y: float, theta: float, t_i_1: float, t_i: float, no_paths: int,
                 w_t_i_1: Types.ndarray, w_t_i: Types.ndarray, log_f_i_1: Types.ndarray, log_f_i: Types.ndarray,
                 log_v_i_1: Types.ndarray, log_v_i: Types.ndarray, int_v_t_paths: Types.ndarray):

    delta_time = (t_i - t_i_1)

    for k in range(0, no_paths):
        delta_w_x_t = w_t_i[0, k] - w_t_i_1[0, k]
        delta_w_y_t = w_t_i[1, k] - w_t_i_1[1, k]
        delta_w_f_t = w_t_i[2, k] - w_t_i_1[2, k]
        log_v_i[k] = log_v_i_1[k] - 0.5 * alpha_theta * alpha_theta * delta_time + \
                     (nu_x * (1.0 - theta) * delta_w_x_t + nu_y * theta * delta_w_y_t)
        v_i_1 = np.exp(log_v_i_1[k])
        v_i = np.exp(log_v_i[k])
        v_mid_i = 0.5 * (v_i_1 + v_i)
        int_v_t_paths[k] = v_mid_i * delta_time
        log_f_i[k] = log_f_i_1[k] - 0.5 * delta_time * v_mid_i + np.sqrt(v_mid_i) * delta_w_f_t



