import numpy as np
import numba as nb

from scipy.integrate import quad_vec, quad, quadrature
from Tools.Types import ndarray
from functools import partial
from ncephes import hyp2f1


# @nb.jit("f8[:](f8[:], f8)", nopython=True, nogil=True)
def get_variance(t: ndarray, beta: float):
    return np.power(-np.log(t), -beta)


@nb.jit("f8(f8, f8, f8, f8)", nopython=True, nogil=True)
def get_kernel(u, s, t, beta):
    if s < t:
        d_t_s = (t - s)
        return np.power((d_t_s + u) * u, -0.5) * np.power(np.log(u) * np.log(d_t_s + u), -0.5 * (beta + 1.0))
    else:
        d_t_s = (s - t)
        return np.power((d_t_s + u) * u, -0.5) * np.power(np.log(u) * np.log(d_t_s + u), -0.5 * (beta + 1.0))


def get_volterra_covariance(s: float, t: float, beta: float):
    # epsilon = 1.0e-03
    if s > 0.0 and t > 0.0:
        f_kernel = partial(get_kernel, s=s, t=t, beta=beta)
        min_t_s = np.minimum(t, s)
        integral_value = quad(f_kernel, 0.0, min_t_s)

        # Approximation for the singularity
        # partial_integral_value = quad(f_kernel, min_t_s * epsilon, min_t_s)
        # if s > t:
        #     d_s_t = (s - t)
        #     alpha = (2.0 * np.sqrt(epsilon * t) / 3.0) * np.power(d_s_t, - 0.5) * np.power(-np.log(d_s_t),
        #                                                                                    - 0.5 * (beta + 1.0))
        #
        #     aux_value = (partial_integral_value[0] + alpha * hyp2f1((beta + 1.0) * 0.5, 0.5, 1.5, min_t_s * epsilon)) * beta
        #
        # elif s < t:
        #     d_s_t = (t - s)
        #     alpha = (2.0 * np.sqrt(epsilon * s) / 3.0) * np.power(d_s_t, - 0.5) * np.power(-np.log(d_s_t),
        #                                                                                    - 0.5 * (beta + 1.0))
        #
        #     aux_value = (partial_integral_value[0] + alpha * hyp2f1((beta + 1.0) * 0.5, 0.5, 1.5, min_t_s * epsilon)) * beta
        #
        # else:
        #
        #     aux_value = partial_integral_value[0] * beta + np.power(-np.log(min_t_s * epsilon), - beta)

        # return (partial_integral_value[0] + alpha * hyp2f1((beta + 1.0) * 0.5, 0.5, 1.5, min_t_s * epsilon)) * beta

        return integral_value[0] * beta

    else:
        return 0.0
