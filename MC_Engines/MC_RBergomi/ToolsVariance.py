import numpy as np
import numba as nb
from Tools.Types import ndarray
from ncephes import hyp2f1
from typing import Callable


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def get_volterra_covariance(s: float, t: float, h: float):
    # We suppose that t > s
    gamma = 0.5 - h
    x = s / t

    return ((1.0 - 2.0 * gamma) / (1.0 - gamma)) * np.power(x, gamma) * hyp2f1(1.0, gamma, 2.0 - gamma, x)


@nb.jit("f8(f8, f8)", nopython=True, nogil=True)
def get_volterra_variance(t: float, h: float):
    return np.power(t, - 2.0 * h)


@nb.jit("f8(f8, f8, f8, f8)", nopython=True, nogil=True)
def get_covariance_matrix(s: float, u: float, t: float, h: float):
    # we suppose that s < u < t
    cov = np.zeros(shape=(3, 3))
    cov[0, 0] = get_volterra_covariance(s, s, h)
    cov[0, 1] = get_volterra_covariance(s, u, h)
    cov[1, 0] = cov[0, 1]
    cov[0, 2] = get_volterra_covariance(s, t, h)
    cov[2, 0] = cov[0, 2]
    cov[1, 1] = get_volterra_covariance(u, u, h)
    cov[1, 2] = get_volterra_covariance(u, t, h)
    cov[2, 1] = cov[1, 2]
    cov[2, 2] = get_volterra_covariance(t, t, h)

    return cov


@nb.jit("f8[:,:](f8, f8, f8, f8[:], f8[:], f8[:,:])", nopython=True, nogil=True)
def get_volterra_bridge_moments(t_i_1: float, t: float, t_i: float, x_t_i_1: float, x_t_i: float, cov: ndarray):
    # We must have that t_i_1 < t < t_i
    no_elements = len(x_t_i_1)

    # moments[0] is mean
    # moments[1] is std
    moments = np.zeros(shape=(no_elements, 2))
    a = np.linalg.cholesky(cov)

    for i in range(0, no_elements):
        moments[i, 0] = (a[2, 0] / a[0, 0]) - (a[2, 1] / a[1, 1]) * (a[1, 0] / a[0, 0]) * x_t_i_1[i] + \
                        (a[2, 1] / a[1, 1]) * x_t_i[i]
        moments[i, 1] = a[2, 1] / a[2, 2]

    return moments


@nb.jit("f8(f8, f8, f8, f8)", nopython=True, nogil=True)
def get_covariance_w_v_w_t(s: float, t: float, rho: float, h: float):
    gamma = 0.5 - h
    d_h = np.sqrt(2.0 * h) / (h + 0.5)
    return rho * d_h * (np.power(t, h + 0.5) - np.power(t - np.minimum(s,t), h + 0.5))


@nb.jit("(f8, f8, f8, f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, f8, f8[:], f8[:])", nopython=True, nogil=True)
def get_gaussian_bridge(u, s, t, w_u, w_t, w_v_u, w_v_t, z_w, z_v, rho_w, h, w_s, w_v_s):
    no_paths = len(w_u)

    d_u_s = s - u
    d_s_t = t - s
    d_u_t = t - u

    process = np.zeros(shape=(no_paths, 2))

    mean_w_s = (d_s_t * w_u + d_u_s * w_t) / d_u_t
    std_w_s = np.sqrt(d_s_t * d_u_s / d_u_t)

    inv_rho_w = np.sqrt(1.0 - rho_w * rho_w)

    cov_w_v = get_covariance_matrix(u, s, t, h)
    moments_w_v_s = get_volterra_bridge_moments(u, s, t, w_v_u, w_v_t, cov_w_v)

    for i in range(0, no_paths):
        w_s[i] = mean_w_s + std_w_s * z_w[i]
        w_v_s[i] = moments_w_v_s[0] + moments_w_v_s[1] * (rho_w * z_w[i] + inv_rho_w * z_v[i])


def get_path_gaussian_bridge(t0, t1, n, no_paths, h, rho, rnd_generator):
    no_steps = int(2 ** n)
    delta_step = (t1 - t0) / no_steps

    w_t_paths = np.zeros(shape=(no_paths, no_steps + 1))
    w_v_t_paths = np.zeros(shape=(no_paths, no_steps + 1))

    z_w_i = np.zeros(no_paths)
    z_w_v_i = np.zeros(no_paths)

    rho_w = get_covariance_w_v_w_t(t1, t1, rho, h)

    np.copyto(z_w_i, rnd_generator.normal(0.0, 1.0, no_paths))
    np.copyto(z_w_v_i, rnd_generator.normal(0.0, 1.0, no_paths))

    w_t_paths[:, -1] = np.sqrt(t1) * z_w_i

    var_t = get_volterra_variance(t1, h)
    w_v_t_paths[:, -1] = np.sqrt(var_t) * (np.add(rho_w * z_w_i, np.sqrt(1.0 - rho_w * rho_w) * z_w_v_i))

    for n_i in range(1, n + 1):
        scale_factor = 2 ** (n - n_i)
        even_nodes = [scale_factor * k for k in range(0, int(2 ** n_i) + 1) if k % 2 == 0]
        odd_nodes = [scale_factor * k for k in range(0, int(2 ** n_i) + 1) if k % 2 == 1]
        rho_w = get_covariance_w_v_w_t(odd_nodes[n_i], odd_nodes[n_i], rho, h)

        get_gaussian_bridge(even_nodes[n_i] * delta_step,
                            odd_nodes[n_i] * delta_step,
                            even_nodes[n_i + 1] * delta_step,
                            w_t_paths[:, even_nodes[n_i]],
                            w_t_paths[:, even_nodes[n_i + 1]],
                            w_v_t_paths[:, even_nodes[n_i]],
                            w_v_t_paths[:, even_nodes[n_i + 1]],
                            rho_w,
                            h,
                            w_t_paths[:, odd_nodes[n_i]],
                            w_v_t_paths[:, odd_nodes[n_i]])

    return w_t_paths, w_v_t_paths

