import numpy as np
import numba as nb

from AnalyticEngines.LocalVolatility.Dupire import DupireFormulas


@nb.jit("f8(f8,f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def svi_total_imp_var_jit(a, b, rho, sigma, m, z):
    diff = (z - m)
    var = a + b * (rho * diff + np.sqrt(np.power(diff, 2.0) + np.power(sigma, 2.0)))

    return var


@nb.jit("f8[:](f8,f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def get_gradient_svi_iv_to_parameters_jit(a, b, rho, sigma, m, z):
    diff = (z - m)

    gradient = np.zeros(5)

    aux_sqrt = np.sqrt(np.power(diff, 2.0) + np.power(sigma, 2.0))
    # partial a
    gradient[0] = b * (rho * diff + aux_sqrt)

    # partial b
    gradient[1] = rho * diff + aux_sqrt

    # partial rho
    gradient[2] = diff * b

    # partial sigma
    gradient[3] = b * sigma / aux_sqrt

    # partial m
    gradient[4] = -b * (diff + rho / aux_sqrt)

    return gradient


@nb.jit("f8[:](f8,f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def get_derive_svi_to_z_jit(a, b, rho, sigma, m, z):
    diff = (z - m)
    output = np.zeros(2)
    aux = np.power(diff, 2.0) + np.power(sigma, 2.0)
    aux_sqrt = np.sqrt(aux)
    output[0] = b * (rho + diff / aux_sqrt)
    output[1] = b * (1.0 - (np.power(diff, 2) / aux)) / aux_sqrt

    return output


@nb.jit("f8[:](f8,f8,f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def get_derive_svi_to_k_jit(a, b, rho, sigma, m, k, z):
    output = np.zeros(2)

    partial_z = get_derive_svi_to_z_jit(a, b, rho, sigma, m, z)
    partial_z_k = 1.0 / k

    output[0] = partial_z[0] * partial_z_k
    output[1] = (partial_z[1] - partial_z[0]) * partial_z_k * partial_z_k

    return output


@nb.jit("f8[:](f8[:],f8[:],f8,f8,f8,f8,f8[:])", nopython=True, nogil=True)
def get_svi_loc_vol(p_i_1, p_i, t_i_1, t, t_i, f0_t, x_t):
    # mu_t = \int_t^T r_d(u) - r_f(u) du
    # x_t = log(f_t_T)
    # f_0_t forward at value_date with maturity t
    # z_t = log(k_t/f0_t) where k_t = exp(x_t - mu_t)
    # p_i_1 left paramters of svi at time t_i_1

    no_paths = len(x_t)
    loc_vol = np.zeros(no_paths)
    log_f0_t = np.log(f0_t)
    alpha_t = (t - t_i_1) / (t_i - t_i_1)

    if len(p_i_1) == 0:
        for i in range(0, no_paths):
            k_t = np.exp(x_t[i])
            z_t = x_t[i] - log_f0_t
            var_z_i = svi_total_imp_var_jit(p_i[0], p_i[1], p_i[2], p_i[3], p_i[4], z_t)

            # First and second derivative with respect to strike of the SVI.
            var_partial_z_i = get_derive_svi_to_z_jit(p_i[0], p_i[1], p_i[2], p_i[3], p_i[4], z_t) *  alpha_t

            #  Time partial of the svi
            var_partial_t = var_z_i / t_i

            # Compute the local vol
            if t < 5.0 / 365.0:
                var_atm_i = svi_total_imp_var_jit(p_i[0], p_i[1], p_i[2], p_i[3], p_i[4], z_t)
                loc_vol[i] = np.sqrt(var_atm_i / t_i)
            else:
                var_t = DupireFormulas.local_vol_from_variance(z_t, var_z_i * alpha_t, var_partial_z_i[0],
                                                               var_partial_z_i[1], var_partial_t)
                loc_vol[i] = np.sqrt(var_t)

    elif len(p_i) == 0:
        for i in range(0, no_paths):
            k_t = np.exp(x_t[i])
            z_t = x_t[i] - log_f0_t  # esto hay que revisarlo, me da que no es asi!!!
            var_z_i_1 = svi_total_imp_var_jit(p_i_1[0], p_i_1[1], p_i_1[2], p_i_1[3], p_i_1[4], z_t)

            # First and second derivative with respect to strike of the SVI.
            var_partial_z_i_1 = get_derive_svi_to_z_jit(p_i_1[0], p_i_1[1], p_i_1[2], p_i_1[3], p_i_1[4], z_t) * (1.0 - alpha_t)

            #  Time partial of the svi
            var_partial_t = var_z_i_1 / t_i_1

            # Compute the local vol
            var_t = DupireFormulas.local_vol_from_variance(z_t, var_z_i_1 * (1.0 - alpha_t), var_partial_z_i_1[0],
                                                           var_partial_z_i_1[1], var_partial_t)

            loc_vol[i] = np.sqrt(var_t)

    else:
        for i in range(0, no_paths):
            k_t = np.exp(x_t[i])
            z_t = x_t[i] - log_f0_t  # esto hay que revisarlo, me da que no es asi!!!

            var_z_i_1 = svi_total_imp_var_jit(p_i_1[0], p_i_1[1], p_i_1[2], p_i_1[3], p_i_1[4], z_t)
            var_z_i = svi_total_imp_var_jit(p_i[0], p_i[1], p_i[2], p_i[3], p_i[4], z_t)

            # First and second derivative with respect to strike of the SVI.
            var_partial_z_i_1 = get_derive_svi_to_z_jit(p_i_1[0], p_i_1[1], p_i_1[2], p_i_1[3], p_i_1[4], z_t)
            var_partial_z_i = get_derive_svi_to_z_jit(p_i[0], p_i[1], p_i[2], p_i[3], p_i[4], z_t)

            #  Time partial of the svi
            var_partial_t = (var_z_i - var_z_i_1) / (t_i - t_i_1)

            # Compute the local vol

            var_t = DupireFormulas.local_vol_from_variance(z_t,
                                                           var_z_i_1 * (1.0 - alpha_t) + var_z_i * alpha_t,
                                                           var_partial_z_i_1[0] * (1.0 - alpha_t) + var_partial_z_i[0] * alpha_t,
                                                           var_partial_z_i_1[1] * (1.0 - alpha_t) + var_partial_z_i[1] * alpha_t,
                                                           var_partial_t)
            loc_vol[i] = np.sqrt(var_t)

    return loc_vol
