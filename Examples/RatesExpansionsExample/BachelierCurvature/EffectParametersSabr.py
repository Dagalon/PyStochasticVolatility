import numpy as np
from prettytable import PrettyTable

from VolatilitySurface.Tools import SABRTools


def curvature_limit_value_analytic(rho: float, nu: float, sigma0: float):
    return (1.0 / 3.0 - 0.5 * rho * rho) * nu * nu / sigma0


# forward
f0 = 0.03
t = 0.01

# CEV parameters
alpha = 0.02
rhos = [-0.5,  -0.3, 0.0, 0.3 , 0.5]
nus = np.linspace(0.1, 0.7, 5)

curvatures = PrettyTable(["\rho/\nu"] + [np.round(nui, 3) for nui in nus])

shift = 0.0001
for rhoi in rhos:
    row = [np.round(rhoi, 3)]
    for nui in nus:
        # hagan's volatility
        partial_iv_hagan_base = SABRTools.sabr_normal_partial_k_jit(f0, f0, alpha, rhoi, nui, t)
        partial_iv_hagan_up = SABRTools.sabr_normal_partial_k_jit(f0, f0 + shift, alpha, rhoi, nui, t)
        partial_iv_hagan_down = SABRTools.sabr_normal_partial_k_jit(f0, f0 - shift, alpha, rhoi, nui, t)

        atm_curvature_malliavin = curvature_limit_value_analytic(rhoi, nui, alpha)

        hagan_curvature = 0.5 * (partial_iv_hagan_up - partial_iv_hagan_down) / shift
        row.append(f'{np.round(hagan_curvature,3)}/{np.round(atm_curvature_malliavin,3)}')

    curvatures.add_row(row)

print(curvatures.get_latex_string())