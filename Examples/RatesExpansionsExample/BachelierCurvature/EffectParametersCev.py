import numpy as np
from prettytable import PrettyTable

from AnalyticEngines.LocalVolatility.Hagan import ExpansionLocVol
from VolatilitySurface.Tools import SABRTools


def curvature_limit_value_analytic(spot: float, sigma: float, gamma: float):
    return np.power(spot, gamma) * (gamma * gamma * sigma - 2.0 * gamma * sigma) / (6 * spot * spot)


# forward
f0 = 0.03
t = 0.01

# CEV parameters
alphas =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
nus = np.linspace(0.01, 0.03, 5)

curvatures = PrettyTable(["\sigma/\nu"] + [np.round(nui, 3) for nui in nus])

shift = 0.0001
for alphai in alphas:
    row = [np.round(alphai, 3)]
    for nui in nus:
        # hagan's curvature
        curvature = expansion_hagan.get_bachelier_curvature(t, f0, f0)



        # Hagan approximation
        expansion_hagan = ExpansionLocVol.hagan_loc_vol(lambda t: nui,
                                                        lambda x: np.power(x, alphai),
                                                        lambda x: alphai * np.power(x, alphai - 1.0),
                                                        lambda x: alphai * (alphai - 1.0) * np.power(x, alphai - 2.0))

        cev_iv = SABRTools.cev_iv_normal_jit(f0, f0, nui, alphai, t)
        cev_iv_upper = SABRTools.cev_iv_normal_jit(f0, f0 + shift, nui, alphai, t)
        cev_iv_lower = SABRTools.cev_iv_normal_jit(f0, f0 - shift, nui, alphai, t)
        finite_difference_curavature = (cev_iv_upper + cev_iv_lower - 2.0 * cev_iv) / (shift * shift)
        limit_value_analytic_at_short_term = curvature_limit_value_analytic(f0, nui, alphai)
        row.append(f'{np.round(finite_difference_curavature,3)}/{np.round(limit_value_analytic_at_short_term,3)}')

    curvatures.add_row(row)

print(curvatures.get_latex_string())

