import matplotlib.pylab as plt
import numpy as np

from Tools import Types
from VolatilitySurface.Tools import SABRTools


def get_smile_for_differents_rho(alpha: float,
                                 rho_s: Types.ndarray,
                                 nu: float,
                                 k_s: Types.ndarray,
                                 f0: float,
                                 T: float):
    no_strikes = len(k_s)

    iv_hagan = []
    for rho_i in rho_s:
        iv_list = []
        for i in range(0, no_strikes):
            z = np.log(f0 / k_s[i])
            iv_list.append(SABRTools.sabr_vol_jit(alpha, rho_i, nu, z, T))

        iv_hagan.append((rho_i, iv_list))

    return iv_hagan


def get_smile_for_differents_nu(alpha: float,
                                 rho: float,
                                 nu_s: Types.ndarray,
                                 k_s: Types.ndarray,
                                 f0: float,
                                 T: float):
    no_strikes = len(k_s)

    iv_hagan = []
    for nu_i in nu_s:
        iv_list = []
        for i in range(0, no_strikes):
            z = np.log(f0 / k_s[i])
            iv_list.append(SABRTools.sabr_vol_jit(alpha, rho, nu_i, z, T))

        iv_hagan.append((nu_i, iv_list))

    return iv_hagan


def get_smile_for_differents_alpha(alpha_s: Types.ndarray,
                                    rho: float,
                                    nu: float,
                                    k: float,
                                    f0: float,
                                    T: float):
    no_strikes = len(k_s)

    iv_hagan = []
    for alpha_i in alpha_s:
        iv_list = []
        for i in range(0, no_strikes):
            z = np.log(f0 / k_s[i])
            iv_list.append(SABRTools.sabr_vol_jit(alpha_i, rho, nu, z, T))

        iv_hagan.append((alpha_i, iv_list))

    return iv_hagan


# parameters
alpha = 0.3
nu = 0.9
rho = -0.85

# option information
f0 = 100.0
k_s = np.arange(40.0, 200.0, 10.0)
T = 1.0

# rho effect in the smile
style_markers = ['*', '.', 'x', '^']
rho_s = np.array([-0.9, -0.5, -0.25, -0.0])
out_rho_s = get_smile_for_differents_rho(alpha, rho_s, nu, k_s, f0, T)

no_outputs = len(out_rho_s)
for i in range(0, no_outputs):
    plt.plot(k_s, out_rho_s[i][1], label="rho="+str(rho_s[i]), linestyle='-', linewidth=0.5, color='black',
             marker=style_markers[i])


# epilon effect in the smile
# style_markers = ['*', '.', 'x', '^']
# nu_s = np.array([0.5, 0.7, 0.9, 1.2])
# out_nu_s = get_smile_for_differents_nu(alpha, rho, nu_s, k_s, f0, T)
#
# no_outputs = len(out_nu_s)
# for i in range(0, no_outputs):
#     plt.plot(k_s, out_nu_s[i][1], label="nu="+str(nu_s[i]), linestyle='-', linewidth=0.5, color='black',
#              marker=style_markers[i])

# alpha effect in the smile
# style_markers = ['*', '.', 'x', '^']
# alpha_s = np.array([0.4, 0.7, 0.8, 1.0])
# out_alpha_s = get_smile_for_differents_alpha(alpha_s, rho, nu, k_s, f0, T)
#
# no_outputs = len(out_alpha_s)
# for i in range(0, no_outputs):
#     plt.plot(k_s, out_alpha_s[i][1], label="alpha="+str(alpha_s[i]), linestyle='-', linewidth=0.5, color='black',
#              marker=style_markers[i])
#
#
# plt.xlabel("strike")
plt.legend()
plt.show()