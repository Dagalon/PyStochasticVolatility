import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

from MC_Engines.MC_LocalVol import LocalVolFunctionals
from VolatilitySurface.Tools import SABRTools

# info
f0 = 100.0
tis = np.linspace(0.01, 1.0, 40)
h = 0.5
epsilon = 0.5
spreads = np.linspace(-epsilon, epsilon, 3)
strikes = [f0 + s for s in spreads]

# sabr parameters
alpha = 0.3
nu = 0.6
rho = -0.6
parameters = [alpha, nu, rho]

# skews
skew_iv = []
skew_lv = []
skew_log_lv = []
ratio_skew = []

# curvatures
curvature_iv = []
curvature_log_iv = []
curvature_lv = []
curvature_log_lv = []
curvature_lv_rho_zero = []
ratio_curvatures = []

# skew for sabr formula
zi = np.log(np.divide(f0, strikes))

for t in tis:
    iv_hagan = [SABRTools.sabr_vol_jit(alpha, rho, nu, z, t) for z in zi]
    skew_iv.append(0.5 * (iv_hagan[2] - iv_hagan[0]) / h)
    curvature_iv.append((iv_hagan[2] - 2.0 * iv_hagan[1] + iv_hagan[0]) / (h * h))
    curvature_log_iv.append(curvature_iv[-1] * f0 * f0 + skew_iv[-1] * f0)

# skew for local vol equivalent
ks = np.array(strikes)

for t in tis:
    lv_output = LocalVolFunctionals.derivatives_local_vol_log_normal_sabr(t, ks, f0, alpha, rho, nu)
    lv_output_rho_zero = LocalVolFunctionals.derivatives_local_vol_log_normal_sabr(t, ks, f0, alpha, 0.0, nu)
    skew_lv.append(lv_output[1, 1])
    curvature_lv.append(lv_output[1, 2])
    curvature_lv_rho_zero.append(lv_output_rho_zero[1, 2])

    # log local vol curvature
    skew_log_lv.append(skew_lv[-1] * f0)
    curvature_log_lv.append(curvature_lv[-1] * f0 * f0 + skew_lv[-1] * f0)

target_skew = [1.0 / (h + 1.5) for i in range(0, len(tis))]

# ratio skew
ratio_skew = [skew_iv[k] / skew_lv[k] for k in range(0, len(skew_lv))]

# ratio curvature
ratio_curvatures = [curvature_log_iv[k] / curvature_log_lv[k] for k in range(0, len(skew_lv))]
diff_log_curvatures = [curvature_log_iv[k] - (1.0 / 3.0) * curvature_log_lv[k] for k in range(0, len(skew_lv))]

target_curvature = [1.0 / (2.0 * (h + 1.0)) for i in range(0, len(tis))]
target_difference = [- np.power(nu * rho, 2.0) / (6.0 * alpha) for i in range(0, len(tis))]


# skew
# plt.scatter(tis, ratio_skew, label="skew_iv / skew_lv", color="blue", marker="o")
# plt.plot(tis, target_skew, label="1/(H + 3/2)", color="red", linestyle="dotted", marker="x")

# curvature
# plt.plot(tis, curvature_lv, label="rho=0", color="blue", marker="x")
# plt.plot(tis, curvature_lv_rho_zero, label="rho=0.6", color="green", marker="o")
# plt.scatter(tis, ratio_curvatures, label="curvature_iv / curvature_lv", color="black", marker="o")
# plt.plot(tis, target_curvature, label="1/3", color="black", linestyle="dotted", marker="x")

plt.scatter(tis, diff_log_curvatures, label="curvature_iv - curvature_lv / 3", color="black", marker="o")


def f_law(x, a, b):
    return a + b * x


# skew
popt, pcov = curve_fit(f_law, tis, diff_log_curvatures)
diff_fit = f_law(tis, *popt)
plt.plot(tis, diff_fit, label="%s+%st" % (round(popt[0], 5), round(popt[1], 5)), color="black",
          linestyle="dashdot", marker=".")

# plt.scatter(tis, target_difference, label="((rho*nu)^2)/(6*alpha)", color="black", linestyle="dotted", marker="x")

# plt.plot(tis, curvature_log_lv, label="curvature_lv", color="black", marker="o")
# plt.plot(tis, curvature_log_iv, label="curvature_iv", color="black", linestyle="dotted", marker="x")

# plt.ylim(0.49, 0.51)

plt.legend()
plt.show()
