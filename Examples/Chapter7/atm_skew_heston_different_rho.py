import numpy as np

from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import Types
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
import matplotlib.pylab as plt

# Parameters
epsilon = 1.1
k = 0.5
rho = -0.9
v0 = 0.05
theta = 0.05

# options
d_t_i = np.arange(7, 737, 7) * (1.0 / 365.0)
no_maturities = len(d_t_i)
f0 = 100
T = 0.1
shift_spot = 0.01
notional = 1.0
options = []
for d_t in d_t_i:
    options.append(EuropeanOption(f0, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_t))

skew_rho_zero = []
skew_rho_no_zero = []

for i in range(0, no_maturities):
    # base
    price_rho_no_zero = options[i].get_analytic_value(0.0, theta, rho, k, epsilon, v0, 0.0,
                                                      model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_ATTARI,
                                                      compute_greek=False)

    price_rho_zero = options[i].get_analytic_value(0.0, theta, 0.0, k, epsilon, v0, 0.0,
                                                   model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_ATTARI,
                                                   compute_greek=False)

    iv_base_rho_no_zero = implied_volatility(price_rho_no_zero, f0, f0, d_t_i[i], 0.0, 0.0, 'c')
    iv_base_rho_zero = implied_volatility(price_rho_zero, f0, f0, d_t_i[i], 0.0, 0.0, 'c')

    # shift right
    options[i].update_strike(f0 * (1 + shift_spot))
    price_rho_no_zero_shift_right = options[i].get_analytic_value(0.0, theta, rho, k, epsilon, v0, 0.0,
                                                                  model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_ATTARI,
                                                                  compute_greek=False)

    price_rho_zero_shift_right = options[i].get_analytic_value(0.0, theta, 0.0, k, epsilon, v0, 0.0,
                                                               model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_ATTARI,
                                                               compute_greek=False)

    iv_base_rho_no_zero_shift_right = implied_volatility(price_rho_no_zero_shift_right, f0, f0 * (1 + shift_spot),
                                                         d_t_i[i], 0.0, 0.0, 'c')
    iv_base_rho_zero_shift_right = implied_volatility(price_rho_zero_shift_right, f0, f0 * (1 + shift_spot),
                                                      d_t_i[i], 0.0, 0.0, 'c')
    options[i].update_strike(f0 * (1 - shift_spot))

    # shift left
    options[i].update_strike(f0 * (1 - shift_spot))
    price_rho_no_zero_shift_left = options[i].get_analytic_value(0.0, theta, rho, k, epsilon, v0, 0.0,
                                                                 model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_ATTARI,
                                                                 compute_greek=False)

    price_rho_zero_shift_left = options[i].get_analytic_value(0.0, theta, 0.0, k, epsilon, v0, 0.0,
                                                              model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_ATTARI,
                                                              compute_greek=False)

    iv_base_rho_no_zero_shift_left = implied_volatility(price_rho_no_zero_shift_left, f0, f0 * (1 - shift_spot),
                                                        d_t_i[i], 0.0, 0.0, 'c')

    iv_base_rho_zero_shift_left = implied_volatility(price_rho_zero_shift_left, f0, f0 * (1 - shift_spot),
                                                     d_t_i[i], 0.0, 0.0, 'c')

    skew_rho_no_zero.append(f0 * (iv_base_rho_no_zero_shift_right - iv_base_rho_no_zero_shift_left) / (2.0 * shift_spot
                                                                                                       * f0))
    skew_rho_zero.append(f0 * (iv_base_rho_zero_shift_right - iv_base_rho_zero_shift_left) / (2.0 * shift_spot * f0))

plt.plot(d_t_i, skew_rho_no_zero, label="atm skew rho=%s" % rho, marker=".", linestyle="--", color="black")
plt.plot(d_t_i, skew_rho_zero, label="atm skew rho=0.0", marker="+", linestyle="--", color="black")

plt.legend()
plt.show()
