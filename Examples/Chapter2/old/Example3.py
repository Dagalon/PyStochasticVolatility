import numpy as np
import matplotlib.pylab as plt

from MC_Engines.GenericSDE.SDE import bs_drift_flat, bs_sigma_flat
from MC_Engines.GenericSDE.SDESimulation import get_euler_step
from py_vollib.black_scholes_merton import black_scholes_merton
from py_vollib.black_scholes_merton.greeks import numerical
from functools import partial


def bs_european_option_hedging(t: float,
                               s0: float,
                               strike: float,
                               r: float,
                               q: float,
                               sigma: float,
                               option_type: str,
                               no_paths: int,
                               no_steps: int):
    t_i = np.linspace(0.0, t, no_steps)
    delta_i = np.diff(t_i)

    opt_t = np.zeros(shape=(no_paths, no_steps))
    opt_t[:, 0] = black_scholes_merton(option_type, s0, strike, t, r, sigma, q)

    s_t_j = s0 * np.ones(no_paths)
    s_t_j_1 = s0 * np.ones(no_paths)

    alpha_t = numerical.delta(option_type, s0, strike, t, r, sigma, q) * np.ones(no_paths)
    alpha_t_delta = numerical.delta(option_type, s0, strike, t, r, sigma, q) * np.ones(no_paths)

    beta_t = np.empty(shape=(no_paths, no_steps))
    beta_t[:, 0] = opt_t[:, 0] - np.multiply(alpha_t, s_t_j_1)

    portfolio_t = np.empty(shape=(no_paths, no_steps))
    portfolio_t[:, 0] = alpha_t * s_t_j + beta_t[:, 0]

    drift_t = partial(bs_drift_flat, rate_t=r, dividend_t=q)
    sigma_t = partial(bs_sigma_flat, sigma_t=sigma)
    acum_delta_t = 0

    for j in range(1, no_steps):
        z_j = np.random.standard_normal(no_paths)

        s_t_j = get_euler_step(s_t_j_1,
                               delta_i[j - 1],
                               z_j,
                               drift_t(t_i[j-1], s_t_j_1),
                               sigma_t(t_i[j-1], s_t_j_1))

        acum_delta_t += delta_i[j-1]

        for k in range(0, no_paths):
            portfolio_t[k, j] = alpha_t[k] * s_t_j[k] + beta_t[k, j-1] * (1 + r * delta_i[j-1])
            alpha_t_delta[k] = numerical.delta(option_type, s_t_j[k], strike, t - acum_delta_t, r, sigma, q)
            beta_t[k, j] = (1 + r * delta_i[j-1]) * beta_t[k, j-1] - (alpha_t_delta[k] - alpha_t[k]) * s_t_j[k] + \
                           q * delta_i[j-1] * alpha_t[k] * s_t_j[k]
            opt_t[k, j] = black_scholes_merton(option_type, s_t_j[k], strike, t - acum_delta_t, r, sigma, q)

        np.copyto(s_t_j_1, s_t_j)
        np.copyto(alpha_t, alpha_t_delta)

    return portfolio_t, opt_t, t_i


t = 10.0
s0 = 100.0
k = 90.0
r = 0.02
q = 0.01
sigma = 0.35
option_type = 'c'

no_paths = 5000
no_steps = 1000

portfolio_t, opt_t, t_i = bs_european_option_hedging(t, s0, k, r, q, sigma, option_type, no_paths, no_steps)

mean_portfolio_t = np.average(portfolio_t, axis=0)
mean_opt_t = np.average(opt_t, axis=0)

plt.plot(t_i, mean_portfolio_t, label='hedging without malliavin')
plt.plot(t_i, mean_opt_t, label='derivative')

plt.legend()
plt.title('Hedging evolution')
plt.show()
