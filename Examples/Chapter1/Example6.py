import numpy as np
import matplotlib.pylab as plt

from py_vollib.black_scholes_merton import black_scholes_merton
from py_vollib.black_scholes_merton.greeks import analytical

T = 2.0
no_time_steps = int(T * 365)
t_i_s = np.linspace(0.0, T, no_time_steps)
diff_t_i_s = np.diff(t_i_s)

# We will do the dynamic hedge of option call under BS model.
k = 100.0
spot = 120.0
r = 0.02
q = 0.01
sigma = 0.7
delta_time = T
notional = 1000.0

portfolio_t_i = np.zeros(no_time_steps)
option_t_i = np.zeros(no_time_steps)

alpha_t_i_1 = analytical.delta('c', spot, k, delta_time, r, sigma, q) * notional
option_t_i[0] = black_scholes_merton('c', spot, k, delta_time, r, sigma, q) * notional
beta_t = option_t_i[0] - alpha_t_i_1 * spot

portfolio_t_i[0] = (alpha_t_i_1 * spot + beta_t)

s_t_i_1 = spot
s_t_i = 0.0
alpha_t_i = 0.0
z_s = np.random.normal(0.0, 1.0, no_time_steps - 1)

rebalanced_index = list(map(lambda x: int(x), list(np.arange(0.0, no_time_steps, 30))))


for i in range(1, no_time_steps):
    z = np.random.normal(0.0, 1.0, 1)
    s_t_i = s_t_i_1 * np.exp((r - q) * diff_t_i_s[i - 1] - 0.5 * sigma * sigma * diff_t_i_s[i - 1] +
                             np.sqrt(diff_t_i_s[i - 1]) * sigma * z_s[i - 1])

    delta_time = T - t_i_s[i]
    option_t_i[i] = notional * black_scholes_merton('c', s_t_i, k, delta_time, r, sigma, q)

    if i in rebalanced_index:
        alpha_t_i = analytical.delta('c', s_t_i, k, delta_time, r, sigma, q) * notional
        beta_t = beta_t * (1 + r * diff_t_i_s[i - 1]) + alpha_t_i_1 * s_t_i_1 * q * diff_t_i_s[i - 1] + \
                 (alpha_t_i_1 - alpha_t_i) * s_t_i

        portfolio_t_i[i] = (alpha_t_i * s_t_i + beta_t)
        alpha_t_i_1 = alpha_t_i
    else:
        beta_t = beta_t * (1 + r * diff_t_i_s[i - 1]) + alpha_t_i_1 * s_t_i_1 * q * diff_t_i_s[i - 1]
        portfolio_t_i[i] = (alpha_t_i_1 * s_t_i + beta_t)

    s_t_i_1 = s_t_i

plt.plot(t_i_s, portfolio_t_i, label='hedge', linestyle='dashed', color='black', linewidth=1)
plt.plot(t_i_s, option_t_i, label='call option', color='black', linewidth=1)

plt.legend()
plt.title('Dynamic Hedge')
plt.show()
