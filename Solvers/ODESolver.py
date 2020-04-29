import numpy as np

from Type import nd_array
from typing import Callable


def ode_euler_solver_malliavin(s: float,
                               t: float,
                               s0: float,
                               no_steps: int,
                               z_t: nd_array,
                               a_f: Callable[[float, float], float]):
    path = np.empty(z_t.shape)
    path[:, 0] = s0

    t_i = np.linspace(s, t, no_steps)
    delta_i = np.diff(t_i)

    for j in range(1, no_steps):
        path[:, j] = path[:, j - 1] + a_f(t_i[j - 1], z_t[:, j-1]) * delta_i[j - 1]

    return path
