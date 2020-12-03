__author__ = 'David Garcia Lorite'

#
# Copyright 2020 David Garcia Lorite
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
#

import numpy as np

from Tools.Types import ndarray
from typing import Callable


def ode_euler_solver_malliavin(s: float,
                               t: float,
                               s0: float,
                               no_steps: int,
                               z_t: ndarray,
                               a_f: Callable[[float, float], float]):
    path = np.empty(z_t.shape)
    path[:, 0] = s0

    t_i = np.linspace(s, t, no_steps)
    delta_i = np.diff(t_i)

    for j in range(1, no_steps):
        path[:, j] = path[:, j - 1] + a_f(t_i[j - 1], z_t[:, j-1]) * delta_i[j - 1]

    return path
