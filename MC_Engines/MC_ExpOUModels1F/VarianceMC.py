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
# See the License for the specific language governing permissions and limitations under the License.
#

import numpy as np
import numba as nb

from Tools.Types import ndarray


def get_variance(nu: float,
                 v: float,
                 t_i_1: float,
                 t_i: float,
                 v_t_i_1: ndarray,
                 u_i: ndarray,
                 no_paths: int):

    paths = np.zeros(no_paths)

    for i in range(0, no_paths):
        pass
