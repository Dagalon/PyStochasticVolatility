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

from typing import Callable


class hagan_loc_vol(object):
    def __init__(self, a_t: Callable[[float], float],
                 loc_vol: Callable[[float], float],
                 loc_vol_first_derive: Callable[[float], float],
                 loc_vol_second_derive: Callable[[float], float]):
        self._a_t = a_t
        self._loc_vol = loc_vol
        self._loc_vol_derive = loc_vol_first_derive
        self._loc_vol_second_derive = loc_vol_second_derive

    def get_implied_vol(self, t: float, f0: float, k: float) -> float:
        f_av = 0.5 * (f0 + k)
        loc_vol_value = self._loc_vol(f_av)
        r1 = self._loc_vol_derive(f_av) / loc_vol_value
        r2 = self._loc_vol_second_derive(f_av) / loc_vol_value
        a_t = self._a_t(t)

        multiplier = a_t * loc_vol_value / f_av
        order0 = (1.0 / 24.0) * (2.0 * r2 - r1 * r1 + 1.0 / (f_av * f_av)) * np.power(a_t * loc_vol_value, 2.0) * t
        order1 = (1.0 / 24.0) * (r2 - 2.0 * r1 * r1 + 2.0 / (f_av * f_av))

        return multiplier * (1.0 + order0 + order1 * (f0 - k) * (f0 - k))

    def update_a(self, a_t: Callable[[float], float]):
        self._a_t = a_t

    def loc_vol(self,
                loc_vol: Callable[[float], float],
                loc_vol_first_derive: Callable[[float], float],
                loc_vol_second_derive: Callable[[float], float]):

        self._loc_vol = loc_vol
        self._loc_vol_derive = loc_vol_first_derive
        self._loc_vol_second_derive = loc_vol_second_derive
