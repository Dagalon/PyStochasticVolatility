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

import numba as nb
import numpy as np


@nb.jit("f8(f8,f8,f8,f8,f8)", nopython=True)
def local_vol_from_variance(z, w_t, w_partial_z, w_partial_zz, w_partial_t):
    if w_t > 0:
        ratio_y_w = z / w_t
        # c_partial_z = - 0.25 - 1.0 / w_t + ratio_y_w * ratio_y_w
        c_partial_z =  0.25 + 1.0 / w_t
        # loc_vol_denominator = 1.0 - ratio_y_w * w_partial_z + 0.25 * c_partial_z * w_partial_z * w_partial_z \
        #                     + 0.5 * w_partial_zz

        loc_vol_denominator = np.power(1.0 - 0.5 * ratio_y_w * w_partial_z, 2.0) - \
                              0.25 * c_partial_z * w_partial_z * w_partial_z + 0.5 * w_partial_zz

        return w_partial_t / loc_vol_denominator
    else:
        return 0.0


@nb.jit("f8(f8,f8,f8,f8,f8,f8,f8)", nopython=True)
def local_vol_from_implied_vol(t, sigma_i, z, k, sigma_partial_t, sigma_partial_k, sigma_partial_k_k):
    sqrt_t = np.sqrt(t)
    sigma_i_t = sigma_i * sqrt_t
    d_1 = (z / sigma_i_t) + 0.5 * sigma_i_t

    numerator_loc_vol = (sigma_i / t) + 2.0 * sigma_partial_t

    denominator_loc_vol = k * k * (sigma_partial_k_k - d_1 * sqrt_t * sigma_partial_k * sigma_partial_k +
                         (1.0 / sigma_i) * np.power(1.0 / (k * sqrt_t) + d_1 * sigma_partial_k, 2.0))

    return np.sqrt(numerator_loc_vol / denominator_loc_vol)
