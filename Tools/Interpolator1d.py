import numba as nb
import numpy as np

from Tools import Types


# @nb.jit("f8[:](f8[:],f8[:],f8)")
# def hermite_cubic_spline(x: Types.ndarray, y: Types.ndarray, z_i: float):
#     no_elements = len(x)
#     index_right = np.searchsorted(x,z_i, side='right')
#     index_left = index_right - 1
#     c_i_s = np.empty(4)
#     c_i_s[0] = y[index_left]
#
#     if index_right == (no_elements - 1):
#
#     elif index_right == 1:
#         l_i = (x[index_right] - x[index_left])
#         l_i_1 = (x[index_left] - x[index_left - 1])
#         s_i =
#     else: