import numpy as np
import numba as nb

from Tools.Types import ndarray
from MC_Engines.MC_Heston import HestonTools
from ncephes import ndtri


def get_variance(hurst_parameter: float,
                 t_i_1: float,
                 t_i: float,
                 v_t_i_1: ndarray,
                 u_i: ndarray):

    return 0.0