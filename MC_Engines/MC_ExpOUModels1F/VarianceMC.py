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
