import numpy as np

from typing import List, NewType
from enum import Enum

Vector = List[float]
ndarray = NewType('ndarray', type(np.ndarray))
min_value = 1e-08


class HESTON_OUTPUT(Enum):
    PATHS = 0,
    INTEGRAL_VARIANCE_PATHS = 1,
    DELTA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL = 2,
    GAMMA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL = 3

    def __str__(self):
        return self.value


class ANALYTIC_MODEL(Enum):
    HESTON_MODEL_ATTARI = 0,
    HESTON_MODEL_REGULAR = 1,
    BLACK_SCHOLES_MODEL = 2,
    SABR_MODEL = 3,
    UNKNOWN = -1

    def __str__(self):
        return self.value


class SABR_OUTPUT(Enum):
    PATHS = 0,
    INTEGRAL_VARIANCE_PATHS = 1,
    DELTA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL = 2,
    GAMMA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL = 3

    def __str__(self):
        return self.value


class TYPE_STANDARD_NORMAL_SAMPLING(Enum):
    REGULAR_WAY = 1,
    ANTITHETIC = 2

    def __str__(self):
        return self.value


class TypeGreeks(Enum):
    DELTA = 0
    GAMMA = 1
    DUAL_DELTA = 2
    UNKNOWN = -1

    def __str__(self):
        return self.value


class TypeModel(Enum):
    ROUGH_BERGOMI = 0
    SABR = 1
    HESTON = 2
    BERGOMI_1F = 3,
    BERGOMI_2F = 4,
    UNKNOWN = -1

    def __str__(self):
        return self.value


class EULER_SCHEME_TYPE(Enum):
    STANDARD = 1
    LOG_NORMAL = 2
    UNKNOWN = -1

    def __str__(self):
        return self.value

