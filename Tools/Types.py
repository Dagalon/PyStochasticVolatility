import numpy as np

from typing import List, NewType
from enum import Enum

Vector = List[float]
ndarray = NewType('ndarray', type(np.ndarray))
min_value = 1e-05

MIN_VALUE_LOG_MONEYNESS = 0.00001


class HESTON_OUTPUT(Enum):
    PATHS = 0,
    INTEGRAL_VARIANCE_PATHS = 1,
    DELTA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL = 2,
    GAMMA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL = 3,
    SPOT_VARIANCE_PATHS = 4,
    UNKNOWN = -1

    def __str__(self):
        return self.value


class RBERGOMI_OUTPUT(Enum):
    PATHS = 0,
    INTEGRAL_VARIANCE_PATHS = 1,
    SPOT_VOLATILITY_PATHS = 2,
    BS_BY_PATH = 3,
    UNKNOWN = -1

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
    GAMMA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL = 3,
    SIGMA_PATHS = 4,
    TIMES = 5,
    UNKNOWN = -1

    def __str__(self):
        return self.value


class LOCAL_VOL_OUTPUT(Enum):
    PATHS = 0,
    INTEGRAL_VARIANCE_PATHS = 1,
    SPOT_VARIANCE_PATHS = 2,
    UNKNOWN = -1

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


class ESTIMATOR_TYPE(Enum):
    INTEGRATED_VARIANCE_FOURIER = 1,
    INTEGRATED_VARIANCE_EMPIRICAL = 2,
    SPOT_VARIANCE_FOURIER = 3,
    UNKNOWN = -1

    def __str__(self):
        return self.value

