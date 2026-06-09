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

from typing import List, NewType
from enum import Enum

Vector = List[float]
ndarray = NewType('ndarray', np.ndarray)
min_value = 1e-05

MIN_VALUE_LOG_MONEYNESS = 0.00001


class HESTON_OUTPUT(Enum):
    """Enumerate the data arrays produced by the Heston Monte Carlo engine."""
    PATHS = 0,
    INTEGRAL_VARIANCE_PATHS = 1,
    DELTA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL = 2,
    GAMMA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL = 3,
    SPOT_VARIANCE_PATHS = 4,
    TIMES = 5,
    UNKNOWN = -1

    def __str__(self):
        """Return the underlying integer value as a string."""
        return self.value


class RBERGOMI_OUTPUT(Enum):
    """Output identifiers used by the rough Bergomi engines."""
    PATHS = 0,
    INTEGRAL_VARIANCE_PATHS = 1,
    SPOT_VOLATILITY_PATHS = 2,
    BS_BY_PATH = 3,
    TIMES = 4,
    VARIANCE_SPOT_PATHS = 5,
    INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS = 6,
    UNKNOWN = -1

    def __str__(self):
        """Return the underlying integer value as a string."""
        return self.value


class MIXEDLOGNORMAL_OUTPUT(Enum):
    """Describe the arrays produced by the mixed log-normal engine."""
    PATHS = 0,
    INTEGRAL_VARIANCE_PATHS = 1,
    SPOT_VARIANCE_PATHS = 2,
    TIMES = 3,
    UNKNOWN = -1


class BERGOMI2F_OUTPUT(Enum):
    """Enumerate outputs for the two-factor Bergomi engine."""
    PATHS = 0,
    INTEGRAL_VARIANCE_PATHS = 1,
    SPOT_VARIANCE_PATHS = 2,
    TIMES = 3,
    UNKNOWN = -1

    def __str__(self):
        """Return the underlying integer value as a string."""
        return self.value


class ANALYTIC_MODEL(Enum):
    """List the available analytic pricing models."""
    HESTON_MODEL_ATTARI = 0,
    HESTON_MODEL_REGULAR = 1,
    HESTON_MODEL_LEWIS = 2,
    BATES_MODEL_LEWIS = 3,
    BLACK_SCHOLES_MODEL = 4,
    SABR_MODEL = 5,
    UNKNOWN = -1

    def __str__(self):
        """Return the underlying integer value as a string."""
        return self.value


class SABR_OUTPUT(Enum):
    """Enumerate the arrays emitted by the SABR Monte Carlo engines."""
    PATHS = 0,
    INTEGRAL_VARIANCE_PATHS = 1,
    DELTA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL = 2,
    GAMMA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL = 3,
    SIGMA_PATHS = 4,
    TIMES = 5,
    INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS = 6,
    INTEGRAL_SIGMA_PATHS = 7,
    VARIANCE_PATHS = 8,
    UNKNOWN = -1

    def __str__(self):
        """Return the underlying integer value as a string."""
        return self.value


class LOCAL_VOL_OUTPUT(Enum):
    """Identify outputs produced by local volatility simulations."""
    PATHS = 0,
    INTEGRAL_VARIANCE_PATHS = 1,
    SPOT_VARIANCE_PATHS = 2,
    TIMES = 3,
    UNKNOWN = -1


class CHEYETTE_OUTPUT(Enum):
    """Outputs available from the Cheyette interest-rate engine."""
    PATHS_X = 0,
    PATHS_Y = 1,
    RATE = 2,
    BANK_ACCOUNT = 3


def __str__(self):
    """Return the underlying integer value as a string."""
    return self.value


class TYPE_STANDARD_NORMAL_SAMPLING(Enum):
    """Specify whether to use regular or antithetic normal sampling."""
    REGULAR_WAY = 1,
    ANTITHETIC = 2

    def __str__(self):
        """Return the underlying integer value as a string."""
        return self.value


class TypeGreeks(Enum):
    """Greeks supported by the Monte Carlo estimators."""
    DELTA = 0
    GAMMA = 1
    DUAL_DELTA = 2
    UNKNOWN = -1

    def __str__(self):
        """Return the underlying integer value as a string."""
        return self.value


class TypeModel(Enum):
    """Supported stochastic volatility model identifiers."""
    ROUGH_BERGOMI = 0
    SABR = 1
    HESTON = 2
    BERGOMI_1F = 3,
    BERGOMI_2F = 4,
    UNKNOWN = -1

    def __str__(self):
        """Return the underlying integer value as a string."""
        return self.value


class EULER_SCHEME_TYPE(Enum):
    """Euler discretisation choices for Monte Carlo engines."""
    STANDARD = 1
    LOG_NORMAL = 2
    UNKNOWN = -1

    def __str__(self):
        """Return the underlying integer value as a string."""
        return self.value


class ESTIMATOR_TYPE(Enum):
    """Estimator variants supported by the variance routines."""
    INTEGRATED_VARIANCE_FOURIER = 1,
    INTEGRATED_VARIANCE_EMPIRICAL = 2,
    SPOT_VARIANCE_FOURIER = 3,
    UNKNOWN = -1

    def __str__(self):
        """Return the underlying integer value as a string."""
        return self.value


class TypeEuropeanOption(Enum):
    """Call or put flag used by analytic and Monte Carlo pricers."""
    CALL = 1
    PUT = -1

    def __str__(self):
        """Return the underlying integer value as a string."""
        return self.value


class TypeSellBuy(Enum):
    """Direction of a trade when quoting payoffs."""
    SELL = -1
    BUY = 1

    def __str__(self):
        """Return the underlying integer value as a string."""
        return self.value
