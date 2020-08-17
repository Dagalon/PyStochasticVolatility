from enum import Enum
from typing import NewType
from numpy import ndarray

np_ndarray = NewType('np_ndarray', type(ndarray))


class BoundaryConditionType(Enum):
    UNKNOWN = -1,
    ZERO_LAPLACIAN = 0,
    ZERO_DIFFUSION = 1,
    ZERO_GRADIENT = 2,
    ROBIN = 3

    def __str__(self):
        return self.name


class SchemeType(Enum):
    UNKNOWN = -1,
    IMPLICIT = 0,
    EXPLICIT = 1,
    THETA = 2
    CRANK_NICOLSON = 3

    def __str__(self):
        return self.name


