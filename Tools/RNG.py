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
import numba as nb
import sobol_seq as sobol

from Tools.Types import TYPE_STANDARD_NORMAL_SAMPLING
# from ncephes import ndtri
from scipy.special import ndtri


@nb.jit("f8[:](f8,f8,f8[:])", nopython=False)
# @nb.jit(nopython=True)
def norm_inv(mu, sigma, z):
    size_z = len(z)
    normal_narray = np.empty(size_z)

    for i in range(0, size_z):
        normal_narray[i] = ndtri(z[i])

    return normal_narray


class RndGenerator(object):
    def __init__(self,
                 initial_seed: int):
        self._seed = initial_seed
        self._rnd_generator = np.random.RandomState(initial_seed)

    @property
    def rnd_generator(self):
        return self._rnd_generator

    def set_seed(self, seed):
        self._rnd_generator.seed(seed)

    def uniform(self,
                a=0.0,
                b=1.0,
                size=None):

        return self._rnd_generator.uniform(a, b, size)

    def normal(self,
               mu=0.0,
               sigma=1.0,
               size=None,
               sampling_type=TYPE_STANDARD_NORMAL_SAMPLING.REGULAR_WAY):

        if sampling_type == TYPE_STANDARD_NORMAL_SAMPLING.REGULAR_WAY:
            return self._rnd_generator.normal(mu, sigma, size)
        else:
            if type(size) is tuple:
                no_elements = size[0] * size[1]
                first_part_rn = self._rnd_generator.normal(mu, sigma, int(0.5 * no_elements))
                return np.concatenate((first_part_rn, - first_part_rn), axis=0).reshape(size)

            else:
                first_part_rn = self._rnd_generator.normal(mu, sigma, int(0.5 * size))
                return np.concatenate((first_part_rn, - first_part_rn), axis=0).reshape(size)

    @staticmethod
    def normal_sobol(mu=0.0,
                     sigma=1.0,
                     size=None):

        if type(size) is tuple:
            m = size[1]
            n = size[0]
            return mu + sigma * norm_inv(0.0, 1.0, sobol.i4_sobol_generate(m, n))

        elif type(size) is int:
            return mu + sigma * norm_inv(0.0, 1.0, np.ndarray.flatten(sobol.i4_sobol_generate(1, size)))

        else:
            raise ValueError('the value for size %i is wrong, it must be tuple of int' % size)

    @staticmethod
    def uniform_sobol(a=0.0,
                      b=1.0,
                      size=None):

        if type(size) is tuple:
            m = size[1]
            n = size[0]
            return a + (b - a) * sobol.i4_sobol_generate(m, n)

        elif type(size) is int:
            return a + (b - a) * np.ndarray.flatten(sobol.i4_sobol_generate(1, size))

        else:
            raise ValueError('the value for size %i is wrong, it must be tuple of int' % size)
