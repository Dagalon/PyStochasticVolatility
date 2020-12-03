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

from typing import Callable, List
from numpy import linspace


def uniform_mesh(no_points: int, T0: float, T1: float):
    return linspace(T0, T1, no_points).tolist()


class Mesh(object):
    def __init__(self,
                 generator: Callable[[int, float, float], List[float]],
                 T0: float,
                 T1: float,
                 no_points: int):

        self._generator = generator
        self._T0 = T0
        self._T1 = T1
        self._no_points = no_points
        self._points = generator(no_points, T0, T1)

    @property
    def left_boundary(self):
        return self._T0

    @property
    def right_boundary(self):
        return self._T1

    @property
    def nodes(self):
        return self._points

    def update(self, no_points: int):
        self._points = self._generator(no_points, self.left_boundary, self.right_boundary)
