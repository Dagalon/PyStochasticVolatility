from Meshes import Mesh
from Types import np_ndarray
from typing import Callable


class TerminalCondition(object):
    def __init__(self, functional: Callable[[Mesh], np_ndarray]):
        self._functional = functional

    def get_value(self, mesh: Mesh) -> np_ndarray:
        return self._functional(mesh)



