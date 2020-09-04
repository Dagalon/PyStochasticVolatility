from Solvers.PDE_Solver.Meshes import Mesh
from Tools.Types import ndarray
from typing import Callable


class TerminalCondition(object):
    def __init__(self, functional: Callable[[Mesh], ndarray]):
        self._functional = functional

    def get_value(self, mesh: Mesh) -> ndarray:
        return self._functional(mesh)

    def update(self, functional: Callable[[Mesh], ndarray]):
        self._functional = functional



