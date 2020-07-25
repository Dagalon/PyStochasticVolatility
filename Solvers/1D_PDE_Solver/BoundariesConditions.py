from abc import abstractmethod
from Types import BoundaryConditionType


class BoundaryCondition(object):
    def __init__(self, bc_type: BoundaryConditionType):
        self._bc_type = bc_type

    def get_type(self):
        return self._bc_type

    @abstractmethod
    def apply_boundary_condition(self, **kwargs):
        pass

    @abstractmethod
    def apply_boundary_condition_after_update(self, **kwargs):
        pass


class Zero_Laplacian_BC(BoundaryCondition):
    def __init__(self):
        BoundaryCondition.__init__(self, BoundaryConditionType.ZERO_LAPLACIAN)

    def apply_boundary_condition(self, **kwargs):
        operator = kwargs['operator']
        operator.diagonal()[0] = 0.0
        operator.diagonal()[-1] = 0.0

    def apply_boundary_condition_after_update(self, **kwargs):
        operator = kwargs['operator']
        mesh = operator.get_mesh()
        no_nodes = mesh.get_size()

        delta_r_0 = mesh.get_shift()[0]
        delta_l_0 = mesh.get_shift()[1]

        delta_r_N = mesh.get_shift()[-2]
        delta_l_N = mesh.get_shift()[-1]

        w_l_0 = 2.0 / (delta_l_0 * (delta_r_0 + delta_l_0))
        w_c_0 = - 2.0 / (delta_l_0 * delta_r_0)
        w_u_0 = 2.0 / (delta_r_0 * (delta_r_0 + delta_l_0))

        w_l_N = 2.0 / (delta_l_N * (delta_r_N + delta_l_N))
        w_c_N = - 2.0 / (delta_l_N * delta_r_N)
        w_u_N = 2.0 / (delta_r_N * (delta_r_N + delta_l_N))

        kwargs['u_t'][0] = - (w_c_0 / w_l_0) * kwargs['u_t'][1] - (w_u_0 / w_l_0) * kwargs['u_t'][2]
        kwargs['u_t'][no_nodes - 1] = - (w_l_N / w_u_N) * kwargs['u_t'][no_nodes - 3] - \
                                      (w_c_N / w_u_N) * kwargs['u_t'][no_nodes - 2]


class RobinCondition(BoundaryCondition):
    def __init__(self, mu: float):
        BoundaryCondition.__init__(self, BoundaryConditionType.ROBIN)
        self._mu = mu

    def apply_boundary_condition(self, **kwargs):
        pass

    def apply_boundary_condition_after_update(self, **kwargs):
        pass
