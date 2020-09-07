import abc
import QuantLib as ql
import numpy as np

from IV_Calibrators.Core_IV_Calibrators.SurfaceVolatility.TermStructureVolatility import ImpliedVolatilitySurface
from IV_Calibrators.Core_IV_Calibrators.SpecialTypes import GeneralTypes
from IV_Calibrators.Core_IV_Calibrators.SurfaceVolatility.Tools import SabrTools, SVITools


class LocalVol(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, iv_surface: ImpliedVolatilitySurface):
        self._iv_surface = iv_surface
        self._day_counter = iv_surface.day_counter

    @abc.abstractmethod
    def get_vol(self, t: int, x_t: GeneralTypes.nd_array, f0_t):
        pass

    @abc.abstractmethod
    def get_pathwise_derive(self, t: float, x_t: GeneralTypes.nd_array):
        pass

    def update_iv_surface(self, iv_surface: ImpliedVolatilitySurface):
        self._iv_surface = iv_surface
        self._day_counter = iv_surface.day_counter


class SABRLocalVol(LocalVol):
    def __init__(self, iv_surface: ImpliedVolatilitySurface):
        LocalVol.__init__(self, iv_surface)

    def get_vol(self, t: int, x_t: GeneralTypes.nd_array, f0_t):
        d_t = self._day_counter.yearFraction(self._iv_surface._value_date, ql.Date(t))

        if d_t > 0:
            parameters = self._iv_surface.get_parameters()
            atm_info = self._iv_surface.get_atm_volatility(ql.Date(t))

            return SabrTools.get_sabr_loc_vol(np.array(parameters['nu']),  # Esto hay que modificarlo
                                              np.array(parameters['rho']),  # Esto hay que modificarlo
                                              atm_info[0],
                                              atm_info[1],
                                              d_t,
                                              f0_t,
                                              x_t)
        else:
            return np.zeros(x_t.shape)

    def get_pathwise_derive(self, t: float, x_t: GeneralTypes.nd_array):
        pass


class SVILocalVol(LocalVol):
    def __init__(self, iv_surface: ImpliedVolatilitySurface):
        LocalVol.__init__(self, iv_surface)

    def get_vol(self, t: int, x_t: GeneralTypes.nd_array, f0_t):
        d_t = self._day_counter.yearFraction(self._iv_surface._value_date, ql.Date(t))
        index = self._iv_surface.get_index_neighbor_slices(t)
        dates = self._iv_surface._slice_date

        if index[0] is None:
            t_j = dates[0]
            p_i = self._iv_surface._parameters[t_j]
            p_i_1 = np.empty(0)
            d_t_j_1 = 0.0
            d_t_j = self._day_counter.yearFraction(self._iv_surface._value_date, ql.Date(t_j))
        elif index[1] is None:
            t_j_1 = dates[-1]
            p_i_1 = self._iv_surface._parameters[t_j_1]
            p_i = np.empty(0)
            d_t_j_1 = self._day_counter.yearFraction(self._iv_surface._value_date, ql.Date(t_j_1))
            d_t_j = 0.0
        else:
            t_j_1 = dates[index[0]]
            t_j = dates[index[1]]
            d_t_j_1 = self._day_counter.yearFraction(self._iv_surface._value_date, ql.Date(t_j_1))
            d_t_j = self._day_counter.yearFraction(self._iv_surface._value_date, ql.Date(t_j))
            p_i_1 = self._iv_surface._parameters[t_j_1]
            p_i = self._iv_surface._parameters[t_j]

        return SVITools.get_svi_loc_vol(p_i_1, p_i, d_t_j_1, d_t, d_t_j, f0_t, x_t)

    def get_pathwise_derive(self, t: float, x_t: GeneralTypes.nd_array):
        pass
