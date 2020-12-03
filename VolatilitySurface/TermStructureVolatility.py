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

import abc
import QuantLib as ql
import numpy as np

from typing import List, Dict
from VolatilitySurface.IVParametric import SABR, SVI
from Tools import Types
from VolatilitySurface.Tools import ParameterTools, SABRTools


class ImpliedVolatilitySurface(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, value_date, day_counter):
        self._day_counter = day_counter
        self._value_date = value_date

    @property
    def day_counter(self):
        return self._day_counter

    @property
    def value_date(self):
        return self._value_date

    @abc.abstractmethod
    def get_impl_volatility(self, f=0.0, k=0.0, t=None):
        pass

    @abc.abstractmethod
    def get_local_vol_derivative(self, z_t: Types.ndarray, t=None):
        pass

    def get_parameters(self, *args):
        pass

    @classmethod
    def build_from_dict(cls, *args):
        pass


class SABRImpliedVolatilitySurface(ImpliedVolatilitySurface):
    def __init__(self, value_date, day_counter, rho_parameters: List[float], v_parameters: List[float],
                 vol_atm: Dict[int, float]):
        ImpliedVolatilitySurface.__init__(self, value_date, day_counter)

        self._parameters = {'rho': rho_parameters, 'nu': v_parameters}

        self._f_rho = lambda t: rho_parameters[0] + (rho_parameters[1] + rho_parameters[2] * t) * np.exp(
            -rho_parameters[3] * t)

        self._f_v = lambda t: v_parameters[0] + (v_parameters[1] + v_parameters[2] * t) * np.power(t, -v_parameters[3])

        self._date_atm_vol = list(vol_atm.keys())
        self._date_atm_vol.sort()

        self._vol_atm = vol_atm.copy()

    def __iter__(self):
        for key in self.__dict__.keys():
            if not key.__eq__("_f_v") and not key.__eq__("_f_rho"):
                if key.__eq__("_vol_atm"):
                    yield (key, dict([(str(item[0]), item[1]) for item in self.__getattribute__(key).items()]))
                elif key.__eq__("_parameters"):
                    yield (key, self.__getattribute__(key))
                elif key.__eq__("_day_counter"):
                    yield (key, str(self.__getattribute__(key)))
                elif key.__eq__("_value_date"):
                    ql_date = self.__getattribute__(key)
                    yield (key, str(ql_date.year()) + (str(ql_date.month()) if ql_date.month() > 9
                                                       else '0' + str(ql_date.month())) + str(ql_date.dayOfMonth()))

    @classmethod
    def build_from_dict(cls, *args):
        info_map = args[0]

        date_str = info_map['_value_date']
        parameters = info_map['_parameters']

        return cls(ql.Date(int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:])),
                   FromInfoToQL.get_type_ql_day_counter(info_map['_day_counter']),
                   parameters['rho'],
                   parameters['nu'],
                   dict([(int(item[0]), item[1]) for item in info_map['_vol_atm'].items()]))

    @property
    def value_date(self):
        return self._value_date

    @property
    def vol_atm(self):
        return self._vol_atm

    @property
    def day_counter(self):
        return self._day_counter

    def get_parameters(self, *args):
        return self._parameters

    def get_rho(self, t=None):
        delta_t_T = self._day_counter.yearFraction(self._value_date, t)
        return self._f_rho(delta_t_T)

    def get_v(self, t=None):
        delta_t_T = self._day_counter.yearFraction(self._value_date, t)
        return self._f_v(delta_t_T)

    def get_alpha(self, rho, v, t=None):
        delta_t_T = self._day_counter.yearFraction(self._value_date, t)
        atm_vol = self.get_atm_volatility(t)
        return ParameterTools.alpha_atm_sabr(rho, v, atm_vol[0], delta_t_T)

    def get_atm_volatility(self, date=None):
        int_date = date.serialNumber()

        # Index 0 is atm level volatility.
        # Index 1 is derivative atm volatility.

        out_atm_vol = np.zeros(2)

        if int_date <= self._date_atm_vol[0]:
            delta_t0_t = self._day_counter.yearFraction(self._value_date, date)
            out_atm_vol[0] = self._vol_atm[self._date_atm_vol[0]]
        elif int_date >= self._date_atm_vol[len(self._date_atm_vol) - 1]:
            out_atm_vol[0] = self._vol_atm[self._date_atm_vol[len(self._date_atm_vol) - 1]]
        else:
            closest_date = min(self._date_atm_vol, key=lambda x: abs(int_date - x))
            index = self._date_atm_vol.index(closest_date)

            if closest_date > int_date:
                index_upper = index
                index_lower = index - 1
            else:
                index_upper = index + 1
                index_lower = index

            closest_date_lower = self._date_atm_vol[index_lower]
            closest_date_upper = self._date_atm_vol[index_upper]

            delta_t0_t = self._day_counter.yearFraction(self._value_date, date)
            delta_T_i_1 = self._day_counter.yearFraction(self._value_date, ql.Date(closest_date_lower))
            delta_T_i_1_t = self._day_counter.yearFraction(ql.Date(closest_date_lower), date)
            delta_T_i_1_T_i = self._day_counter.yearFraction(ql.Date(closest_date_lower), ql.Date(closest_date_upper))
            delta_T_i = (delta_T_i_1 + delta_T_i_1_T_i)

            var_upper = np.power(self._vol_atm[closest_date_upper], 2.0) * delta_T_i
            var_lower = np.power(self._vol_atm[closest_date_lower], 2.0) * delta_T_i_1

            alpha_t = (delta_T_i_1_t / delta_T_i_1_T_i)
            interp_var = (1 - alpha_t) * var_lower + alpha_t * var_upper
            inter_sigma = np.sqrt(interp_var / delta_t0_t)
            out_atm_vol[0] = np.sqrt(interp_var / delta_t0_t)
            out_atm_vol[1] = (0.5 / delta_t0_t) * (
                        (var_upper - var_lower) / (delta_T_i_1_T_i * out_atm_vol[0]) - out_atm_vol[0])

        return out_atm_vol

    def get_impl_volatility(self, f=0.0, k=0.0, t=None):
        # SABR model suposses that the log-moneyness is log(F/K)
        delta_T = self._day_counter.yearFraction(self._value_date, t)
        rho = self._f_rho(delta_T)
        v = self._f_v(delta_T)
        sigma_atm = self.get_atm_volatility(t)[0]
        alpha = ParameterTools.alpha_atm_sabr(rho, v, sigma_atm, delta_T)

        return SABR.get_implied_volatility(alpha, rho, v, f=f, k=k, t=delta_T)

    def get_local_vol_derivative(self, z_t: Types.ndarray, t=None):
        delta_t_T = self._day_counter.yearFraction(self._value_date, t)
        rho_t = self._f_rho(delta_t_T)
        nu_t = self._f_v(delta_t_T)
        atm_vol_t = self.get_atm_volatility(t)[0]
        alpha_t = ParameterTools.alpha_atm_sabr(rho_t, nu_t, atm_vol_t, delta_t_T)

        return SABRTools.f_partial_der_parameters(z_t, delta_t_T, alpha_t, rho_t, nu_t)

    def get_index_neighbor_slices(self, t=None):
        if t <= self._date_atm_vol[0]:
            return None, 0

        elif t >= self._date_atm_vol[-1]:
            return len(self._slice_date) - 1, None

        else:
            index_right = np.array(self._date_atm_vol).searchsorted(t, side='right')
            return index_right - 1, index_right


class SVIImpliedVolatilitySurface(ImpliedVolatilitySurface):
    def __init__(self, value_date, day_counter, parameters: Dict[int, List[float]], vol_atm: Dict[int, float]):
        ImpliedVolatilitySurface.__init__(self, value_date, day_counter)

        self._parameters = parameters

        self._slice_date = list(parameters.keys())
        self._slice_date.sort()
        self._vol_atm = vol_atm.copy()

    def __iter__(self):
        for key in self.__dict__.keys():
            if key.__eq__("_vol_atm"):
                yield (key, dict([(str(item[0]), item[1]) for item in self.__getattribute__(key).items()]))
            elif key.__eq__("_parameters"):
                yield (key, dict([(str(item[0]), item[1].tolist()) for item in self.__getattribute__(key).items()]))
            elif key.__eq__("_day_counter"):
                yield (key, str(self.__getattribute__(key)))
            elif key.__eq__("_value_date"):
                ql_date = self.__getattribute__(key)
                yield (key, str(ql_date.year()) + (str(ql_date.month()) if ql_date.month() > 9
                                                   else '0' + str(ql_date.month())) + str(ql_date.dayOfMonth()))

    @classmethod
    def build_from_dict(cls, *args):
        info_map = args[0]

        date_str = info_map['_value_date']
        parameters = info_map['_parameters']

        return cls(ql.Date(int(date_str[6:]), int(date_str[4:6]), int(date_str[0:4])),
                   FromInfoToQL.get_type_ql_day_counter(info_map['_day_counter']),
                   parameters,
                   dict([(int(item[0]), item[1]) for item in info_map['_vol_atm'].items()]))

    @property
    def value_date(self):
        return self._value_date

    @property
    def vol_atm(self):
        return self._vol_atm

    @property
    def day_counter(self):
        return self._day_counter

    @property
    def parameters(self):
        return self._parameters

    def interp_volatility(self, z=0.0, date=None):
        int_date = date.serialNumber()

        if int_date <= self._slice_date[0]:
            params_t0 = self._parameters[self._slice_date[0]]
            T = self.day_counter.yearFraction(self._value_date, date)
            var_svi_t = SVI.svi_total_imp_var(params_t0[0], params_t0[1], params_t0[2], params_t0[3], params_t0[4], z=z)
            interp_vol = np.sqrt(var_svi_t / T)

        elif int_date >= self._slice_date[len(self._slice_date) - 1]:
            params_tn = self._parameters[self._slice_date[len(self._slice_date) - 1]]
            T = self.day_counter.yearFraction(self._value_date, date)
            interp_vol = np.sqrt(SVI.svi_total_imp_var(params_tn[0], params_tn[1], params_tn[2], params_tn[3],
                                                       params_tn[4], z=z) / T)

        else:
            closest_date = min(self._slice_date, key=lambda x: abs(int_date - x))
            index = self._slice_date.index(closest_date)

            if closest_date > int_date:
                index_upper = index
                index_lower = index - 1
            else:
                index_upper = index + 1
                index_lower = index

            closest_date_lower = self._slice_date[index_lower]
            closest_date_upper = self._slice_date[index_upper]

            params_lower = self._parameters[closest_date_lower]
            params_upper = self._parameters[closest_date_upper]

            delta_t = self._day_counter.yearFraction(self._value_date, date)
            delta_T_i_1 = self._day_counter.yearFraction(self._value_date, ql.Date(closest_date_lower))
            delta_T_i = self._day_counter.yearFraction(self._value_date, ql.Date(closest_date_upper))

            delta_T_i_1_t = self._day_counter.yearFraction(ql.Date(closest_date_lower), date)
            delta_T_i_1_T_i = self._day_counter.yearFraction(ql.Date(closest_date_lower), ql.Date(closest_date_upper))

            alpha_t = delta_T_i_1_t / delta_T_i_1_T_i

            var_t_i_1 = SVI.svi_total_imp_var(params_lower[0], params_lower[1], params_lower[2], params_lower[3],
                                              params_lower[4], z=z)

            var_t_i = SVI.svi_total_imp_var(params_upper[0], params_upper[1], params_upper[2], params_upper[3],
                                            params_upper[4], z=z)

            interp_var = (1.0 - alpha_t) * var_t_i_1 + alpha_t * var_t_i
            interp_vol = np.sqrt(interp_var / delta_t)

        return interp_vol

    def get_impl_volatility(self, f=0.0, k=0.0, t=None):
        # SVI model suposses that the log-moneyness is log(K/F)
        z = np.log(k / f)
        return self.interp_volatility(z, t)

    def get_index_neighbor_slices(self, t=None):
        if t <= self._slice_date[0]:
            return None, 0

        elif t >= self._slice_date[-1]:
            return len(self._slice_date) - 1, None

        else:
            index_right = np.array(self._slice_date).searchsorted(t, side='right')
            return index_right - 1, index_right

    def get_local_vol_derivative(self, z_t: Types.ndarray, t=None):
        pass
