import abc
import numpy as np

from VolatilitySurface.Tools import SVITools, SABRTools


class ParametricImpliedVolatility(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @staticmethod
    def get_implied_volatility(*args, f=0.0, k=0.0, t=0.0):
        pass

    @staticmethod
    def get_variance(*args, f=0.0, k=0.0, t=0.0):
        pass

    @staticmethod
    def get_gradient_iv_to_parameters(*args, t):
        pass

    @staticmethod
    def get_derive_to_forward(*args, t):
        pass


class SVI(ParametricImpliedVolatility):

    def __init__(self):
        ParametricImpliedVolatility.__init__(self)
        pass

    @staticmethod
    def get_variance(*args, f=0.0, k=0.0):
        x = np.log(k / f)
        var = SVITools.svi_total_imp_var_jit(args[0], args[1], args[2], args[3], args[4], x)

        return var

    @staticmethod
    def get_implied_volatility(*args, f=0.0, k=0.0, t=0.0):
        return np.sqrt(SVI.get_variance(args[0], args[1], args[2], args[3], args[4], f, k) / t)

    @staticmethod
    def svi_total_imp_var(*args, z=0.0):
        return SVITools.svi_total_imp_var_jit(args[0], args[1], args[2], args[3], args[4], z)

    @staticmethod
    def get_gradient_iv_to_parameters(*args):
        return SVITools.get_gradient_svi_iv_to_parameters_jit(args[0], args[1], args[2], args[3], args[4])

    @staticmethod
    def get_derive_to_forward(*args, strike=0.0, t=0.0):
        # return SVITools.get_derive_svi_to_forward_jit(args[0], args[1], args[2], args[3], args[4], strike, t)
        pass


class SABR(ParametricImpliedVolatility):

    def __init__(self):
        ParametricImpliedVolatility.__init__(self)
        pass

    @staticmethod
    def get_implied_volatility(*args, f=0.0, k=0.0, t=0.0):
        return SABRTools.sabr_vol_jit(args[0], args[1], args[2], np.log(f / k), t)

    @staticmethod
    def get_variance(*args, f=0.0, k=0.0, t=None):
        return t * SABR.get_implied_volatility(args[0], args[1], args[2], f, k, t) ** 2

    @staticmethod
    def get_gradient_iv_to_parameters(*args, t=0.0):
        pass

    @staticmethod
    def get_derive_to_forward(*args, t=0.0):
        pass
