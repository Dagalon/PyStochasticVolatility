import numpy as np
from Tools.Types import TypeSellBuy, TypeEuropeanOption, ndarray
from MCPricers.ForwardStartEuropeanPricers import forward_start_call_operator, forward_start_put_operator
from Tools.Types import ndarray
from typing import Callable, List


class ForwardEuropeanPayoff(object):
    def __init__(self,
                 f_price: Callable[[int, List[float]], List[float]]):
        self._f_price = f_price

    def get_value(self, index_start_date: int, x: ndarray):
        return self._f_price(index_start_date, x)


class ForwardStartEuropeanOption(object):
    def __init__(self,
                 strike: float,
                 notional: float,
                 buy_sell: TypeSellBuy,
                 option_type: TypeEuropeanOption,
                 spot: float,
                 forward_start_time: float,
                 delta_time: float):

        self._strike = strike
        self._forward_start_time = forward_start_time
        self._forward_start_index = 0
        self._notional = notional
        self._option_type = option_type
        self._buy_sell = buy_sell
        self._spot = spot
        self._delta_time = delta_time

        if buy_sell == TypeSellBuy.BUY:
            mult_buy_sell = 1.0
        else:
            mult_buy_sell = -1.0

        if option_type == TypeEuropeanOption.CALL:
            self._payoff = ForwardEuropeanPayoff(lambda index_strike, x: mult_buy_sell * notional *
                                                 forward_start_call_operator(strike, index_strike, x))
        else:
            self._payoff = ForwardEuropeanPayoff(lambda index_strike, x: mult_buy_sell * notional *
                                                 forward_start_put_operator(strike, index_strike, x))

    def update_strike(self, strike: float):
        self._strike = strike

    def get_forward_start_date_index(self, sampling_dates: ndarray):
        index = np.searchsorted(sampling_dates, self._forward_start_time, side='right')
        if sampling_dates[index] == self._forward_start_time:
            self._forward_start_index = index

            return sampling_dates

        else:
            no_dates = len(sampling_dates)
            full_sampling_dates = np.zeros(no_dates)
            full_sampling_dates[index] = self._forward_start_time
            full_sampling_dates[0:index] = sampling_dates[0:index]
            full_sampling_dates[index:] = sampling_dates[index:]

            return full_sampling_dates

    def get_price(self, x: ndarray) -> ndarray:
        return self._payoff.get_value(self._forward_start_index, x)
