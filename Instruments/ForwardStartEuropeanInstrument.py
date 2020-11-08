from Tools.Types import TypeSellBuy, TypeEuropeanOption, ndarray
from MCPricers.EuropeanPricers import call_operator, put_operator
from Tools.Types import ndarray
from typing import Callable, List


class ForwardEuropeanPayoff(object):
    def __init__(self,
                 f_price: Callable[[List[float]], List[float]]):
        self._f_price = f_price

    def get_value(self, index_start_date: int,  x: ndarray):
        return self._f_price(x)


class ForwardStartEuropeanOption(object):
    def __init__(self,
                 strike: float,
                 notional: float,
                 buy_sell: TypeSellBuy,
                 option_type: TypeEuropeanOption,
                 spot: float,
                 forward_start_date: float,
                 delta_time: float):

        self._strike = strike
        self._forward_start_date = forward_start_date
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
            self._payoff = ForwardEuropeanPayoff(lambda x: mult_buy_sell * notional * call_operator(x, strike))
        else:
            self._payoff = ForwardEuropeanPayoff(lambda x: mult_buy_sell * notional * put_operator(x, strike))

    def update_strike(self, strike: float):
        self._strike = strike

    def get_price(self, x: ndarray) -> ndarray:
        return self._payoff.get_value(x)
