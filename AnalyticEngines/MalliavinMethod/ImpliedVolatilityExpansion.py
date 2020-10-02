import numpy as np
import numba as nb

from Tools import Types


def vol_swap_approximation(parameters: Types.ndarray, model_type: Types.TypeModel):
    if model_type == Types.TypeModel.SABR:
        pass
    elif model_type == Types.TypeModel.HESTON:
        pass
    elif model_type == Types.TypeModel.ROUGH_BERGOMI:
        pass
    elif model_type == Types.TypeModel.BERGOMI_1F:
        pass
    else:
        print(f'At the moment we have not implemented the expansion implied vol for {str(model_type)} ')


def get_atm_iv_approximation(parameters: Types.ndarray, model_type: Types.TypeModel, t: float):
    if model_type == Types.TypeModel.SABR:
        pass
    elif model_type == Types.TypeModel.HESTON:
        pass
    elif model_type == Types.TypeModel.ROUGH_BERGOMI:
        pass
    elif model_type == Types.TypeModel.BERGOMI_1F:
        pass
    else:
        print(f'At the moment we have not implemented the expansion implied vol for {str(model_type)} ')
