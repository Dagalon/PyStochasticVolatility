import os
import pandas as pd
import QuantLib as ql
import numpy as np

from pathlib import Path
from VolatilitySurface import TermStructureVolatility
from VolatilitySurface.Tools import SABRTools, ParameterTools
from AnalyticEngines.LocalVolatility.Dupire import NonParametricLV

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
sabr_parameter_paths = os.path.join(folder_directory, 'Data', 'SabrParameters.txt')
vol_atm_path = os.path.join(folder_directory, 'Data', 'VolAtmSabr.txt')

parameters = pd.read_csv(sabr_parameter_paths, header=None, names=["parameter", "rho", "nu"], sep=";")
vol_atm = pd.read_csv(vol_atm_path, header=None, names=["value_date", "date", "vol"], sep=";")
no_dates = len(vol_atm['date'])

# We must build the term structure volatility
rho_parameter = list(parameters["rho"])
rho_parameter.remove("rho")
nu_parameter = list(parameters["nu"])
nu_parameter.remove("nu")

vol_atm_list = list(vol_atm["vol"])
dates_list = list(vol_atm["date"])

vol_atm_map = dict([(int(dates_list[i]), float(vol_atm_list[i])) for i in range(0, no_dates)])
value_date = list(vol_atm["value_date"])[0]
ql_date = ql.Date(int(value_date))
day_counter = ql.Actual365Fixed()

sabr_term_structure_vol = TermStructureVolatility.SABRImpliedVolatilitySurface(value_date, day_counter, rho_parameter, nu_parameter, vol_atm)
loc_vol = NonParametricLV.LocalVol(sabr_term_structure_vol)

no_z_i_s = 50
z_i_s = np.linspace(-1.0, 1.0, no_z_i_s)

m_iv = np.zeros( shape=(no_dates, no_z_i_s))
m_loc_vol = np.zeros(shape=(no_dates, no_z_i_s))

p_rho = np.array(list(map(lambda x: float(x), rho_parameter)))
p_nu = np.array(list(map(lambda x: float(x), nu_parameter)))

for i in range(0, no_dates):
    delta_time = day_counter.yearFraction(ql_date, ql.Date(int(dates_list[i])))
    rho_i = ParameterTools.rho_sabr(p_rho, delta_time)
    nu_i = ParameterTools.nu_sabr(p_nu, delta_time)
    alpha_i = ParameterTools.alpha_atm_sabr(rho_i, nu_i, vol_atm_map[int(dates_list[i])], delta_time)
    for j in range(0, no_z_i_s):
        m_iv[i, j] = SABRTools.sabr_vol_jit(alpha_i, rho_i, nu_i, z_i_s[j], delta_time)
        # m_loc_vol[i, j] = loc_vol.get_vol(