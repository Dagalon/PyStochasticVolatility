from setuptools import setup, find_packages

setup(
    name='PyStochasticVolatility',
    version='1.1',
    # packages=['Tools', 'Solvers', 'Solvers.ODE_Solver', 'Solvers.PDE_Solver', 'Solvers.PDE_Solver.Examples', 'Examples', 'MCPricers', 'MC_Engines', 'MC_Engines.MC_SABR', 'MC_Engines.MC_Heston', 'MC_Engines.GenericSDE', 'MC_Engines.MC_Cheyette', 'MC_Engines.MC_LocalVol', 'MC_Engines.MC_RBergomi', 'MC_Engines.MC_MixedLogNormal', 'MC_Engines.MC_SRoughVolatility', 'Instruments', 'AnalyticEngines', 'AnalyticEngines.BetaZeroSabr', 'AnalyticEngines.FourierMethod', 'AnalyticEngines.FourierMethod.COSMethod', 'AnalyticEngines.FourierMethod.CharesticFunctions', 'AnalyticEngines.LocalVolatility', 'AnalyticEngines.LocalVolatility.Hagan', 'AnalyticEngines.LocalVolatility.Dupire', 'AnalyticEngines.MalliavinMethod', 'AnalyticEngines.VolatilityTools', 'VolatilitySurface', 'VolatilitySurface.Tools', 'FractionalBrownian'],
    packages=find_packages(),
    url='https://github.com/Dagalon/PyStochasticVolatility.git',
    license='http://www.apache.org/licenses/LICENSE-2.0',
    author='David Garcia Lorite',
    author_email='david.garcia.lorite@gmail.com',
    description='Financial library created to give support the book Malliavin Calculus and Stochastic Volatility.'
)
