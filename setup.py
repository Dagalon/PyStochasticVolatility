from setuptools import setup

setup(
    name='PyStochasticVolatility',
    version='1.0',
    packages=['Tools', 'Tools.VolatilityTools', 'Solvers', 'Solvers.ODE_Solver', 'Solvers.PDE_Solver',
              'Solvers.PDE_Solver.Examples', 'MCPricers', 'MC_Engines', 'MC_Engines.MC_SABR', 'MC_Engines.MC_Heston',
              'MC_Engines.GenericSDE', 'MC_Engines.MC_RBergomi', 'MC_Engines.MC_Bergomi2F',
              'MC_Engines.MC_LocalVol', 'Instruments', 'AnalyticEngines', 'AnalyticEngines.Expansions',
              'AnalyticEngines.FourierMethod', 'AnalyticEngines.FourierMethod.COSMethod',
              'AnalyticEngines.FourierMethod.CharesticFunctions', 'AnalyticEngines.LocalVolatility',
              'AnalyticEngines.LocalVolatility.Hagan', 'AnalyticEngines.LocalVolatility.Dupire',
              'AnalyticEngines.MalliavinMethod', 'VolatilitySurface', 'VolatilitySurface.Tools', 'FractionalBrownian'],
    url='',
    license='http://www.apache.org/licenses/LICENSE-2.0',
    author='David Garcia Lorite',
    author_email='david.garcia.lorite@outlook.com',
    description='Financial library creted to give support the book Malliavin Calculus and Stochastic Volatility.'
)
