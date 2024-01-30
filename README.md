# PyStochasticVolatility
This repository contains different tools to simulate underlyings under SV dynamics. As well, we have implemented several tools for computing option price under SV. 

## Packages you need to include
The next packages are using of a intesive way:
* python ~= 3.9
* numpy~=1.21.5
* numba~=0.55.1
* ncephes~=1.1.0
* matplotlib~=3.5.1
* scipy~=1.7.3
* pandas~=1.4.2
* statsmodels~=0.13.2
* tabulate~=0.8.9
* prettytable~=2.4.0
* setuptools~=61.2.0
* py_vol ~= Last version
* sobol_seq ~= Last version

## Module description
We will a brief description of each module of the library.

### AnalyticEngines
In this module, the reader can find two numerical methods Fourier inversion or COS method to price options under dynamics with known characteristic function. Currently, we have implemented the characteristic function for the Heston and Merton model. In addition, we have included the different approximations for implied volatility, vol swap, variance swap $\cdots$ that we have obtained using Malliavin tools.

### Examples
 Here the reader can find the numerical experiments performed for each chapter of the book. 
 
### FractionalBrownian
Tools for sampling the fractional brownian motion and the truncated fractional brownian.
 
### Instruments
In this module, the reader can find the different instruments that we have used in the book. These instruments are independent of the model, for this reason, we cause the same instrument for different MC engines.
 
### MCPricers
This module contains the different MC pricers that the user can use in any MC Engine.
 
### Solvers
We have included a one dimensional PDE solver. Under the local volatility model, this numerical is so stable and fast. The user can use generic boundary conditions and solver the PDE using explicit, implicit, or theta scheme.
  
### Tools
This module contains generic functionalities (random number generator, numba functions, ...) that are used in several parts of the library.
  
### VolatilitySurface
The term struct for building volatility surface. We have implemented SVI and SABR.


