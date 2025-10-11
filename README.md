# PyStochasticVolatility

PyStochasticVolatility is a collection of numerical tools that support the book
*Malliavin Calculus and Stochastic Volatility*. The library focuses on
implementing stochastic-volatility (SV) models and pricing routines that are
useful for both research and production prototyping. It bundles analytical
pricers, Monte Carlo engines, utilities for building volatility surfaces, and
worked examples illustrating end-to-end workflows.

## Table of contents
- [Key features](#key-features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Module overview](#module-overview)
- [Development](#development)
- [Support & contributions](#support--contributions)

## Key features
- Fourier-based analytic pricers (COS and Fourier inversion) for models with
  known characteristic functions such as Heston and Merton.
- Monte Carlo pricing engines, variance reduction techniques, and reusable
  instrument definitions that decouple payoff logic from model dynamics.
- Tools for sampling fractional Brownian motion and building volatility
  surfaces with SVI and SABR parameterisations.
- Utilities written with `numba` to accelerate critical numerical kernels.
- Reproducible numerical experiments that accompany each chapter of the book.

## Installation
PyStochasticVolatility targets Python 3.8 and newer. The recommended workflow
uses [uv](https://github.com/astral-sh/uv), but any PEP 517 compliant build tool
is supported.

```bash
# Create and activate a virtual environment (optional but recommended)
uv venv
source .venv/bin/activate

# Install the project in editable mode together with its runtime dependencies
uv pip install -e .
```

If you prefer `pip`, install the package directly from the repository root:

```bash
python -m pip install -e .
```

The main runtime dependencies include:

- numpy
- numba
- scipy
- matplotlib
- pandas
- statsmodels
- quantlib
- tabulate / prettytable
- sympy
- ncephes

Refer to [`pyproject.toml`](pyproject.toml) for the authoritative list of
constraints.

## Quick start
1. Install the project following the instructions above.
2. Explore the notebooks and scripts in [`Examples/`](Examples/) to reproduce the
   numerical experiments from the book.
3. Use the Monte Carlo engines from [`MC_Engines/`](MC_Engines/) together with
   the instruments in [`Instruments/`](Instruments/) to price custom payoffs.
4. Build or calibrate a volatility surface with the tools under
   [`VolatilitySurface/`](VolatilitySurface/).

A minimal script that prices a European option under the Heston model using the
COS method could look like:

```python
from AnalyticEngines.COS import HestonCOSPricer
from Instruments.options import EuropeanCall

instrument = EuropeanCall(strike=100.0, maturity=1.0)
model_params = {
    "kappa": 1.5,
    "theta": 0.04,
    "sigma": 0.3,
    "rho": -0.6,
    "v0": 0.04,
}

price = HestonCOSPricer().price(spot=100.0, rate=0.01, div=0.0,
                               instrument=instrument,
                               params=model_params)
print(f"Option price: {price:.4f}")
```

## Project structure
```
PyStochasticVolatility/
├── AnalyticEngines/     # Fourier-based pricing routines and characteristic functions
├── Examples/            # Reproducible experiments and tutorials
├── FractionalBrownian/  # Fractional Brownian motion sampling utilities
├── Instruments/         # Payoff definitions reusable across models
├── MCPricers/           # Monte Carlo pricing logic decoupled from engines
├── MC_Engines/          # Simulation engines for stochastic processes
├── Solvers/             # PDE solver utilities for local volatility models
├── Tools/               # Shared helpers (random numbers, numba kernels, ...)
└── VolatilitySurface/   # Construction and calibration of volatility surfaces
```

## Module overview
### AnalyticEngines
Fourier-inversion and COS methods that operate on characteristic functions. The
module provides implementations for classical models such as Heston and Merton
alongside Malliavin-based approximations for implied volatility and variance
products.

### Examples
Notebooks and scripts that replicate the experiments presented in each chapter
of the book, serving as practical guides for using the library.

### FractionalBrownian
Sampling algorithms for both full and truncated fractional Brownian motion,
including helper routines for parameter calibration.

### Instruments
A catalogue of model-agnostic instruments (European options, variance swaps,
volatility swaps, etc.) designed to interface seamlessly with either analytic or
Monte Carlo pricing engines.

### MCPricers
Reusable pricing logic for Monte Carlo simulations. These pricers consume an
instrument and a simulated path bundle, separating payoff evaluation from the
choice of engine.

### MC_Engines
Simulation back-ends for a variety of stochastic processes under stochastic
volatility dynamics. Engines can be combined with the pricers above to perform
scenario analysis or pricing.

### Solvers
A robust one-dimensional PDE solver tailored for local volatility models. The
solver supports explicit, implicit, and theta schemes with configurable boundary
conditions.

### Tools
Shared utilities such as random number generators, `numba`-accelerated helper
functions, and general-purpose numerical routines reused throughout the
repository.

### VolatilitySurface
Components for constructing and calibrating volatility term structures. Current
implementations include SVI and SABR parameterisations.

## Development
1. Install development dependencies:
   ```bash
   uv pip install -e .[dev]
   ```
   or use the `requirements.txt` file if you prefer classic `pip` workflows.
2. Run your preferred test or lint suite. (A comprehensive test harness is not
   yet bundled with the project.)
3. Format contributions using the conventions already present in the codebase.

## Support & contributions
Issues and pull requests are welcome on the
[GitHub repository](https://github.com/Dagalon/PyStochasticVolatility). If you
use the library in academic work, please consider citing the book *Malliavin
Calculus and Stochastic Volatility* and referencing this repository.

The project is distributed under the Apache 2.0 License. Individual source
files contain the relevant license headers.
