import matplotlib.pylab as plt
import os
from pathlib import Path

import numpy as np
import pandas as pd

currentDirectory = os.path.dirname(os.path.realpath(__file__))
folderDirectory = Path(currentDirectory)
smileDataPath = os.path.join(folderDirectory, 'MarketData', 'swaption5Yx5Y.csv')

smileDf = pd.read_csv(smileDataPath, header=None, names=["Maturity", "Forward", "Spread", "Vol", "Vol Market"], sep=";")
maturities = smileDf.Maturity.unique()[1:]

strikes = {}
ivs = {}
ivsMarket = {}
spreads = {}
forwards = {}

for ti in maturities:
    subDf = smileDf.query(f"Maturity == '{ti}'")
    spreads = list(subDf['Spread'].map(lambda x: float(x)))
    forwards = list(subDf['Forward'].map(lambda x: float(x)))
    strikes = list(np.sum(spreads, forwards))
    ivs[ti] = list(subDf['Vol'].map(lambda x: float(x)))
    ivsMarket[ti] = list(subDf['VolMarket'].map(lambda x: float(x)))

plot_to_print = 2

plt.plot(strikes[maturities[plot_to_print]], ivs[maturities[plot_to_print]], linestyle='dashed', color="blue",
         label="SABR")

plt.scatter(strikes[maturities[plot_to_print]], ivsMarket[maturities[plot_to_print]], color="black", label="Market")
plt.title("T=" + maturities[plot_to_print])
plt.xlabel("K")
plt.ylabel("IV")
plt.legend()
plt.show()
