import matplotlib.pylab as plt
import os
from pathlib import Path
import operator
import numpy as np

import pandas as pd

currentDirectory = os.path.dirname(os.path.realpath(__file__))
folderDirectory = Path(currentDirectory)
smileDataPath = os.path.join(folderDirectory, 'MarketData', 'SwaptionTenor5Y.csv')

smileDf = pd.read_csv(smileDataPath, header=None,
                      names=["FORWARD", "SPREAD", "MATURITY", "MARKET_VOLATILITY", "SABR_VOLATILITY"], sep=";")
maturities = smileDf.MATURITY.unique()[1:]

strikes = {}
ivs = {}
ivsMarket = {}
spreads = {}
forwards = {}

for ti in maturities:
    subDf = smileDf.query(f"MATURITY == '{ti}'")
    spreads[ti] = list(subDf['SPREAD'].map(lambda x: float(x)))
    forwards[ti] = list(subDf['FORWARD'].map(lambda x: float(x)))
    strikes[ti] = list(map(operator.add, spreads[ti], forwards[ti]))
    ivs[ti] = list(subDf['SABR_VOLATILITY'].map(lambda x: float(x)))
    ivsMarket[ti] = list(subDf['MARKET_VOLATILITY'].map(lambda x: float(x)))

plot_to_print = 3

plt.plot(strikes[maturities[plot_to_print]], ivs[maturities[plot_to_print]], linestyle='dashed', color="black",
         label="SABR")
plt.scatter(strikes[maturities[plot_to_print]], ivsMarket[maturities[plot_to_print]], color="black", label="Market",
            s=15, marker="x")

# plt.grid(linestyle='dashed')

plt.title("T=" + maturities[plot_to_print])
plt.xlabel("K")
plt.ylabel("IV")
plt.legend()

plt.ylim((0.0, np.max(ivs[maturities[plot_to_print]])+0.01))


pathToSave = "C:/Users/Pc/Desktop/Book_Figures/swaption_tenor_5Y_" + maturities[plot_to_print] + ".png"
plt.savefig(pathToSave, dpi=300, bbox_inches='tight')


plt.show()