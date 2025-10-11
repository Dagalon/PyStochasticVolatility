import matplotlib.pylab as plt
import os
from pathlib import Path
import operator
import numpy as np

import pandas as pd

currentDirectory = os.path.dirname(os.path.realpath(__file__))
folderDirectory = Path(currentDirectory)
smileDataPath = os.path.join(folderDirectory, 'MarketData', 'volatility_market_swaption.csv')

smileDf = pd.read_csv(smileDataPath, header=None,
                      names=["TENOR", "SPREAD", "VOL_MARKET"], sep=";")
tenors = smileDf.TENOR.unique()[1:]

strikes = {}
ivs = {}
ivsMarket = {}
spreads = {}
forwards = {}

for tn in tenors:
    subDf = smileDf.query(f"TENOR == '{tn}'")
    spreads[tn] = list(subDf['SPREAD'].map(lambda x: float(x)))
    strikes[tn] = list(spreads[tn])
    ivs[tn] = list(subDf['VOL_MARKET'].map(lambda x: float(x)))

plot_to_print = 1

plt.plot(strikes[tenors[plot_to_print]], ivs[tenors[plot_to_print]], linestyle='dashed', color="black",
         label="IV Market", marker="x")

# plt.grid(linestyle='dashed')

plt.title("Tenors=" + tenors[plot_to_print])
plt.xlabel("K")
plt.ylabel("IV")
plt.legend()

plt.ylim((np.max(ivs[tenors[plot_to_print]]) - 0.01, np.max(ivs[tenors[plot_to_print]])+0.01))


pathToSave = "C:/Users/Pc/Desktop/Book_Figures/volatility_smile_" + tenors[plot_to_print] + ".png"
plt.savefig(pathToSave, dpi=300, bbox_inches='tight')


plt.show()