import numpy as np
import matplotlib.pylab as plt
import os
from pathlib import Path

import pandas as pd

currentDirectory = os.path.dirname(os.path.realpath(__file__))
folderDirectory = Path(currentDirectory)
smileDataPath = os.path.join(folderDirectory, 'MarketData', 'swaption5Yx5Y.csv')

smileDf = pd.read_csv(smileDataPath, header=None, names=["Maturity", "Strike", "Vol"], sep=";")
maturities = smileDf.Maturity.unique()[1:]

strikes = {}
ivs = {}

for ti in maturities:
    subDf = smileDf.query(f"Maturity == '{ti}'")
    strikes[ti] = list(subDf['Strike'].map(lambda x: float(x)))
    ivs[ti] = list(subDf['Vol'].map(lambda x: float(x)))


fig, axs = plt.subplots(1, 3)
fig.suptitle('Swaption 5Yx5Y Smile')

# plot 1
axs[0].set(xlabel='K', ylabel='IV')
axs[0].plot(strikes[maturities[0]], ivs[maturities[0]], linestyle='dashed', color='blue')

# plot 2
axs[1].set(xlabel='K', ylabel='IV')
axs[1].plot(strikes[maturities[1]], ivs[maturities[1]], linestyle='dashed', color='blue')

# plot 3
axs[2].set(xlabel='K', ylabel='IV')
axs[2].plot(strikes[maturities[2]], ivs[maturities[2]], linestyle='dashed', color='blue')

plt.show()
