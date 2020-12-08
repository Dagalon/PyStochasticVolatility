import os
import pandas as pd
import matplotlib.pylab as plt
from pathlib import Path

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
historical_data_path = os.path.join(folder_directory, 'Data', 'VIX_Info.txt')

vix_data = pd.read_csv(historical_data_path, header=None, names=["Maturity", "Strike", "IV"], sep=";")

strikes = list(map(lambda x: float(x), list(vix_data[vix_data["Maturity"] == "43572"]["Strike"])))
iv = list(map(lambda x: float(x), list(vix_data[vix_data["Maturity"] == "43572"]["IV"])))

plt.plot(strikes, iv, label='T = %s' % '17/04/2019', color='black', linestyle='--', marker='.')

plt.xlabel('K')
plt.legend()
plt.show()

