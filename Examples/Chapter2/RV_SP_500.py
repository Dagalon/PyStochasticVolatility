import os
import pandas as pd

from pathlib import Path

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
historical_data_path = os.path.join(folder_directory, 'Data', 'SPX_sample.txt')

historical_data = pd.read_csv(historical_data_path, header=None, names=["date", "price"], sep=";")