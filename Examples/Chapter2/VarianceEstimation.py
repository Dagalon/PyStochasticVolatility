import os
import pandas as pd
from pathlib import Path

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
spx_path_info_file = os.path.join(folder_directory, 'MarketData', 'SPX_sample.txt')
vix_path_info_file = os.path.join(folder_directory, 'MarketData', 'VIX_sample.txt')

spx_df = pd.read_csv(spx_path_info_file, header=None, names=["DateTime", "Open", "High", "Low", "Close"])
vix_df = pd.read_csv(vix_path_info_file, header=None, names=["DateTime", "Open", "High", "Low", "Close"])

# Spot variance using Fouier estimators

