'''
Regime prediction with factor data
classify the regime, and train model with each regime's data
it would be beneficial to merge those process into one model, not in seperate model.
'''

import pandas as pd
from src.dataset import DataLoader

# build data
r = pd.read_parquet('./src/cache/us/return.parquet')
p = (r+1).cumprod()

rft = pd.read_csv('./src/cache/us/bt_plot.csv', index_col=0, parse_dates=True)
rft.columns = ['inv','mom','prf','smb','hml']

dl = DataLoader(p, flag='train')

c = 1