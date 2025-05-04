'''
Regime prediction with factor data
classify the regime, and train model with each regime's data
it would be beneficial to merge those process into one model, not in seperate model.
'''

import pandas as pd

# build data
r = pd.read_parquet('./src/cache/us/return.parquet')
rft = pd.read_csv('./src/cache/us/bt_plot.csv', index_col=0, parse_dates=True)
rft.columns = ['inv','mom','prf','smb','hml']

# concat rft with each column of r
# each data shape is (96, 6)
dataset = []
for col in r.columns:
    dataset.append(pd.concat([rft, r[col]], axis=1).dropna().values)
