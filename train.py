'''
Regime prediction with factor data
classify the regime, and train model with each regime's data
it would be beneficial to merge those process into one model, not in seperate model.
'''
import argparse
import configparser
import datetime
import os.path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.models.transformers import Model

from src.dataset import TS_dataset

def train(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0
    losses = []
    print(f"Training...{datetime.datetime.now()}")
    for i, (x, y, x_mark, y_mark) in enumerate(dataloader):
        x = x.float().to(device)
        y = y.float().to(device)

        seq_len = y.size(1)
        tgt_mask = model.generate_square_subsequent_mask(seq_len).to(device)

        pred = model(x, y, tgt_mask)
        if pred.shape != y.shape:
            y = y[:,:,0].reshape(list(y.shape[:2]) + [1])
        loss = loss_fn(pred, y)
        loss.backward()

        if i % 100 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")
            opt.step()
            opt.zero_grad()

        losses.append(loss.detach().item())
        total_loss += loss.detach().item()

    return total_loss / len(dataloader)

def validation(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            y_input = y[:,:-1]
            y_expected = y[:,1:]

            seq_len = y_input.size(1)
            tgt_mask = model.generate_square_subsequent_mask(seq_len).to(device)

            pred = model(x, y_input, tgt_mask)

            if pred.shape != y_expected.shape:
                y_expected = y_expected[:, :, 0].reshape(list(y_expected.shape[:2]) + [1])
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)

def fit(dl, dl_val, epochs=32):
    for epoch in range(epochs):
        loss = train(model, opt, loss_fn, dl)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        validation_loss = validation(model, loss_fn, dl_val)
        print(f"Epoch {epoch+1}, Validation Loss: {validation_loss:.4f}")

def load_factors(path, factors, format='csv'):
    fct = []
    for f in factors:
        if format=='csv':
            fct.append(pd.read_csv(f'./{path}/{f}.csv', index_col=0, parse_dates=True))
        elif format=='parquet':
            fct.append(pd.read_parquet(f'{path}/{f}.parquet'))
        else:
            raise ValueError(f"Unsupported file format: {f}")
    return fct

class read_config():
    def __init__(self, path):
        config = configparser.ConfigParser()
        config.read(path)
        items = config.items('VARS')
        for item in items:
            name = item[0]
            var = item[1]
            # convert to appropriate type
            if var.isdigit():
                var = int(var)
            elif var.replace('.', '', 1).isdigit():
                var = float(var)
            elif var.lower() in ['true', 'false']:
                var = var.lower() == 'true'
            else:
                var = var
            setattr(self, name, var)

if __name__ == "__main__":

    config = read_config('./config.ini')

    country = 'europe'
    flag = 'train'
    batch_size = config.batch_size
    # build data
    r = pd.read_parquet(f'./data/{country}/returns.parquet')
    p = (r + 1).cumprod()

    fct = load_factors('./data/europe', ['value', 'size', 'momentum', 'investment', 'profitability'], 'parquet')

    skip_col = [0, 2, 3, 4]  # columns to skip normalization
    ds = TS_dataset(p, fct=fct, size=[config.seq_len, config.label_len, config.pred_len], flag='train', skip_col=skip_col)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    ds_val = TS_dataset(p, fct=fct, flag='valid', skip_col=skip_col)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    ds_test = TS_dataset(p, fct=fct, flag='test', skip_col=skip_col)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(config).float().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    '''if config.checkpoints is not None:
        if os.path.exists(config.checkpoints):
            model.load_state_dict(torch.load(config.checkpoints))
            print(f"Model loaded from {config.checkpoints}")
        else:
            print(f"Checkpoints file {config.checkpoints} does not exist, starting training from scratch.")
            os.makedirs(os.path.dirname(config.checkpoints), exist_ok=True)'''

    fit(dl, epochs=config.train_epochs, dl_val=dl_val)
    # save model
    torch.save(model.state_dict(), config.checkpoints)
    # load model
    #model.load_state_dict(torch.load('./src/cache/us/model.pth'))