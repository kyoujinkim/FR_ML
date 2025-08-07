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

class LongTermLearner():
    def __init__(self, config, Model, trn_dl, val_dl, tst_dl, opt, loss_fn, device='cpu'):
        self.config = config
        self.model = Model
        self.trn_dl = trn_dl
        self.val_dl = val_dl
        self.tst_dl = tst_dl
        self.device = device
        self.opt = opt
        self.loss_fn = loss_fn

    def train(self):
        self.model.train()
        total_loss = 0
        losses = []

        print(f"Training...{datetime.datetime.now()}")
        for i, (x, y, x_mark, y_mark) in enumerate(self.trn_dl):
            x = x.float().to(self.device)
            y = y.float().to(self.device)
            x_mark = x_mark.float().to(self.device)
            y_mark = y_mark.float().to(self.device)

            f_dim = -1 if self.config.features == 'MS' else 0
            pred = self.model(x, x_mark, y, y_mark)
            pred = pred[:, -self.config.pred_len:, f_dim:]
            y = y[:, -self.config.pred_len:, f_dim:].to(self.device)

            loss = self.loss_fn(pred, y) / 100
            loss.backward()

            if (i+1) % 100 == 0:
                print(f"Batch {i}, Loss: {loss.item():.4f}")
                self.opt.step()
                self.opt.zero_grad()

            losses.append(loss.detach().item())
            total_loss += loss.detach().item()

        return total_loss / len(self.trn_dl)

    def validation(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for i, (x, y, x_mark, y_mark) in enumerate(self.val_dl):
                x = x.to(device)
                y = y.to(device)
                x_mark = x_mark.to(device)
                y_mark = y_mark.to(device)

                f_dim = -1 if self.config.features == 'MS' else 0
                pred = self.model(x, x_mark, y, y_mark)
                pred = pred[:, -self.config.pred_len:, f_dim:]
                y = y[:, -self.config.pred_len:, f_dim:].to(self.device)

                loss = self.loss_fn(pred, y)
                total_loss += loss.detach().item()

        return total_loss / len(self.val_dl)

    def fit(self, epochs=32):
        for epoch in range(epochs):
            loss = self.train()
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
            validation_loss = self.validation()
            print(f"Epoch {epoch+1}, Validation Loss: {validation_loss:.4f}")

        return self.model

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

    model = Model(config).to(device).float()
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.L1Loss()

    '''if config.checkpoints is not None:
        if os.path.exists(config.checkpoints):
            model.load_state_dict(torch.load(config.checkpoints))
            print(f"Model loaded from {config.checkpoints}")
        else:
            print(f"Checkpoints file {config.checkpoints} does not exist, starting training from scratch.")
            os.makedirs(os.path.dirname(config.checkpoints), exist_ok=True)'''

    ltl = LongTermLearner(config, model, dl, dl_val, dl_test, opt, loss_fn, device)

    ltl.fit(epochs=config.train_epochs)
    # save model
    torch.save(model.state_dict(), config.checkpoints)
    # load model
    #model.load_state_dict(torch.load('./src/cache/us/model.pth'))