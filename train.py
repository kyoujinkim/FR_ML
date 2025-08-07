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
from src.utils.tools import EarlyStopping, adjust_learning_rate, visual
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

    def train(self, dl):
        self.model.train()
        total_loss = 0
        losses = []

        print(f"Training...{datetime.datetime.now()}")
        for i, (x, y, x_mark, y_mark) in enumerate(dl):
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

            if (i+1) % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item():.4f}")
                self.opt.step()
                self.opt.zero_grad()

            losses.append(loss.detach().item())
            total_loss += loss.detach().item()

        return total_loss / len(dl)

    def validation(self, dl):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for i, (x, y, x_mark, y_mark) in enumerate(dl):
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

        return total_loss / len(dl)

    def fit(self, epochs=32):
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        for epoch in range(epochs):
            loss = self.train(self.trn_dl)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
            validation_loss = self.validation(self.val_dl)
            print(f"Epoch {epoch+1}, Validation Loss: {validation_loss:.4f}")
            test_loss = self.validation(self.tst_dl)
            print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}")
            early_stopping(validation_loss, self.model, self.config.checkpoints)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.opt, epoch+1, self.config)

        best_model_path = self.config.checkpoints + '/checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

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

    country = 'us'
    flag = 'train'
    batch_size = config.batch_size
    # build data
    r = pd.read_parquet(f'./data/{country}/returns.parquet')
    p = (r + 1).cumprod()

    fct = load_factors('./data/us', ['value', 'size', 'momentum', 'investment', 'profitability'], 'parquet')

    size = [config.seq_len, config.label_len, config.pred_len]
    skip_col = [0, 2, 3, 4]  # columns to skip normalization
    ds = TS_dataset(p, fct=fct, size=size, flag='train', skip_col=skip_col)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    ds_val = TS_dataset(p, fct=fct, size=size, flag='valid', skip_col=skip_col)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    ds_test = TS_dataset(p, fct=fct, size=size, flag='test', skip_col=skip_col)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    print(f"Data loaded: {len(ds)} train, {len(ds_val)} val, {len(ds_test)} test")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(config).to(device).float()
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.L1Loss()

    ltl = LongTermLearner(config, model, dl, dl_val, dl_test, opt, loss_fn, device)

    model = ltl.fit(epochs=config.train_epochs)