'''
Regime prediction with factor data
classify the regime, and train model with each regime's data
it would be beneficial to merge those process into one model, not in seperate model.
'''
import argparse
import datetime

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
    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        seq_len = y.size(1)
        tgt_mask = model.generate_square_subsequent_mask(seq_len).to(device)

        pred = model(x, y, tgt_mask)
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

            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)

def fit(dl, dl_val, epochs=32):
    for epoch in range(epochs):
        loss = train(model, opt, loss_fn, dl)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        validation_loss = validation(model, loss_fn, dl_val)
        print(f"Epoch {epoch+1}, Validation Loss: {validation_loss:.4f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Transformer Model for Time Series')
    parser.add_argument('--rpath', type=str, default='./src/cache/us/return.parquet', help="return file path")
    parser.add_argument('--fpath', type=str, default='./src/cache/us/return_factor.parquet', help="factor file path")
    parser.add_argument('--flag', type=str, default="train", help="train or valid or test")
    parser.add_argument('--d_model', type=int, default=64, help="dimension of model")
    parser.add_argument('--n_heads', type=int, default=4, help="head of MHA")
    parser.add_argument('--n_layers', type=int, default=2, help="Layer of Transformer")
    parser.add_argument('--use_factor', type=bool, default=False, help="use or not use factor data")
    parser.add_argument('--c_out', type=int, default=1, help="output channel")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--epochs', type=int, default=32, help="batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--savepath', type=str, default='./src/cache/us/model.pth', help="model save path")
    parser.add_argument('--loadpath', type=str, default=None, help="model load path")
    args = parser.parse_args()

    # build data
    r = pd.read_parquet(args.rpath)
    p = (r + 1).cumprod()

    if args.use_factor:
        rft = pd.read_parquet(args.fpath)
        rft.columns = ['inv', 'mom', 'prf', 'smb', 'hml']
        c_in = 1 + 5
    else:
        rft = None
        c_in = 1

    ds = TS_dataset(p, fct=rft, flag=args.flag)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    ds_val = TS_dataset(p, fct=rft, flag='valid')
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    ds_test = TS_dataset(p, fct=rft, flag='test')
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, c_in=c_in, c_out=args.c_out).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    if args.loadpath is not None:
        model.load_state_dict(torch.load(args.loadpath))
        print(f"Model loaded from {args.loadpath}")

    fit(dl, epochs=args.epochs, dl_val=dl_val)
    # save model
    torch.save(model.state_dict(), args.savepath)
    # load model
    #model.load_state_dict(torch.load('./src/cache/us/model.pth'))