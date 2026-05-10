"""
Test script for Global-Aware AMD-Trans model.

Trains AMD-Trans on financial factor data and evaluates against
PatchTST and iTransformer baselines.  Results are written to logs/.

Usage:
    python test_profit.py                  # AMD-Trans only
    python test_profit.py --compare        # + PatchTST & iTransformer baselines
    python test_profit.py --test-only      # skip training, load saved checkpoint
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import TS_dataset
from src.models.profit import Model as ProfitModel
from src.models.patchTST import Model as PatchTSTModel
from src.models.iTransformer import Model as iTransformerModel
from src.utils.metrics import metric
from train import LongTermLearner, read_config, load_factors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_dataloaders(config, country, batch_size, data_apath='data', skip_col=None, year:int=-1, sector:int=-1):
    r = pd.read_parquet(f'{data_apath}/{country}/returns.parquet')
    p = (r + 1).cumprod()
    fct = load_factors(
        f'{data_apath}/{country}',
        ['value', 'size', 'momentum', 'investment', 'profitability'],
        'parquet'
    )
    size = [config.seq_len, config.label_len, config.pred_len]
    skip_col = skip_col  # factor columns — skip std-scale normalisation
    train_pct = [1.0, .0, .0]#[0.6, 0.2, 0.2]

    if year!=-1:
        start_date = f'{year-1}-01-01'
        end_date = f'{year}-12-31'
        p = p.loc[start_date:end_date]

    if sector!= -1:
        sector_csv = pd.read_csv(f'{data_apath}/sector.csv', index_col=0, dtype=str)
        sector_csv['sector'] = sector_csv['sector'].str[:2]
        sector_matched = sector_csv[sector_csv['sector'] == sector]
        intercross_matched = p.columns.intersection(sector_matched.index)
        if len(intercross_matched) == 0:
            print(f"No matching stocks found for sector {sector} in country {country}.")
            return None, None, None
        p = p.loc[:, intercross_matched]

    ds = TS_dataset(p, fct=fct, size=size, train_pct=train_pct, std_scale=False, flag='train', skip_col=skip_col)
    #ds_val = TS_dataset(p, fct=fct, size=size, train_pct=train_pct, std_scale=False, flag='valid', skip_col=skip_col)
    #ds_tst = TS_dataset(p, fct=fct, size=size, train_pct=train_pct, std_scale=False, flag='test',  skip_col=skip_col)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    #dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    #dl_tst = DataLoader(ds_tst, batch_size=batch_size, shuffle=False)

    print(f"Data  — test: {len(ds):,}")
    return dl, dl, dl


def run_model(config, model, model_name, country, dl_trn, dl_val, dl_tst,
              device, loss_fn, cpath='checkpoints', spath='logs', year=-1):
    checkpath = f'{cpath}/{model_name}_{country}'
    save_path = spath
    os.makedirs(save_path, exist_ok=True)

    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    learner = LongTermLearner(config, model, dl_trn, dl_val, dl_tst, opt, loss_fn, device)

    print(f"\n  Evaluating  {model_name}  ...")
    result = learner.test(
        model_name=model_name,
        country=country,
        checkpath=checkpath,
        save_path=save_path,
        year=year,
    )

    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--country',    default='korea', help='data sub-folder name')
    p.add_argument('--config',     default='./config.ini')
    p.add_argument('--data_apath',   default='./data', help='absolute path to config file (overrides --config)')
    p.add_argument('--check_apath',  default='./checkpoints', help='absolute path to checkpoints (overrides default)')
    p.add_argument('--save_apath',   default='./logs', help='absolute path to logs (overrides default)')
    p.add_argument('--skip_col',  nargs='*', type=int, default=[0, 1, 2, 3, 4],)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args() # tmp
    config = read_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    torch.manual_seed(config.random_seed)

    for country in ['korea', 'us', 'japan', 'europe']:
        total_result = []

        for year in [-1]:
            for sector in ['10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60']:
                print(f"\n{'='*60}")
                print(f"  Year: {year}  ")
                print(f"{'='*60}")

                dl_trn, dl_val, dl_tst = build_dataloaders(config, country, config.batch_size, args.data_apath, args.skip_col, year, sector)

                amd_loss = nn.L1Loss()

                amd_model = ProfitModel(config).to(device).float()
                param_count = sum(p.numel() for p in amd_model.parameters() if p.requires_grad)
                print(f"\nPROFIT parameters: {param_count:,}")

                result = run_model(
                    config=config,
                    model=amd_model,
                    model_name='profit_rev-1-2-3',
                    country=country,
                    dl_trn=dl_trn, dl_val=dl_val, dl_tst=dl_tst,
                    device=device,
                    loss_fn=amd_loss,
                    cpath=args.check_apath,
                    spath=args.save_apath,
                    year=year,
                )
                result['sector'] = sector
                total_result.append(result)
        pd.concat(total_result).to_csv(f'{args.save_apath}/profit_{country}_yearly_sector.csv', index=False)