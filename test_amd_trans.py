"""
Test script for Global-Aware AMD-Trans model.

Trains AMD-Trans on financial factor data and evaluates against
PatchTST and iTransformer baselines.  Results are written to logs/.

Usage:
    python test_amd_trans.py                  # AMD-Trans only
    python test_amd_trans.py --compare        # + PatchTST & iTransformer baselines
    python test_amd_trans.py --test-only      # skip training, load saved checkpoint
"""
import argparse
import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import TS_dataset
from src.models.amd_trans import Model as AMDTransModel, HybridLoss
from src.models.patchTST import Model as PatchTSTModel
from src.models.iTransformer import Model as iTransformerModel
from src.utils.metrics import metric
from train import LongTermLearner, read_config, load_factors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_dataloaders(config, country, batch_size, data_apath='data', skip_col=None):
    r = pd.read_parquet(f'{data_apath}/{country}/returns.parquet')
    p = (r + 1).cumprod()
    fct = load_factors(
        f'{data_apath}/{country}',
        ['value', 'size', 'momentum', 'investment', 'profitability'],
        'parquet'
    )
    size = [config.seq_len, config.label_len, config.pred_len]
    skip_col = skip_col  # factor columns — skip std-scale normalisation
    train_pct = [0.6, 0.2, 0.2]

    ds_trn = TS_dataset(p, fct=fct, size=size, train_pct=train_pct, std_scale=False, flag='train', skip_col=skip_col)
    ds_val = TS_dataset(p, fct=fct, size=size, train_pct=train_pct, std_scale=False, flag='valid', skip_col=skip_col)
    ds_tst = TS_dataset(p, fct=fct, size=size, train_pct=train_pct, std_scale=False, flag='test',  skip_col=skip_col)

    dl_trn = DataLoader(ds_trn, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_tst = DataLoader(ds_tst, batch_size=batch_size, shuffle=False)

    print(f"Data  —  train: {len(ds_trn):,}  val: {len(ds_val):,}  test: {len(ds_tst):,}")
    return dl_trn, dl_val, dl_tst


def run_model(config, model, model_name, country, dl_trn, dl_val, dl_tst,
              device, loss_fn, epochs, test_only=False, cpath='checkpoints', spath='logs'):
    checkpath = f'{cpath}/{model_name}_{country}'
    save_path = spath
    os.makedirs(save_path, exist_ok=True)

    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    learner = LongTermLearner(config, model, dl_trn, dl_val, dl_tst, opt, loss_fn, device)

    if not test_only:
        print(f"\n{'='*60}")
        print(f"  Training  {model_name}  [{country}]")
        print(f"{'='*60}")
        learner.fit(
            model_name=model_name,
            country=country,
            epochs=epochs,
            checkpath=checkpath,
            save_path=save_path,
        )

    print(f"\n  Evaluating  {model_name}  ...")
    learner.test(
        model_name=model_name,
        country=country,
        checkpath=checkpath,
        save_path=save_path,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--country',    default='korea', help='data sub-folder name')
    p.add_argument('--config',     default='./config.ini')
    p.add_argument('--compare',    action='store_true',
                   help='also train/test PatchTST and iTransformer baselines')
    p.add_argument('--test-only',  action='store_true',
                   help='skip training; load existing checkpoints')
    p.add_argument('--loss',       default='hybrid', choices=['hybrid', 'l1', 'mse'],
                   help='loss function for AMD-Trans')
    p.add_argument('--data_apath',   default=None, help='absolute path to config file (overrides --config)')
    p.add_argument('--check_apath',  default=None, help='absolute path to checkpoints (overrides default)')
    p.add_argument('--save_apath',   default=None, help='absolute path to logs (overrides default)')
    p.add_argument('--skip_col',  nargs='*', type=int, default=[0, 2, 3, 4],)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args() # tmp
    config = read_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    torch.manual_seed(config.random_seed)

    dl_trn, dl_val, dl_tst = build_dataloaders(config, args.country, config.batch_size, args.data_apath, args.skip_col)

    # ------------------------------------------------------------------
    # Loss function for AMD-Trans
    # ------------------------------------------------------------------
    if args.loss == 'hybrid':
            amd_loss = HybridLoss(alpha=config.hybrid_alpha, beta=config.hybrid_beta)
    elif args.loss == 'l1':
        amd_loss = nn.L1Loss()
    else:
        amd_loss = nn.MSELoss()

    # ------------------------------------------------------------------
    # AMD-Trans
    # ------------------------------------------------------------------
    amd_model = AMDTransModel(config).to(device).float()
    param_count = sum(p.numel() for p in amd_model.parameters() if p.requires_grad)
    print(f"\nAMD-Trans parameters: {param_count:,}")

    run_model(
        config=config,
        model=amd_model,
        model_name='amd_trans',
        country=args.country,
        dl_trn=dl_trn, dl_val=dl_val, dl_tst=dl_tst,
        device=device,
        loss_fn=amd_loss,
        epochs=config.train_epochs,
        test_only=args.test_only,
        cpath=args.check_apath,
        spath=args.save_apath,
    )

    # ------------------------------------------------------------------
    # Baselines (optional)
    # ------------------------------------------------------------------
    if args.compare:
        baselines = {
            'patchTST':     PatchTSTModel(config),
            'iTransformer': iTransformerModel(config),
        }
        baseline_loss = nn.L1Loss()

        for model_name, model in baselines.items():
            model = model.to(device).float()
            run_model(
                config=config,
                model=model,
                model_name=model_name,
                country=args.country,
                dl_trn=dl_trn, dl_val=dl_val, dl_tst=dl_tst,
                device=device,
                loss_fn=baseline_loss,
                epochs=config.train_epochs,
                test_only=args.test_only,
                cpath=args.check_apath,
                spath=args.save_apath,
            )

    print('\nDone. Results saved to logs/result_long_term_forecast.txt')