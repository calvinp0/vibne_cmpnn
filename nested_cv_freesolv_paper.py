#!/usr/bin/env python
"""
Nested scaffold CV + 20 % hold-out evaluation for FreeSolv.

Outputs:
    logs/outer_test_predictions.csv   (smiles,y_true,y_pred)
    logs/outer_test_metrics.json      (rmse,mae,r2,ci_low,ci_hi)
"""
from __future__ import annotations
import os, json, random, argparse, math, hashlib
from collections import defaultdict
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import optuna
from sklearn.utils import resample
from tqdm import tqdm

from dataset import CMPNNDataset, generate_scaffold
from model   import CMPNNEncoder, FFNHead, initialize_xavier
from utils   import rmse_torch, mae_torch, r2_score_torch   # <-- your helpers
from preprocessing import scaffold_split_indices_repeated
from torch.utils.data import Subset
# --------------------------------------------------------------------------------------

# ------------------------------------------------- splitting helpers
def outer_scaffold_split(df: pd.DataFrame, test_ratio: float = .20,
                         seed: int = 42) -> tuple[list[int], list[int]]:
    """Return (train_idx, test_idx) with ≈test_ratio molecules in test."""
    rng = np.random.default_rng(seed)
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(df.smiles):
        scaffolds[generate_scaffold(smi)].append(i)

    groups = list(scaffolds.values())
    rng.shuffle(groups)                              # random order but reproducible

    test, train = [], []
    n_test = int(len(df) * test_ratio + .5)
    for g in groups:
        (test if len(test) < n_test else train).extend(g)
    return train, test


def inner_5fold_indices(train_idx: list[int],
                        df: pd.DataFrame,
                        seed: int = 0
                        ) -> list[tuple[list[int], list[int]]]:
    """
    5-fold scaffold split that returns **global row numbers**
    (i.e. indices valid for the original DataFrame / Dataset).
    """
    rng = np.random.default_rng(seed)

    # 1) bucket global ids by Bemis-Murcko scaffold
    from collections import defaultdict
    buckets: defaultdict[str, list[int]] = defaultdict(list)
    for gid in train_idx:                                 # ← global id
        smi = df.smiles.iat[gid]
        buckets[generate_scaffold(smi)].append(gid)

    # 2) largest buckets first, then round-robin into 5 folds
    groups = sorted(buckets.values(), key=len, reverse=True)
    rng.shuffle(groups)

    folds = [[] for _ in range(5)]
    for i, g in enumerate(groups):
        folds[i % 5].extend(g)

    # 3) build (train,val) tuples – still global numbers
    splits = []
    for k in range(5):
        val   = folds[k]
        train = [g for j in range(5) if j != k for g in folds[j]]
        splits.append((train, val))
    return splits

# ------------------------------------------------- tiny training loop
def build_model(in_node, in_edge, hidden, steps, dropout_mp, dropout_head,
                device):
    enc = CMPNNEncoder(in_node, in_edge, hidden_dim=hidden,
                       num_steps=steps, dropout=dropout_mp).to(device)
    head = FFNHead(in_dim=hidden*2, hidden_dim=hidden//2, out_dim=1,
                   dropout=dropout_head).to(device)
    enc.apply(initialize_xavier)
    head.apply(initialize_xavier)
    return enc, head


def fit_one_epoch(loader, enc, head, opt, device):
    criterion = torch.nn.MSELoss()
    enc.train(); head.train()
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        z = enc.embed(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(head(z).view(-1), batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(head.parameters()), 5.)
        opt.step()


@torch.no_grad()
def evaluate(loader, enc, head, device):
    enc.eval(); head.eval()
    outs, tgts = [], []
    for b in loader:
        b = b.to(device)
        z = enc.embed(b.x, b.edge_index, b.edge_attr, b.batch)
        outs.append(head(z).view(-1).cpu().numpy())
        tgts.append(b.y.cpu().numpy())
    y_pred = np.concatenate(outs)
    y_true = np.concatenate(tgts)
    return rmse_torch(y_true, y_pred)      # works on NumPy arrays


# ------------------------------------------------- objective for Optuna
def objective(trial, ds_full, outer_train_idx, df, device):
    hidden   = trial.suggest_int ('hidden' ,  64, 512, log=True)
    steps    = trial.suggest_int ('steps'  ,   2,  8)
    lr       = trial.suggest_float('lr', 1e-5, 3e-4, log=True)
    drop_mp  = trial.suggest_float('d_mp'  ,0.0 ,0.1 )
    drop_hd  = trial.suggest_float('d_head',0.0 ,0.3 )

    in_node = ds_full[0].x.size(1)
    in_edge = ds_full[0].edge_attr.size(1)

    rmses = []
    for k, (tr_idx, va_idx) in enumerate(
            inner_5fold_indices(outer_train_idx, df, seed=0)):
        tr_ds = Subset(ds_full, tr_idx)   # <- instead of index_select
        va_ds = Subset(ds_full, va_idx)
        tr_loader=DataLoader(tr_ds,batch_size=64,shuffle=True ,collate_fn=Batch.from_data_list, num_workers=4, pin_memory=True)
        va_loader=DataLoader(va_ds,batch_size=64,shuffle=False,collate_fn=Batch.from_data_list, num_workers=4, pin_memory=True)

        enc, head = build_model(in_node,in_edge,hidden,steps,drop_mp,drop_hd,device)
        opt = torch.optim.Adam(list(enc.parameters())+list(head.parameters()),
                               lr=lr)
        try:
            for epoch in range(10):                    # short budget
                fit_one_epoch(tr_loader, enc, head, opt, device)
                if epoch == 4:                          # cheap early-stop check
                    rmse_mid = evaluate(va_loader, enc, head, device)
                    trial.report(rmse_mid, step=k*10+epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
        except (RuntimeError, ValueError):             # NaNs, divergence …
            return 1e9
        rmse = evaluate(va_loader, enc, head, device)
        rmses.append(rmse)
    return float(np.mean(rmses))


# ------------------------------------------------- main pipeline
def main(args):
    Path("logs").mkdir(exist_ok=True)

    df = pd.read_csv(args.data)
    outer_train_idx, outer_test_idx = outer_scaffold_split(df, test_ratio=.20,
                                                           seed=42)

    # Dataset & scaling fitted on outer-train
    ds_full = CMPNNDataset(root='.', csv_file=os.path.basename(args.data),
                           force_reload=True)
    mean, std = ds_full.compute_normalization(outer_train_idx)
    ds_full.apply_normalization()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.getenv("FORCE_CPU", "0") == "1":
        device = torch.device('cpu')

    # -------------------- hyper-parameter search on inner CV
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
)
    
    study.optimize(
        lambda t: objective(t, ds_full, outer_train_idx, df, device),
        n_trials=args.n_trials, show_progress_bar=True)

    split = inner_5fold_indices(outer_train_idx, df, seed=0)
    for tr_idx, va_idx in split:
        assert max(tr_idx+va_idx) < len(ds_full), \
            "inner indices are still local – they must be global!"
    best = study.best_params
    print("Best inner-CV params:\n", best)

    # -------------------- train K seeds on outer-train + val
    seeds = [42, 7, 13, 21, 99]
    in_node = ds_full[0].x.size(1); in_edge = ds_full[0].edge_attr.size(1)
    train_ds = Subset(ds_full, outer_train_idx)
    test_ds  = Subset(ds_full, outer_test_idx)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              collate_fn=Batch.from_data_list)
    test_loader  = DataLoader(test_ds , batch_size=64, shuffle=False,
                              collate_fn=Batch.from_data_list)

    preds_all = []
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd); random.seed(sd)
        enc, head = build_model(in_node,in_edge,
                                best['hidden'], best['steps'],
                                best['d_mp'], best['d_head'], device)
        opt = torch.optim.Adam(list(enc.parameters())+list(head.parameters()),
                               lr=best['lr'])
        for _ in range(args.epochs):
            fit_one_epoch(train_loader, enc, head, opt, device)
        # prediction
        preds = []
        enc.eval(); head.eval()
        with torch.no_grad():                              # ① turn grad off
            for b in test_loader:
                b = b.to(device)
                z = enc.embed(b.x, b.edge_index, b.edge_attr, b.batch)
                preds.append(
                    head(z).view(-1).detach().cpu().numpy()  # ② detach before NumPy
                )
        preds_all.append(np.concatenate(preds))

    y_pred_norm = np.mean(np.stack(preds_all), axis=0)
    y_true_norm = np.concatenate([d.y.numpy() for d in test_ds])

    # inverse-scale
    y_pred = y_pred_norm * std + mean
    y_true = y_true_norm * std + mean

    rmse = rmse_torch(y_true, y_pred)
    mae  = mae_torch (y_true, y_pred)
    r2   = r2_score_torch(y_true, y_pred)

    # -------------------- bootstrap CI
    boot = []
    for _ in range(10_000):
        yt, yp = resample(y_true, y_pred)
        boot.append(rmse_torch(yt, yp))
    ci_low, ci_hi = np.percentile(boot, [2.5, 97.5])

    print(f"\nOUTER-TEST RMSE = {rmse:.3f} kcal/mol "
          f"[{ci_low:.3f}, {ci_hi:.3f}] (95 % CI)")

    # save artefacts
    out_csv = pd.DataFrame({
        'smiles': df.smiles.iloc[outer_test_idx].values,
        'y_true': y_true,
        'y_pred': y_pred,
    })
    out_csv.to_csv("logs/outer_test_predictions.csv", index=False)

    meta = dict(
    torch_version = torch.__version__,
    cuda          = torch.version.cuda,
    optuna        = optuna.__version__,
    python        = platform.python_version(),
    seeds         = seeds,
    search_space  = dict(
        hidden=[64,512], steps=[2,8], lr=[1e-4,3e-3],
        d_mp=[0.0,0.1], d_head=[0.0,0.3]),
    n_trials      = args.n_trials,
    train_epochs  = args.epochs,
    batch_size    = 64,
    target_unit   = "kcal/mol",
    csv_sha1      = hashlib.sha1(open(args.data,'rb').read()).hexdigest()[:12],
)
    metrics = dict(rmse=float(rmse), mae=float(mae), r2=float(r2),
                ci_low=float(ci_low), ci_hi=float(ci_hi),
                best_params=best, **meta)
    with open("logs/outer_test_metrics.json","w") as f:
        json.dump(metrics, f, indent=2)



# ------------------------------------------------- CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="raw/SAMPL.csv")
    p.add_argument("--n-trials", type=int, default=40,
                   help="Optuna trials for inner CV")
    p.add_argument("--epochs", type=int, default=40,
                   help="Epochs for final seed training")
    args = p.parse_args()
    main(args)
