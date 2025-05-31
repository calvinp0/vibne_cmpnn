#!/usr/bin/env python

"""
Scaffold split cross-validation with a *seed ensemble* identical to the
Lightning pipeline (5 independent models per fold, predictions averaged).

Usage
-----
python cross_val.py                     # 5×5 CV, 5-seed ensemble
python cross_val.py --seeds 0 1 2       # custom seed list
python cross_val.py -n-splits 3 -n-repeats 1      # quicker smoke-run
"""
from __future__ import annotations
import os, random, argparse, json, math, warnings
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset        import CMPNNDataset
from model          import CMPNNEncoder, FFNHead, initialize_xavier
from preprocessing  import scaffold_split_indices_repeated
from utils          import rmse_torch, mse_torch, mae_torch, r2_score_torch


torch.autograd.set_detect_anomaly(True)  # track operation producing NaNs or infs

def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_scaffold_cv(
    df_path      : str,
    *,
    n_splits     : int   = 5,
    n_repeats    : int   = 5,
    seeds        : Sequence[int] = (42, 7, 13, 21, 99),
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    hidden_dim   : int   = 300,
    num_steps    : int   = 5,
    head_hidden  : int   = 64,
    lr           : float = 1e-4,
    dropout_mp   : float = 0.05,
    dropout_head : float = 0.10,
    n_epochs     : int   = 200,
    patience     : int   = 25,
) -> list[tuple[int,int,float]]:
    df       = pd.read_csv(df_path)
    full_ds  = CMPNNDataset(root='.', csv_file=os.path.basename(df_path),
                        atom_messages=False, force_reload=True)

    in_node_feats = full_ds[0].x.size(1)
    in_edge_feats = full_ds[0].edge_attr.size(1)

    splits = scaffold_split_indices_repeated(df,
                                            n_splits = n_splits,
                                            n_repeats= n_repeats)

    results : list[tuple[int,int,float]] = []

    for rep_idx, folds in enumerate(splits, 1):
        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds, 1):
            print(f"\n=== repeat {rep_idx}  fold {fold_idx} ===")

            # ── 2.1  normalise targets on *train* only
            mean, std = full_ds.compute_normalization(train_idx)
            full_ds.apply_normalization()

            train_ds = Subset(full_ds, train_idx)
            val_ds   = Subset(full_ds, val_idx)
            test_ds  = Subset(full_ds, test_idx)

            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                                      collate_fn=Batch.from_data_list)
            val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                                      collate_fn=Batch.from_data_list)
            test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False,
                                      collate_fn=Batch.from_data_list)
            preds_per_seed : list[np.ndarray] = []

            for seed in seeds:
                print(f"  └─ seed {seed}")
                _set_all_seeds(seed)
                enc  = CMPNNEncoder(in_node_feats, in_edge_feats,
                                    hidden_dim=hidden_dim,
                                    num_steps=num_steps,
                                    dropout=dropout_mp).to(device)
                head = FFNHead(in_dim=hidden_dim*2,
                               hidden_dim=head_hidden,
                               out_dim=1,
                               dropout=dropout_head).to(device)
                enc.apply(initialize_xavier)
                head.apply(initialize_xavier)

                optim = torch.optim.Adam(list(enc.parameters())+list(head.parameters()), lr=lr)
                sched = ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=10)

                best_val = math.inf
                best_enc_state, best_head_state = None, None
                wait = 0

                for epoch in range(1, n_epochs+1):
                    enc.train(); head.train()
                    for batch in tqdm(train_loader, leave=False):
                        batch = batch.to(device)
                        optim.zero_grad()
                        z = enc.embed(batch.x, batch.edge_index,
                                      batch.edge_attr, batch.batch)
                        y_pred = head(z).view(-1)
                        loss   = torch.nn.functional.mse_loss(y_pred, batch.y.view(-1))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(enc.parameters(),   5.)
                        torch.nn.utils.clip_grad_norm_(head.parameters(),  5.)
                        optim.step()

                    # ─ validation
                    enc.eval(); head.eval()
                    vloss = 0.0
                    with torch.no_grad():
                        for batch in val_loader:
                            batch = batch.to(device)
                            z = enc.embed(batch.x, batch.edge_index,
                                          batch.edge_attr, batch.batch)
                            vloss += torch.nn.functional.mse_loss(
                                head(z).view(-1), batch.y.view(-1)
                            ).item() * batch.num_graphs
                    vloss /= len(val_ds)
                    sched.step(vloss)


                    if vloss < best_val:
                        best_val = vloss
                        best_enc_state  = {k:v.cpu() for k,v in enc.state_dict().items()}
                        best_head_state = {k:v.cpu() for k,v in head.state_dict().items()}
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            break

                # ── 2.2.1 inference on test set for this seed ------------------
                enc.load_state_dict(best_enc_state);  enc.to(device).eval()
                head.load_state_dict(best_head_state); head.to(device).eval()

                preds_seed = []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        z = enc.embed(batch.x, batch.edge_index,
                                      batch.edge_attr, batch.batch)
                        preds_seed.append(head(z).view(-1).cpu().numpy())
                preds_per_seed.append(np.concatenate(preds_seed))

            # ── 2.3  ensemble predictions & metric -----------------------------
            y_pred_scaled = np.mean(np.stack(preds_per_seed, axis=0), axis=0)
            y_true_scaled = np.concatenate([d.y.numpy() for d in test_ds])

            y_pred = y_pred_scaled * std + mean
            y_true = y_true_scaled * std + mean

            rmse = rmse_torch(y_true, y_pred)
            results.append((rep_idx, fold_idx, float(rmse)))
            print(f"      fold RMSE  =  {rmse:6.3f}  kcal mol-1")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="raw/SAMPL.csv")
    parser.add_argument("-n-splits",  type=int, default=5)
    parser.add_argument("-n-repeats", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42,7,13,21,99])
    args = parser.parse_args()

    res = run_scaffold_cv(args.data,
                          n_splits = args.n_splits,
                          n_repeats= args.n_repeats,
                          seeds    = args.seeds)
    df_res = pd.DataFrame(res, columns=["repeat","fold","rmse"])
    df_res.to_csv("cv_results.csv", index=False)

    print("\nmean ± std  :  ",
          df_res.rmse.mean(), "±", df_res.rmse.std(), "kcal mol-1")

