#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import random
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.data import Subset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
from utils import RMSELoss

from preprocessing import scaffold_split_indices_repeated
from dataset import CMPNNDataset
from pl_module import CMPNNLitModel

# random seeds for ensemble
SEEDS = [42, 7, 13, 21, 99]

# Trainer settings
TRAINER_PARAMS = dict(
    max_epochs=100,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    logger=False,
    callbacks=[EarlyStopping(monitor='val_rmse_unscaled', patience=10)],
    enable_checkpointing=False,
)

# Lightning metrics to log (first is loss)
METRICS = [RMSELoss(), MeanSquaredError(squared=True), MeanAbsoluteError(), R2Score()]

def ensemble_cross_val(
    df_path: str,
    seeds: list = SEEDS,
    n_splits: int = 5,
    n_repeats: int = 5,
    batch_size: int = 64,
    hidden_dim: int = 300,
    num_steps: int = 5,
    dropout_mp: float = 0.05,
    dropout_head: float = 0.1,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    max_epochs: int = 100,
    early_stop_patience: int = 10
):
    """
    Perform a scaffold cross-validation ensemble across multiple random seeds.
    Returns list of fold RMSE values.
    """
    # setup trainer params inside function
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    trainer_params = {
        # 'max_epochs': max_epochs,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 1,
        'logger': False,
        'callbacks': [EarlyStopping(monitor='val_meansquarederror', patience=early_stop_patience)],
        'enable_checkpointing': False,
        'enable_model_summary': False,
        'num_sanity_val_steps': 0,
    }
    # load data
    df = pd.read_csv(df_path)
    full_ds = CMPNNDataset(root='.', csv_file=os.path.basename(df_path), force_reload=True)
    in_node_feats = full_ds.num_atom_features
    in_edge_feats = full_ds.num_bond_features
    n_tasks = full_ds.num_targets
    splits = scaffold_split_indices_repeated(df, n_splits=n_splits, n_repeats=n_repeats)
    fold_rmses = []
    for rep_idx, folds in enumerate(splits, start=1):
        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
            print(f"=== Repeat {rep_idx} / Fold {fold_idx} ===")
            mean, std = full_ds.compute_normalization(train_idx)
            full_ds.apply_normalization()
            train_ds = Subset(full_ds, train_idx)
            val_ds   = Subset(full_ds, val_idx)
            test_ds  = Subset(full_ds, sorted(test_idx))
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=Batch.from_data_list)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list)
            test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list)
            preds_per_seed = []
            for seed in seeds:
                print(f"  Seed {seed}")
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                model = CMPNNLitModel(
                    in_node_feats, in_edge_feats,
                    target_mean=mean, target_std=std,
                    hidden_dim=hidden_dim, num_steps=num_steps,
                    dropout_mp=dropout_mp, dropout_head=dropout_head,
                    n_tasks=n_tasks, metrics=METRICS,
                    lr=lr, weight_decay=weight_decay,
                    readout='gru', use_booster=True
                )

                trainer = Trainer(
                    max_epochs=max_epochs,
                    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                    devices=1,
                    logger=False,
                    callbacks=[EarlyStopping(monitor='val_rmse_unscaled', patience=early_stop_patience)],
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    num_sanity_val_steps=0,
                )

                trainer.fit(model, train_loader, val_loader)
                preds_batches = trainer.predict(model, test_loader)
                preds = np.concatenate([batch.cpu().numpy() for batch in preds_batches], axis=0)
                preds_per_seed.append(preds)

                # manual inference without Lightning predict bar to collect seed predictions

            y_pred_ensemble = np.stack(preds_per_seed, axis=0).mean(axis=0)
            y_true = np.concatenate([batch.y.cpu().numpy() for batch in test_loader])
            # unscale predictions and true values to original units
            #y_pred_un = y_pred * std + mean
            y_true_un = y_true * std + mean
            rmse = np.sqrt(((y_pred_ensemble - y_true_un)**2).mean())
            fold_rmses.append(rmse)
            print(f"  Fold RMSE: {rmse:.3f}")
    return fold_rmses

if __name__ == '__main__':
    # call the ensemble CV function with defaults or custom params
    fold_rmses = ensemble_cross_val(
        df_path='raw/SAMPL.csv'
    )
    mean_cv = np.mean(fold_rmses)
    std_cv  = np.std(fold_rmses)
    print(f'Ensemble Cross-Validation RMSE: {mean_cv:.3f} ± {std_cv:.3f} kcal/mol')
