#!/usr/bin/env python
import argparse
import json
import os
import numpy as np
import torch
import optuna
from cross_val import run_scaffold_cv
import random

def objective(trial, args):
    # Suggest hyperparameters
    hidden_dim      = trial.suggest_int('hidden_dim', 128, 512, log=True)
    num_steps       = trial.suggest_int('num_steps', 1, 10)
    head_hidden_dim = trial.suggest_int('head_hidden_dim', 32, 256, log=True)
    lr              = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout_mp      = trial.suggest_float('dropout_mp', 0.0, 0.5)
    dropout_head    = trial.suggest_float('dropout_head', 0.0, 0.5)

    # set fixed RNG seeds for reproducibility
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Run short CV (3-fold, 1 repeat), catch failures as NaN
    try:
        # run ensemble CV to match final evaluation
        rmses = run_scaffold_cv(
            args.data_path,
            n_splits=3,
            n_repeats=1,
            device=args.device,
            atom_messages=args.atom_messages,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            head_hidden_dim=head_hidden_dim,
            lr=lr,
            trial=trial,
            ensemble=True,
            n_epochs=50,
            patience=10,
            dropout_mp=dropout_mp,
            dropout_head=dropout_head
        )
        rmse_vals = [r for _, _, r in rmses]
        # record per-fold RMSEs
        trial.set_user_attr('fold_rmses', rmse_vals)
        # penalize high variance among folds
        mean_rmse = float(np.mean(rmse_vals))
        std_rmse = float(np.std(rmse_vals))
        alpha = 0.5
        return mean_rmse + alpha * std_rmse
    except Exception as e:
        print(f"Trial failed with exception: {e}")
        # return a large value to mark this trial as poor
        return float('inf')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter optimization via Optuna')
    parser.add_argument('--data-path', type=str, default='raw/SAMPL.csv')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--atom-messages', action='store_true')
    args = parser.parse_args()

    # use pruning to stop poor trials early
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.SuccessiveHalvingPruner()
    )
    # run optimization, passing args via lambda
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    # Save best params
    os.makedirs('logs', exist_ok=True)
    best = study.best_params
    with open('logs/best_hparams.json', 'w') as f:
        json.dump(best, f, indent=2)
    print('Best hyperparameters:', best)

    # Final evaluation with 5x5 CV
    print('Running final 5x5 CV with best params...')
    results = run_scaffold_cv(
        args.data_path,
        n_splits=5,
        n_repeats=5,
        device=args.device,
        atom_messages=args.atom_messages,
        hidden_dim=best['hidden_dim'],
        num_steps=best['num_steps'],
        head_hidden_dim=best['head_hidden_dim'],
        lr=best['lr'],
        dropout_mp=best['dropout_mp'],
        dropout_head=best['dropout_head']
    )
    # Aggregate and save
    mean_rmse = np.mean([r for _, _, r in results])
    std_rmse  = np.std([r for _, _, r in results])
    summary = {'mean_rmse': mean_rmse, 'std_rmse': std_rmse}
    with open('logs/final_cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('Final CV mean RMSE:', mean_rmse, 'std:', std_rmse)