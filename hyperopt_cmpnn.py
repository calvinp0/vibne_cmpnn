# ─────────────────────── install / import ─────────────────────────
# (run once per env; commented out here)
# !pip install optuna wandb pytorch-lightning[extra] optuna-dashboard

import os
import sys
import argparse
from sklearn.preprocessing import PowerTransformer
import torch
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
# Your local modules
from dataset import (
    CMPNNDataset,
    MultiCMPNNDataset,
    MultiCMPNNDatasetSDF,
    CMPNNDataModule,
)
from preprocessing import scaffold_split_indices, random_split_indices, scaffold_cross_validation_repeated, kennard_stone_cross_validation_repeated
from pl_module import MultiCMPNNLitModel
from utils_paired import collate_pairs
from metrics import MSE, MAE, RMSE, R2Score, PinballLoss
from sklearn.model_selection import KFold
import os
from rdkit import Chem
from itertools import chain
# ────────────────── Argument Parsing ──────────────────────────────
parser = argparse.ArgumentParser(
    description="Hyperparameter search + final training/testing for Multi-CMPNN model"
)
parser.add_argument(
    "--label",
    type=str,
    choices=["k_rev (TST+T)", "k_for (TST+T)", "default"],
    default="k_for (TST+T)",
    help="Which label-computation to use (e.g. compute k_rev via TST+T or use a default target).",
)
parser.add_argument(
    "--wandb-project",
    type=str,
    default="cmpnn-hyperopt",
    help="Weights & Biases project name to log into (offline mode).",
)
parser.add_argument(
    "--wandb-prefix",
    type=str,
    default="",
    help="Prefix for all W&B run names (helps group trials).",
)
parser.add_argument(
    "--db-name",
    type=str,
    default="cmpnn_optuna.db",
    help="Filename for the Optuna SQLite database (will be created relative to cwd).",
)

parser.add_argument(
    "--extra-feats",
    type=str,
    default="/home/calvin.p/Code/dmpnn_customized/DATA/sdf_data/all_sdf_features.csv",
    help="Path to CSV or file containing extra atom‐level features. If omitted, use the default.",
)

args = parser.parse_args()

# ─────────────────────── Initialization ───────────────────────────
pl.seed_everything(42)  # for reproducibility
# Force W&B into offline mode so nothing goes to the cloud
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB__SERVICE_WAIT"] = "300"  # quiet-down wait

import wandb  # noqa: E402
import pathlib  # noqa: E402

# ────────────────── Helper: log_final_model ───────────────────────
def log_final_model(checkpoint_path: str, rank: int, val_loss: float):
    """
    Creates an offline W&B run named with the global prefix + final rank,
    then logs the checkpoint file as an Artifact.
    """
    run_name = f"{args.wandb_prefix}_final_rank{rank}" if args.wandb_prefix else f"final_rank{rank}"
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        mode="offline",
        config={"rank": rank, "val_loss": val_loss, "label_type": args.label},
    )

    artifact_name = f"{args.wandb_prefix}_cmpnn_final_rank{rank}" if args.wandb_prefix else f"cmpnn_final_rank{rank}"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata={"val_loss": val_loss, "label_type": args.label},
    )
    artifact.add_file(checkpoint_path)
    run.log_artifact(artifact)
    run.finish()


# ──────────────────── Dataset Constructor ─────────────────────────
SDF_DIR = "/home/calvin.p/Code/dmpnn_customized/DATA/sdf_data"

TARGETS_PATH = "/home/calvin.p/Code/Data/target_data/kinetics_summary.csv"

TARGETS_COLS = ["A_log10", "n", "Ea_yj"]
TARGETS_TYPES = {"A_log10": "continuous", "n": "continuous", "Ea_yj": "continuous"}
# EXTRA_FEATS = "/home/calvin.p/Code/dmpnn_customized/DATA/sdf_data/all_sdf_features.csv"


def extract_donor_smiles_list(ds):
    """Returns list of SMILES for the donor molecule (type == 'r1h') in each sample."""
    smiles_list = []
    for i in range(len(ds)):
        mol1, mol2 = ds[i]
        smi = mol1.smiles
        smiles_list.append(smi)

    return smiles_list

def make_dataset(atom_messages: bool, extra_feats_path: str, label_type: str) -> MultiCMPNNDatasetSDF:
    # Read the CSV into a local DataFrame called targets_df
    targets_df = pd.read_csv(TARGETS_PATH)

    # Filter by label_type if that column exists
    if "label" in targets_df.columns:
        # Keep only rows matching the chosen label
        targets_df = targets_df[targets_df["label"] == label_type]
    else:
        raise ValueError("No label column")

    if not pathlib.Path(extra_feats_path).exists():
        raise FileNotFoundError(f"Extra features file not found: {extra_feats_path}")

    targets_df = targets_df.dropna(subset=['rxn', 'label', 'A', 'n', 'Ea'])

    # 3) Dedupe by (rxn, label), keeping the first occurrence
    targets_df = targets_df.drop_duplicates(subset=['rxn','label'], keep='first')


    # 4) Filter out non-positive A (cannot log-transform those)
    targets_df = targets_df[targets_df['A'] > 0].copy()


    # 5) Log10-transform A into a new column
    targets_df['A_log10'] = np.log10(targets_df['A'])

    # 6) Fit & apply Yeo–Johnson transform to the entire Ea column
    # Fit on a pure numpy array
    Ea_vals = targets_df['Ea'].to_numpy().reshape(-1, 1)   # ndarray, no column names
    pt_ea = PowerTransformer(method='yeo-johnson')
    Ea_yj = pt_ea.fit_transform(Ea_vals).ravel()
    targets_df['Ea_yj'] = Ea_yj



    ds = MultiCMPNNDatasetSDF(
        root=".",
        sdf_files=SDF_DIR,
        target_df=targets_df,
        target_cols=TARGETS_COLS,
        target_types=TARGETS_TYPES,
        atom_messages=atom_messages,
        keep_hs=True,
        sanitize=True,
        force_reload=False,
        prune_value=-10,
        atom_extra_feats=extra_feats_path,
        rbf_num_centers=16,
    )

    reactions_in_dataset = []
    for i in range(len(ds)):
        mol1, _ = ds[i]
        reactions_in_dataset.append(mol1.reaction)

    targets_df = targets_df[targets_df["rxn"].isin(reactions_in_dataset)].copy()
    targets_df = targets_df.reset_index(drop=True)  # match new row order

    # Update donor_smiles
    donor_smiles = extract_donor_smiles_list(ds)
    if len(donor_smiles) != len(targets_df):
        raise ValueError(f"Mismatch: {len(donor_smiles)} SMILES vs {len(targets_df)} targets")
    targets_df["smiles"] = donor_smiles
    ds.target_df = targets_df  # overwrite with matched version

    return ds


# ────────────────── Optuna Objective Function ─────────────────────
def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective now runs a 5-fold CV on ds_trial.
    We build 5 disjoint train/val splits of the entire dataset,
    train a separate model on each, record each fold’s val_loss,
    and return their mean.
    """
    # If a previous W&B run is still open, close it
    if wandb.run is not None:
        wandb.finish()


    SPLIT_SAVE_DIR = f"./saved_splits/{args.label}/trial_{trial.number}"
    os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        name=f"{args.wandb_prefix}trial{trial.number}" if args.wandb_prefix else f"trial{trial.number}",
        mode="offline",
        reinit=True,
    )
    run.config.update({"label_type": args.label, **trial.params}, allow_val_change=True)

    # ───── Sample hyperparameters ─────────────────────────────────
    hparams = dict(
        hidden_dim=trial.suggest_int("hidden_dim", 256, 1024, step=256),
        head_layers=trial.suggest_int("head_layers", 1, 3),
        dropout_mp=trial.suggest_float("dropout_mp", 0.0, 0.4, step=0.05),
        dropout_head=trial.suggest_float("dropout_head", 0.0, 0.4, step=0.05),
        lr=trial.suggest_loguniform("lr", 1e-5, 3e-3),
        weight_decay=trial.suggest_loguniform("weight_decay", 1e-6, 3e-4),
        mpn_shared=trial.suggest_categorical("mpn_shared", [True, False]),
        readout=trial.suggest_categorical("readout", ["sum", "gru"]),
        atom_messages=trial.suggest_categorical("atom_messages", [True, False]),
        use_booster=trial.suggest_categorical("use_booster", [True, False]),
        use_residual=trial.suggest_categorical("use_residual", [True, False]),
        use_graph_residual=trial.suggest_categorical("use_graph_residual", [True, False]),
        pinball_q=trial.suggest_float("pinball_q", 0.80, 0.95),
        pinball_weight=trial.suggest_float("pinball_weight", 0.10, 0.50),
        num_steps=trial.suggest_int("num_steps", 3, 6),
    )

    # ───── Build the full dataset for this trial ────────────────────
    ds_trial = make_dataset(
        atom_messages=hparams["atom_messages"],
        extra_feats_path=args.extra_feats,
        label_type=args.label,
    )
    donor_smiles = extract_donor_smiles_list(ds_trial)
    ds_trial.target_df["smiles"] = donor_smiles  # insert column needed for scaffold CV
    # ───── Metrics list (weights are fixed here) ────────────────────
    metrics_list = [
        MSE(task_weights=[10, 1, 1]),
        PinballLoss(q=hparams["pinball_q"], task_weights=[10, 1, 1]),
        RMSE([10, 1, 1]),
        MAE([10, 1, 1]),
        R2Score([10, 1, 1]),
    ]

    # ───── Run 5-Fold CV ────────────────────────────────────────────
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_losses, fold_maes, fold_r2s = [], [], []

    df_targets = ds_trial.target_df  # or however you're accessing the targets
    ks_folds = list(chain.from_iterable(
    kennard_stone_cross_validation_repeated(df_targets, n_splits=5, n_repeats=3, seed=42)
))
    for fold_idx, (train_df, val_df) in enumerate(ks_folds):
        train_idx = train_df.index.tolist()
        val_idx = val_df.index.tolist()
        ds_fold = make_dataset(
            atom_messages=hparams["atom_messages"],
            extra_feats_path=args.extra_feats,
            label_type=args.label,
        )
        
        train_df[['rxn', 'smiles']].to_csv(f"{SPLIT_SAVE_DIR}/fold{fold_idx+1}_train.csv", index=False)
        val_df[['rxn', 'smiles']].to_csv(f"{SPLIT_SAVE_DIR}/fold{fold_idx+1}_val.csv", index=False)
        ds_fold.attach_atom_extra_features(train_idx)
        ds_fold.compute_normalization(train_idx)
        ds_fold.apply_normalization()

        train_loader = DataLoader(
            Subset(ds_fold, train_idx),
            batch_size=64, shuffle=True,
            collate_fn=collate_pairs, pin_memory=True, num_workers=16
        )
        val_loader = DataLoader(
            Subset(ds_fold, val_idx),
            batch_size=32, shuffle=False,
            collate_fn=collate_pairs, pin_memory=True, num_workers=16
        )

        model = MultiCMPNNLitModel(
            in_node_feats=ds_fold.num_atom_features,
            in_edge_feats=ds_fold.num_bond_features,
            hidden_dim=hparams["hidden_dim"],
            num_steps=hparams["num_steps"],
            dropout_mp=hparams["dropout_mp"],
            dropout_head=hparams["dropout_head"],
            head_layers=hparams["head_layers"],
            n_tasks=3,
            metrics=[
                MSE(task_weights=[10,1,1]),
                PinballLoss(q=hparams["pinball_q"], task_weights=[10,1,1]),
                RMSE([10,1,1]),
                MAE([10,1,1]),
                R2Score([10,1,1]),
            ],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            mpn_shared=hparams["mpn_shared"],
            readout=hparams["readout"],
            use_booster=hparams["use_booster"],
            use_residual=hparams["use_residual"],
            use_graph_residual=hparams["use_graph_residual"],
            target_mean=ds_fold.mean,
            target_std=ds_fold.std,
            pinball_weight=hparams["pinball_weight"],
        )

        ckpt_cb  = ModelCheckpoint(monitor="val_loss", mode="min")
        early_cb = EarlyStopping(monitor="val_loss", patience=8, mode="min")
        class NanKiller(pl.callbacks.Callback):
            def on_validation_end(self, trainer, pl_module):
                loss = trainer.callback_metrics.get("val_loss")
                if loss is not None and torch.isnan(loss):
                    raise optuna.exceptions.TrialPruned()

        trainer = pl.Trainer(
            max_epochs=60,
            accelerator="auto",
            devices=1,
            logger=None,
            enable_progress_bar=False,
            callbacks=[early_cb, ckpt_cb, NanKiller()],
        )
        trainer.fit(model, train_loader, val_loader)

        # Grab all three metrics from callback_metrics
        val_loss = ckpt_cb.best_model_score.item()        # <— use BEST, not last
        val_mae  = trainer.callback_metrics["val_mae"].item()
        val_r2   = trainer.callback_metrics["val_r2"].item()

        fold_losses.append(val_loss)
        fold_maes.append(val_mae)
        fold_r2s.append(val_r2)

    # Compute averages across the 5 folds
    mean_loss = float(np.mean(fold_losses))
    mean_mae  = float(np.mean(fold_maes))
    mean_r2   = float(np.mean(fold_r2s))

    # Store them in the Optuna DB for this trial
    trial.set_user_attr("mean_mae", mean_mae)
    trial.set_user_attr("mean_r2",  mean_r2)

    metrics_to_log = {
        f"fold{idx+1}_val_loss": l for idx, l in enumerate(fold_losses)
    } | {
        f"fold{idx+1}_val_mae":  m for idx, m in enumerate(fold_maes)
    } | {
        f"fold{idx+1}_val_r2":   r for idx, r in enumerate(fold_r2s)
    }

    metrics_to_log.update({
        "avg_val_loss": float(np.mean(fold_losses)),
        "avg_val_mae":  float(np.mean(fold_maes)),
        "avg_val_r2":   float(np.mean(fold_r2s)),
    })
    run.log(metrics_to_log)      # one history row
    run.finish()
    return metrics_to_log["avg_val_loss"]



# ───────────────────── run / resume the study ──────────────────────
storage_url = f"sqlite:///{args.db_name}"
study = optuna.create_study(
    study_name="cmpnn_full",
    direction="minimize",
    storage=storage_url,
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
)


# ── guard: study may be empty on first run
# ── guard: study may be empty *or* contain only pruned/failed trials
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

if completed:
    old_best = min(t.value for t in completed)
    print(f"Best completed loss so far: {old_best:.4f}")
else:
    old_best = None
    print("No completed trials yet – starting fresh.")

# Run 50 trials or until 6 hours elapse
study.optimize(objective, n_trials=150, timeout=6 * 3600)

# ───────────────────── top-k retrain ─────────────────────────────
# ───────────────────── top-3 retrain w/ 10-fold CV ─────────────────────
TOPK = 3
# Sort Optuna trials by their 5-fold CV average (t.value)
completed_trials = [t for t in study.trials if t.value is not None]
top_trials = sorted(completed_trials, key=lambda t: t.value)[:TOPK]

print("\n=== Top-3 Optuna Trials (by 5-fold CV) ===")
for rank, tr in enumerate(top_trials, 1):
    print(f"Rank {rank}: trial #{tr.number} → 5-fold val_loss = {tr.value:.4f}")
    print("  params:", tr.params)
print("=========================================\n")

# We'll store (trial_number, mean_10fold_loss) for each of the top 3
results_10fold = []

for rank, tr in enumerate(top_trials, 1):
    hp = tr.params
    print(f"\n--- Running 10-fold CV for top trial #{tr.number} (rank={rank}) ---")

    # Build the dataset using that trial's hparams
    ds_best = make_dataset(
        atom_messages=hp["atom_messages"],
        extra_feats_path=args.extra_feats,
        label_type=args.label,
    )

    # Prepare the same metrics list
    metrics_list = [
        MSE(task_weights=[10, 1, 1]),
        PinballLoss(q=hp["pinball_q"], task_weights=[10, 1, 1]),
        RMSE([10, 1, 1]),
        MAE([10, 1, 1]),
        R2Score([10, 1, 1]),
    ]

    cv_run = wandb.init(
        project=args.wandb_project,
        name=f"{args.wandb_prefix}_10fold_rank{rank}" if args.wandb_prefix else f"10fold_rank{rank}",
        mode="offline",
    )
    cv_run.config.update({**hp, "rank": rank, "trial_number": tr.number, "label_type": args.label}, allow_val_change=True)

    # 10-fold splitter over the entire dataset
    ds_base_10fold = make_dataset(
    atom_messages=hp["atom_messages"],
    extra_feats_path=args.extra_feats,
    label_type=args.label,
)
    donor_smiles_10 = extract_donor_smiles_list(ds_base_10fold)
    ds_base_10fold.target_df["smiles"] = donor_smiles_10
    ks_folds = list(chain.from_iterable(
    kennard_stone_cross_validation_repeated(ds_base_10fold.target_df, n_splits=5, n_repeats=3, seed=42)
))
    fold_losses_10 = []  # to store each fold's val_loss
    for fold_idx, (train_df, val_df) in enumerate(ks_folds):
       # Rebuild a fresh dataset for this fold
       train_idx = train_df.index.tolist()
       val_idx = val_df.index.tolist()
       ds_fold10 = make_dataset(
           atom_messages=hp["atom_messages"],
           extra_feats_path=args.extra_feats,
           label_type=args.label,
       )

       # Attach + normalize on ds_fold10 using train_idx
       ds_fold10.attach_atom_extra_features(train_idx)
       ds_fold10.compute_normalization(train_idx)
       ds_fold10.apply_normalization()

       # Build DataLoaders
       train_loader = DataLoader(
            Subset(ds_fold10, train_idx),
            batch_size=64,
            shuffle=True,
            collate_fn=collate_pairs,
            pin_memory=True,
            num_workers=4
        )
       val_loader = DataLoader(
            Subset(ds_fold10, val_idx),
            batch_size=32,
            shuffle=False,
            collate_fn=collate_pairs,
            pin_memory=True,
            num_workers=4
        )

       # Build a fresh model with these fixed hyperparams
       model_10 = MultiCMPNNLitModel(
           in_node_feats=ds_fold10.num_atom_features,
           in_edge_feats=ds_fold10.num_bond_features,
           hidden_dim=hp["hidden_dim"],
           num_steps=hp["num_steps"],
           dropout_mp=hp["dropout_mp"],
           dropout_head=hp["dropout_head"],
           head_layers=hp["head_layers"],
           n_tasks=3,
           metrics=metrics_list,
           lr=hp["lr"],
           weight_decay=hp["weight_decay"],
           mpn_shared=hp["mpn_shared"],
           readout=hp["readout"],
           use_booster=hp["use_booster"],
           use_residual=hp["use_residual"],
           use_graph_residual=hp["use_graph_residual"],
           target_mean=ds_fold10.mean,
           target_std=ds_fold10.std,
           pinball_weight=hp["pinball_weight"],
       )

       # Trainer setup
       ckpt10  = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
       early10 = EarlyStopping(monitor="val_loss", patience=8, mode="min")
       class Nan10(pl.callbacks.Callback):
           def on_validation_end(self, trainer, pl_module):
               loss = trainer.callback_metrics.get("val_loss")
               if loss is not None and torch.isnan(loss):
                   raise optuna.exceptions.TrialPruned()

       trainer_10 = pl.Trainer(
           max_epochs=60,
           accelerator="auto",
           devices=1,
           logger=None,
           enable_progress_bar=False,
           callbacks=[early10, ckpt10, Nan10()],
       )

       # Train & validate
       trainer_10.fit(model_10, train_loader, val_loader)
       fold_val_loss_10 = ckpt10.best_model_score.item()
       fold_losses_10.append(fold_val_loss_10)
       mean_10 = float(np.mean(fold_losses_10))
       std_10  = float(np.std(fold_losses_10))
       cv_run.log({f"fold{fold_idx+1}_val_loss": fold_val_loss_10})
    fold_val_loss_10 = ckpt10.best_model_score.item()
    fold_losses_10.append(fold_val_loss_10)
    cv_run.log({f"fold{fold_idx+1}_val_loss": fold_val_loss_10})
    print(f"→ Trial #{tr.number} → 10-fold CV mean val_loss = {mean_10:.5f} ± {std_10:.5f}")

    results_10fold.append((tr.number, mean_10, std_10, hp))

# Pick the hyperparams with the lowest 10-fold mean
best_10 = min(results_10fold, key=lambda x: x[1])
best_trial_num, best_mean10, best_std10, best_hparams = best_10

print("\n=== FINAL CHOICE BASED ON 10-FOLD CV ===")
print(f"Trial #{best_trial_num} 10-fold mean val_loss = {best_mean10:.5f} ± {best_std10:.5f}")
print("Hyperparameters:", best_hparams)
print("=========================================\n")

# ─── FINAL SINGLE TRAIN/VAL/TEST ──────────────────────────────────────
ds_final = make_dataset(
    atom_messages=best_hparams["atom_messages"],
    extra_feats_path=args.extra_feats,
    label_type=args.label,
)

# 80/10/10 random split
# Load predefined reaction names for train/val/test
split_path = "/home/calvin/code/chemprop_phd_customised/habnet/data/processed/target_data/rxn_split.csv"
ds_names = pd.read_csv(split_path)

# Build reaction name → dataset index map
reaction_to_idx = {}
for idx in range(len(ds_final)):
    mol1, _ = ds_final[idx]
    reaction = mol1.reaction
    reaction_to_idx[reaction] = idx

# Extract index lists (dropna to avoid NaNs from padding)
train_idx = [reaction_to_idx[r] for r in ds_names['train'].dropna()]
val_idx   = [reaction_to_idx[r] for r in ds_names['val'].dropna()]
test_idx  = [reaction_to_idx[r] for r in ds_names['test'].dropna()]


# Attach + normalize on train_idx, apply to val/test
ds_final.attach_atom_extra_features(train_idx)
ds_final.compute_normalization(train_idx)
ds_final.apply_normalization()

# W&B run for final training run
final_run = wandb.init(
    project=args.wandb_project,
    name=f"{args.wandb_prefix}_final_train" if args.wandb_prefix else "final_train",
    mode="offline",
)
final_run.config.update({**best_hparams, "label_type": args.label}, allow_val_change=True)

train_loader = DataLoader(
    Subset(ds_final, train_idx),
    batch_size=64, shuffle=True,
    collate_fn=collate_pairs, pin_memory=True, num_workers=16
)
val_loader = DataLoader(
    Subset(ds_final, val_idx),
    batch_size=32, shuffle=False,
    collate_fn=collate_pairs, pin_memory=True, num_workers=16
)
test_loader = DataLoader(
    Subset(ds_final, test_idx),
    batch_size=32, shuffle=False,
    collate_fn=collate_pairs, pin_memory=True, num_workers=16
)

# Build final model with best_hparams
metrics_list = [
    MSE(task_weights=[10, 1, 1]),
    PinballLoss(q=best_hparams["pinball_q"], task_weights=[10, 1, 1]),
    RMSE([10, 1, 1]),
    MAE([10, 1, 1]),
    R2Score([10, 1, 1]),
]
final_model = MultiCMPNNLitModel(
    in_node_feats=ds_final.num_atom_features,
    in_edge_feats=ds_final.num_bond_features,
    hidden_dim=best_hparams["hidden_dim"],
    num_steps=best_hparams["num_steps"],
    dropout_mp=best_hparams["dropout_mp"],
    dropout_head=best_hparams["dropout_head"],
    head_layers=best_hparams["head_layers"],
    n_tasks=3,
    metrics=metrics_list,
    lr=best_hparams["lr"],
    weight_decay=best_hparams["weight_decay"],
    mpn_shared=best_hparams["mpn_shared"],
    readout=best_hparams["readout"],
    use_booster=best_hparams["use_booster"],
    use_residual=best_hparams["use_residual"],
    use_graph_residual=best_hparams["use_graph_residual"],
    target_mean=ds_final.mean,
    target_std=ds_final.std,
    pinball_weight=best_hparams["pinball_weight"],
)

# Train for a longer run (e.g. 300 epochs)
FULL_EPOCHS = 300
ckpt_final  = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
early_final = EarlyStopping(monitor="val_loss", patience=25, mode="min")
trainer_final = pl.Trainer(
    max_epochs=FULL_EPOCHS,
    accelerator="gpu",
    devices=1,
    logger=None,
    callbacks=[early_final, ckpt_final],
)
trainer_final.fit(final_model, train_loader, val_loader)

# Evaluate on hold-out test
best_ckpt = ckpt_final.best_model_path
best_val_loss = trainer_final.callback_metrics["val_loss"].item()


print(f"Final best model trained on train/val → val_loss = {best_val_loss:.4f}")

best_model = MultiCMPNNLitModel.load_from_checkpoint(best_ckpt, metrics=metrics_list)
test_trainer = pl.Trainer(accelerator="gpu", devices=1, logger=None)
test_results = test_trainer.test(best_model, test_loader)
print(f"▶ Test metrics: {test_results}")
new_best = study.best_trial.value


# Log final validation and test metrics
final_run.log({
    "final_val_loss": best_val_loss,
    **{f"test_{k}": v for d in test_results for k, v in d.items()},
})

# Optional: Log model checkpoint as artifact
final_artifact = wandb.Artifact(
    name="cmpnn_final_model",
    type="model",
    metadata={**best_hparams, "val_loss": best_val_loss},
)
final_artifact.add_file(best_ckpt)
final_run.log_artifact(final_artifact)

# Close the run
final_run.finish()

print("Best loss after second run:", new_best)
if new_best == old_best:
    print("No new trial beat the previous best.")
else:
    print("A new trial improved the best loss!")
