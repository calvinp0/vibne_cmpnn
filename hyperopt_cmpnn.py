# ─────────────────────── install / import ─────────────────────────
# (do this once per env)
# !pip install optuna wandb pytorch-lightning[extra] optuna-dashboard

import os, torch, optuna, pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data        import DataLoader, Subset
from dataset import CMPNNDataset, MultiCMPNNDataset, MultiCMPNNDatasetSDF, CMPNNDataModule
from preprocessing import scaffold_split_indices, random_split_indices
from pl_module import MultiCMPNNLitModel
from utils_paired import collate_pairs
from metrics import MSE, MAE, RMSE, R2Score, PinballLoss

pl.seed_everything(42)  # for reproducibility

# ─────── WandB OFFLINE so nothing is pushed to the cloud ──────────
os.environ["WANDB_MODE"]   = "offline"
os.environ["WANDB__SERVICE_WAIT"] = "300"   # keeps the daemon quiet

import wandb, pathlib

def log_final_model(checkpoint_path: str, rank: int, val_loss: float):
    run = wandb.init(
        project = "cmpnn-hyperopt",
        name    = f"final_rank{rank}",
        mode    = "offline",
        config  = {"rank": rank, "val_loss": val_loss},
    )

    artifact = wandb.Artifact(
        name = f"cmpnn_final_rank{rank}",
        type = "model",
        metadata = {"val_loss": val_loss},
    )
    artifact.add_file(checkpoint_path)
    run.log_artifact(artifact)
    run.finish()


SDF_DIR = "/home/calvin.p/Code/dmpnn_customized/DATA/sdf_data"
TARGETS = "/home/calvin.p/Code/Data/target_data/temp_kinetics_for.csv"
TARGETS_COLS  = ['A_log10','n', 'Ea_yj']
TARGETS_TYPES = {'A_log10': 'continuous', 'n': 'continuous', 'Ea_yj': 'continuous'}
EXTRA_FEATS = "/home/calvin.p/Code/dmpnn_customized/DATA/sdf_data/all_sdf_features.csv"

def make_dataset(atom_messages: bool) -> MultiCMPNNDatasetSDF:
    return MultiCMPNNDatasetSDF(
        root           = ".",
        sdf_files      = SDF_DIR,
        target_df      = TARGETS,
        target_cols    = TARGETS_COLS,
        target_types   = TARGETS_TYPES,
        atom_messages  = atom_messages,
        keep_hs        = True,
        sanitize       = True,
        force_reload   = False,       # cached by auto hash
        prune_value    = -10,
        atom_extra_feats = EXTRA_FEATS,
        rbf_num_centers=16,
    )


# ────────────────── optuna objective function ─────────────────────
def objective(trial: optuna.Trial) -> float:

    import wandb
    if wandb.run is not None:
        wandb.finish()

    # -------- sample hyper-parameters -------------------------------------
    hparams = dict(
        hidden_dim       = trial.suggest_int ("hidden_dim",        256, 1024, step=256),
        head_layers      = trial.suggest_int ("head_layers",       1,   3),
        dropout_mp       = trial.suggest_float("dropout_mp",       0.0, 0.4, step=0.05),
        dropout_head     = trial.suggest_float("dropout_head",     0.0, 0.4, step=0.05),
        lr               = trial.suggest_loguniform("lr",          1e-5, 3e-3),
        weight_decay     = trial.suggest_loguniform("weight_decay",1e-6, 3e-4),
        mpn_shared       = trial.suggest_categorical("mpn_shared",       [True, False]),
        readout          = trial.suggest_categorical("readout",          ["sum", "gru"]),
        atom_messages    = trial.suggest_categorical("atom_messages",    [True, False]),
        use_booster      = trial.suggest_categorical("use_booster",      [True, False]),
        use_residual     = trial.suggest_categorical("use_residual",     [True, False]),
        use_graph_residual = trial.suggest_categorical("use_graph_residual",[True, False]),
        pinball_q        = trial.suggest_float("pinball_q",        0.80, 0.95),
        pinball_weight   = trial.suggest_float("pinball_weight",   0.10, 0.50),
        num_steps        = trial.suggest_int ("num_steps",         3, 6),
    )

    ds_trial = make_dataset(hparams["atom_messages"])
    train_idx, val_idx, _ = random_split_indices(ds_trial, 0.1, 0.1, seed=42)
    ds_trial.attach_atom_extra_features(train_idx)
    ds_trial.compute_normalization(train_idx)
    ds_trial.apply_normalization()

    train_loader = DataLoader(Subset(ds_trial, train_idx), batch_size=64, shuffle=True, collate_fn=collate_pairs, pin_memory=True, num_workers=16)
    val_loader   = DataLoader(Subset(ds_trial, val_idx),   batch_size=32, shuffle=False, collate_fn=collate_pairs, pin_memory=True, num_workers=16)


    # -------- metrics list (weights fixed here) ---------------------------
    metrics = [
        MSE(task_weights=[10,1,1]),
        PinballLoss(q=hparams["pinball_q"], task_weights=[10,1,1]),
        RMSE([10,1,1]), MAE([10,1,1]), R2Score([10,1,1]),
    ]

    # -------- build model --------------------------------------------------
    model = MultiCMPNNLitModel(
        in_node_feats      = ds_trial.num_atom_features,
        in_edge_feats      = ds_trial.num_bond_features,
        hidden_dim         = hparams["hidden_dim"],
        num_steps          = hparams["num_steps"],
        dropout_mp         = hparams["dropout_mp"],
        dropout_head       = hparams["dropout_head"],
        head_layers        = hparams["head_layers"],
        n_tasks            = 3,
        metrics            = metrics,
        lr                 = hparams["lr"],
        weight_decay       = hparams["weight_decay"],
        mpn_shared         = hparams["mpn_shared"],
        readout            = hparams["readout"],
        use_booster        = hparams["use_booster"],
        use_residual       = hparams["use_residual"],
        use_graph_residual = hparams["use_graph_residual"],
        target_mean        = ds_trial.mean,
        target_std         = ds_trial.std,
        pinball_weight     = hparams["pinball_weight"],
    )

    # -------- offline WandB logger per trial ------------------------------
    wandb_logger = WandbLogger(
        project = "cmpnn-hyperopt",
        name    = f"trial_{trial.number}",
        reinit  = True,
        config  = hparams,        # every sampled value
    )

    # -------- trainer ------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs      = 60,            # short fidelity for speed
        accelerator     = "gpu",
        devices         = 1,
        logger          = wandb_logger,
        enable_progress_bar = False,
        callbacks = [
            EarlyStopping("val_loss", patience=8, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min"),
        ],
        log_every_n_steps=10,     # now you’ll get logs after every 10 batches
    )

    trainer.fit(model, train_loader, val_loader)

    val_loss = trainer.callback_metrics["val_loss"].item()
    wandb_logger.log_metrics({"val_loss": val_loss})
    wandb_logger.finalize(status="finished")

    # report to Optuna (lower is better)
    return val_loss

# ────────────────── run / resume the study ────────────────────────
storage = "sqlite:///cmpnn_optuna.db"
study   = optuna.create_study(
    study_name = "cmpnn_full",
    direction  = "minimize",
    storage    = storage,
    load_if_exists=True,
    sampler    = optuna.samplers.TPESampler(multivariate=True, seed=42),
    pruner     = optuna.pruners.MedianPruner(n_startup_trials=10),
)
SWEEP_EPOCHS = 50          # fast search
study.optimize(objective, n_trials=50, timeout=6*3600)

# ─────────────────────── top‐k retrain ─────────────────────────────
TOPK       = 3
top_trials = sorted(study.trials, key=lambda t: t.value)[:TOPK]

# Print out the top‐3 trials right here:
print("\n=== Top‐3 Optuna Trials ===")
for rank, tr in enumerate(top_trials, 1):
    print(f"Rank {rank}: trial #{tr.number} → val_loss = {tr.value:.4f}")
    print("  params:", tr.params)
print("============================\n")


FULL_EPOCHS   = 300
final_models  = []

for rank, tr in enumerate(top_trials, 1):
    hp = tr.params

    # Rebuild & normalize dataset for final run
    ds_best               = make_dataset(hp["atom_messages"])
    train_i, val_i, test_i = random_split_indices(ds_best, 0.1, 0.1, seed=42)
    ds_best.attach_atom_extra_features(train_i)
    ds_best.compute_normalization(train_i)
    ds_best.apply_normalization()

    train_loader = DataLoader(
        Subset(ds_best, train_i),
        batch_size=64,
        shuffle=True,
        collate_fn=collate_pairs,
        pin_memory=True,
        num_workers=16,
    )
    val_loader = DataLoader(
        Subset(ds_best, val_i),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_pairs,
        pin_memory=True,
        num_workers=16,
    )
    test_loader = DataLoader(
        Subset(ds_best, test_i),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_pairs,
        pin_memory=True,
        num_workers=16,
    )

    # Metrics list identical to the sweep
    metrics = [
        MSE(task_weights=[10, 1, 1]),
        PinballLoss(q=hp["pinball_q"], task_weights=[10, 1, 1]),
        RMSE([10, 1, 1]),
        MAE([10, 1, 1]),
        R2Score([10, 1, 1]),
    ]

    # Build the final model
    model = MultiCMPNNLitModel(
        in_node_feats      = ds_best.num_atom_features,
        in_edge_feats      = ds_best.num_bond_features,
        hidden_dim         = hp["hidden_dim"],
        num_steps          = hp["num_steps"],
        dropout_mp         = hp["dropout_mp"],
        dropout_head       = hp["dropout_head"],
        head_layers        = hp["head_layers"],
        n_tasks            = 3,
        metrics            = metrics,
        lr                 = hp["lr"],
        weight_decay       = hp["weight_decay"],
        mpn_shared         = hp["mpn_shared"],
        readout            = hp["readout"],
        atom_messages      = hp["atom_messages"],
        use_booster        = hp["use_booster"],
        use_residual       = hp["use_residual"],
        use_graph_residual = hp["use_graph_residual"],
        target_mean        = ds_best.mean,
        target_std         = ds_best.std,
        pinball_weight     = hp["pinball_weight"],
    )

    # Train for 300 epochs
    trainer = pl.Trainer(
        max_epochs  = FULL_EPOCHS,
        accelerator = "gpu",
        devices     = 1,
        logger      = None,  # we’ll log the final checkpoint separately
        callbacks   = [
            EarlyStopping("val_loss", patience=25, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min"),
        ],
    )
    trainer.fit(model, train_loader, val_loader)

    final_val_loss = trainer.callback_metrics["val_loss"].item()
    ckpt_path      = trainer.checkpoint_callback.best_model_path
    final_models.append((rank, final_val_loss, ckpt_path))

    # Log this final checkpoint as an offline W&B Artifact
    log_final_model(ckpt_path, rank, final_val_loss)

# ────────────────────── Test the best final model ─────────────────
best_rank, best_loss, best_ckpt = final_models[0]
print(f"Testing best final model (Rank 1, val_loss={best_loss:.4f})")

# Create a fresh trainer for testing
test_trainer = pl.Trainer(accelerator="gpu", devices=1, logger=None)

# Load the model from checkpoint and run test
best_model = MultiCMPNNLitModel.load_from_checkpoint(best_ckpt, metrics=metrics)
test_results = test_trainer.test(best_model, test_loader)
print(f"Test metrics for Rank 1: {test_results}")

# ────────────────────── Print summary ─────────────────────────────
print("=== Summary ===")
print("Optuna best trial number:   ", study.best_trial.number)
print("Optuna best trial val_loss: ", study.best_trial.value)
print("Best hyperparameters:       ", study.best_trial.params)
for rank, val_loss, ckpt in final_models:
    print(f"Final Rank {rank}: val_loss={val_loss:.4f}, checkpoint={ckpt}")