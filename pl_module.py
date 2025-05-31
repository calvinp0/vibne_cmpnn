import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import CMPNNEncoder, MLP, PeriodicHead, RawPeriodicHead, RegressionHead
from utils import rmse_torch, mae_torch, r2_score_torch
import math
import warnings
from metrics import MetricRegistry, LossRegistry, CMPNNMetric
from torchmetrics import Metric
from typing import Dict, List
import torch.nn.functional as F
from torch.utils.data import Subset

class CMPNNLitModel(pl.LightningModule):
    """
    Lightning wrapper supporting multiple metrics; first metric is used for loss.
    """
    def __init__(self,
                 in_node_feats: int,
                 in_edge_feats: int,
                 target_mean: float = 0.0,
                 target_std: float = 1.0,
                 hidden_dim: int = 128,
                 num_steps: int = 5,
                 dropout_mp: float = 0.05,
                 dropout_head: float = 0.1,
                 n_tasks: int = 1,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 scheduler_patience: int = 10,
                 scheduler_factor: float = 0.1,
                 readout: str = 'gru',
                 use_booster: bool = True,
                 metrics: list = None):
        super().__init__()
        self.save_hyperparameters(ignore=['metrics'])
        # encoder and head
        self.encoder = CMPNNEncoder(
            in_node_feats=in_node_feats,
            in_edge_feats=in_edge_feats,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            dropout=dropout_mp,
            n_tasks=n_tasks,
            readout=readout,
            use_booster=use_booster,
        )
        self.head = FFNHead(
            in_dim=(2 * hidden_dim if readout=='gru' else hidden_dim),
            hidden_dim=hidden_dim // 2,
            out_dim=n_tasks,
            dropout=dropout_head,
        )
        # metrics: first is loss, rest are additional metrics
        import torch.nn as nn
        if metrics is None:
            metrics = [nn.MSELoss()]
        self.metrics = nn.ModuleList(m.to(self.device) for m in metrics)
        self.criterion = self.metrics[0]
        # store target normalization for unscaled metrics
        self.target_mean = target_mean
        self.target_std = target_std

    def forward(self, batch):
        # unpack once
        x, edge_index, edge_attr, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        z = self.encoder.embed(x, edge_index, edge_attr, b)
        return self.head(z).view(-1)

    def training_step(self, batch, batch_idx):
        y_pred = self.forward(batch)
        y_true = batch.y.view(-1)
        # compute and log loss (first metric)
        loss = self.criterion(y_pred, y_true)
        self.log('train_loss', loss, prog_bar=True, batch_size=batch.num_graphs)
        # compute and log other metrics on training set
        for metric in self.metrics[1:]:
            name = metric.__class__.__name__.lower()
            val = metric(y_pred, y_true)
            self.log(f'train_{name}', val, prog_bar=False, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred = self.forward(batch)
        y_true = batch.y.view(-1)
        # compute and log all metrics
        for i, metric in enumerate(self.metrics):
            name = metric.__class__.__name__.lower()
            val = metric(y_pred, y_true)
            if i == 0:
                self.log('val_loss', val, prog_bar=True, batch_size=batch.num_graphs)
            else:
                self.log(f'val_{name}', val, prog_bar=False, batch_size=batch.num_graphs)
        # log inverse-scaled metrics
        if self.target_std != 1.0 or self.target_mean != 0.0:
            y_pred_un = y_pred * self.target_std + self.target_mean
            y_true_un = y_true * self.target_std + self.target_mean
            self.log('val_rmse_unscaled', rmse_torch(y_pred_un, y_true_un), prog_bar=True, batch_size=batch.num_graphs)
            self.log('val_mae_unscaled', mae_torch(y_pred_un, y_true_un), prog_bar=False, batch_size=batch.num_graphs)
            self.log('val_r2_unscaled', r2_score_torch(y_pred_un, y_true_un), prog_bar=False, batch_size=batch.num_graphs)
        return None

    def test_step(self, batch, batch_idx):
        y_pred = self.forward(batch)
        y_true = batch.y.view(-1)
        # compute and log all metrics
        for i, metric in enumerate(self.metrics):
            name = metric.__class__.__name__.lower()
            val = metric(y_pred, y_true)
            if i == 0:
                self.log('test_loss', val, prog_bar=True, batch_size=batch.num_graphs)
            else:
                self.log(f'test_{name}', val, prog_bar=False, batch_size=batch.num_graphs)
        # log inverse-scaled metrics
        if self.target_std != 1.0 or self.target_mean != 0.0:
            y_pred_un = y_pred * self.target_std + self.target_mean
            y_true_un = y_true * self.target_std + self.target_mean
            self.log('test_rmse_unscaled', rmse_torch(y_pred_un, y_true_un), prog_bar=True, batch_size=batch.num_graphs)
            self.log('test_mae_unscaled', mae_torch(y_pred_un, y_true_un), prog_bar=False, batch_size=batch.num_graphs)
            self.log('test_r2_unscaled', r2_score_torch(y_pred_un, y_true_un), prog_bar=False, batch_size=batch.num_graphs)
        return None

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # get the (normalized) output
        y_pred_norm = self(batch)
        # invert the scaling
        y_pred = y_pred_norm * self.target_std + self.target_mean
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.head.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # scheduler monitors first metric on validation
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.hparams.scheduler_factor,
                patience=self.hparams.scheduler_patience,
            ),
            'monitor': f'val_{self.metrics[0].__class__.__name__.lower()}',
            'interval': 'epoch',
            'frequency': 1,
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class MultiCMPNNLitModel(pl.LightningModule):
    """
    A lightning module that allows for multiple molecules per sample to be passed in.
    """
    def __init__(self,in_node_feats: int,
                 in_edge_feats: int,
                 in_global_feats: int = 0,
                 target_mean: float = 0.0,
                 target_std: float = 1.0,
                 hidden_dim: int = 128,
                 num_steps: int = 5,
                 dropout_mp: float = 0.05,
                 dropout_head: float = 0.1,
                 n_tasks: int = 1,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 scheduler_patience: int = 10,
                 scheduler_factor: float = 0.1,
                 readout: str = 'gru',
                 use_booster: bool = True,
                 metrics: list = None,
                 continuous_metrics: list = None,
                 periodic_metrics: list = None,
                 mpn_shared: bool = False,
                 task_weights: list = None,
                 target_types: Dict[str, str] = None,
                 ignore_val: float = None,
                 jitter: bool = False,
                 normalize_head: bool = False,
                 X_d_transform= None,
                 mean_dir = None,
                 pinball_weight: float = None,
                 head_layers: int = 2,
                 use_residual: bool = True,
                 use_graph_residual: bool = False,
                 **kwargs) -> None:
        super().__init__()
        # store hyperparams for checkpointing (skip large objects)
        filtered_hparams = {
    "in_node_feats": in_node_feats,
    "in_edge_feats": in_edge_feats,
    "target_mean": target_mean if isinstance(target_mean, float) or len(target_mean) > 0 else 0.0,
    "target_std":  target_std  if isinstance(target_std, float) or len(target_std) > 0 else 1.0,
    "hidden_dim": hidden_dim,
    "num_steps": num_steps,
    "dropout_mp": dropout_mp,
    "dropout_head": dropout_head,
    "n_tasks": n_tasks,
    "lr": lr,
    "weight_decay": weight_decay,
    "scheduler_patience": scheduler_patience,
    "scheduler_factor": scheduler_factor,
    "readout": readout,
    "use_booster": use_booster,
    "metrics": metrics,
    "continuous_metrics": continuous_metrics,
    "periodic_metrics": periodic_metrics,
    "mpn_shared": mpn_shared,
    "task_weights": task_weights,
    "target_types": target_types,
    "ignore_val": ignore_val,
    "jitter": jitter,
    "normalize_head": normalize_head,
    "pinball_weight": pinball_weight,
    "head_layers": head_layers,
    "use_residual": use_residual,
    "use_graph_residual": use_graph_residual,
    "kwargs": kwargs,
}
        self.save_hyperparameters(filtered_hparams)
        filtered_hparams.pop("metrics", None)
        self.jitter = jitter
        self.in_global_feats = in_global_feats
        self.X_d_transform = X_d_transform
        self.mean_dir = mean_dir

        # ------------------------------------------------------------------ encoders
        if mpn_shared:
            self.encoder = CMPNNEncoder(
                in_node_feats, in_edge_feats,
                hidden_dim=hidden_dim, num_steps=num_steps,
                dropout=dropout_mp, n_tasks=n_tasks,
                readout=readout, use_booster=use_booster,
                use_graph_residual=use_graph_residual,
            )
            self.encoders = [self.encoder]            # iterable helper
        else:
            self.encoder1 = CMPNNEncoder(
                in_node_feats, in_edge_feats,
                hidden_dim=hidden_dim, num_steps=num_steps,
                dropout=dropout_mp, n_tasks=n_tasks,
                readout=readout, use_booster=use_booster,
                use_graph_residual=use_graph_residual,
            )
            self.encoder2 = CMPNNEncoder(
                in_node_feats, in_edge_feats,
                hidden_dim=hidden_dim, num_steps=num_steps,
                dropout=dropout_mp, n_tasks=n_tasks,
                readout=readout, use_booster=use_booster,
                use_graph_residual=use_graph_residual,
            )
            self.encoders = [self.encoder1, self.encoder2]

        # # split indices
        # names = list(target_types.keys())
        # self.cont_idx = [i for i,n in enumerate(names) if target_types[n]=='continuous']
        # self.per_idx  = [i for i,n in enumerate(names) if target_types[n]=='periodic']

        # two heads
        embed_dim = hidden_dim * (2 if readout=='gru' else 1)
        head_in = 2 * embed_dim + 2 * in_global_feats
        head_hidden = head_in // 2

        self.head = RegressionHead(
            input_dim=head_in,
            hidden_dim=head_hidden,
            output_dim=n_tasks,
            dropout=dropout_head,
            n_layers=head_layers,
            use_residual=use_residual,
        )

        self.pinball_weight = pinball_weight if pinball_weight is not None else 0.0

        # self.head = RawPeriodicHead(
        #     input_dim=head_in,
        #     hidden_dim=head_hidden,
        # )
        # self.head = PeriodicHead(
        #     input_dim=head_in,
        #     hidden_dim=head_hidden,
        #     output_dim=2,
        #     dropout=dropout_head,
        #     n_layers=2,
        #     use_residual=True,
        #     normalize=normalize_head,
        # )

        # if self.cont_idx:
        #     self.head_cont = MLP(input_dim=head_in,
        #                              hidden_dim=head_hidden,
        #                              output_dim=len(self.cont_idx),
        #                              dropout=dropout_head)
        # else:
        #     self.head_cont = None

        # if self.per_idx:
        #     self.head_per = PeriodicHead(input_dim=head_in,
        #                             hidden_dim=head_hidden,
        #                             output_dim=2*len(self.per_idx),
        #                             dropout=dropout_head)
        # else:
        #     self.head_per = None


        # Metrics
        self.metrics = torch.nn.ModuleDict()
        for idx, item in enumerate(metrics):
            if isinstance(item, CMPNNMetric):
                m, alias = item, item.alias
            elif isinstance(item, Metric):   # catch any torchmetrics.Metric
                m     = item
                alias = getattr(m, "alias", m.__class__.__name__.lower())
            elif isinstance(item, str):
                m_cls = MetricRegistry[item]
                m     = m_cls(task_weights=task_weights)
                alias = item
            else:
                raise TypeError(f"metrics must be CMPNNMetric, Metric, or str; got {type(item)}")
            self.metrics[alias] = m
            if idx == 0:
                self.criterion       = m
                self.criterion_alias = alias
            

        # # default metrics
        # continuous_metrics = continuous_metrics or ['mse']
        # periodic_metrics   = periodic_metrics   or ['rmse']
        # cont_weights = task_weights or [1.0] * len(self.cont_idx)
        # per_weights  = [1.0] * (2 * len(self.per_idx)) if self.per_idx else []

        # # instantiate losses & metrics
        # self.metrics_cont = torch.nn.ModuleDict()
        # # handle each metric spec (string alias or instance)
        # for m in continuous_metrics:
        #     key = m if isinstance(m, str) else m.alias
        #     cls = MetricRegistry[key]
        #     self.metrics_cont[key] = cls(task_weights=cont_weights, ignore_value=ignore_val)
        
        # if self.cont_idx:
        #     first_c = continuous_metrics[0]
        #     key_c = first_c if isinstance(first_c, str) else first_c.alias
        #     self.loss_cont = LossRegistry[key_c](task_weights=cont_weights, ignore_value=ignore_val)
        # else:
        #     self.loss_cont = None

        # self.metrics_per = torch.nn.ModuleDict()
        # for m in periodic_metrics:
        #     key = m if isinstance(m, str) else m.alias
        #     cls = MetricRegistry[key]
        #     self.metrics_per[key] = cls(task_weights=per_weights, ignore_value=ignore_val)

        # if self.per_idx:
        #     first_p = periodic_metrics[0]
        #     key_p = first_p if isinstance(first_p, str) else first_p.alias
        #     self.loss_per = LossRegistry[key_p](task_weights=per_weights, ignore_value=ignore_val)
        # else:
        #     self.loss_per = None

        # store task weights for unscaled metrics
        self.task_weights = task_weights

        for metric in self.metrics.values():
            if isinstance(metric, CMPNNMetric):
                metric._init_states(n_tasks)      # eager initialisation
        # ------------------------------------------------------------------


        # optimizer params
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor

        # normalisation (for un-scaled reporting)
        self.register_buffer("t_mean", torch.as_tensor(target_mean))
        self.register_buffer("t_std",  torch.as_tensor(target_std))
        torch.autograd.set_detect_anomaly(True)
    
    def forward(self, big_batch):
        """
        big_batch : PyG Batch with 2 × N graphs
        pair_idx  : LongTensor [2N] holding 0,0,1,1,2,2,…

        returns   : Tensor [N]  (prediction per pair)
        """
        z_all = self._encode_big_batch(big_batch)          # (2N, embed)

        # even positions --> first molecule, odd positions --> second molecule
        z1 = z_all[0::2]          # shape (N, embed)
        z2 = z_all[1::2]          # shape (N, embed)

        z_cat = torch.cat([z1, z2], dim=1)

        if hasattr(big_batch, "global_features"):
            g1 = big_batch.global_features[0::2]
            g2 = big_batch.global_features[1::2]
            # print("g1 shape:", g1.shape)
            # print("g2 shape:", g2.shape)
            g_cat = torch.cat([g1, g2], dim=1)
            
            if self.X_d_transform is not None:
                g_cat = self.X_d_transform(g_cat)

            z_cat = torch.cat([z_cat, g_cat], dim=1)

        # out
        return self.head(z_cat)                   # (N, 2×out_dim)


    def _encode_big_batch(self, batch) -> torch.Tensor:
        # shared encoder – unchanged
        if hasattr(self, "encoder"):
            return self.encoder.embed(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )

        # ---------- two encoders ----------
        n_graphs_tot = batch.num_graphs            # 2 · N
        n_half       = n_graphs_tot // 2           # N
        node_off     = batch.ptr[n_half].item()    # first node of 2nd half

        # masks on NODES for slicing tensors
        nmask1 = batch.batch <  n_half
        nmask2 = batch.batch >= n_half
        emask1 = nmask1[batch.edge_index[0]]       # src-node mask
        emask2 = ~emask1

        # run both encoders
        z1 = self.encoder1.embed(
            batch.x[nmask1],
            batch.edge_index[:, emask1],           # indices already start at 0
            batch.edge_attr[emask1],
            batch.batch[nmask1],                   # 0 … N-1
        )
        z2 = self.encoder2.embed(
            batch.x[nmask2],
            batch.edge_index[:, emask2] - node_off,    # re-index nodes
            batch.edge_attr[emask2],
            batch.batch[nmask2] - n_half,              # 0 … N-1 again
        )

        # put them back in graph order (first N graphs, then next N graphs)
        z_all             = torch.empty(
            (n_graphs_tot, z1.size(1)), dtype=z1.dtype, device=z1.device
        )
        z_all[:n_half]    = z1
        z_all[n_half:]    = z2

        if torch.isnan(z_all).any():
            print("NaNs in z_all!")
            print("z1 max", z1.abs().max().item(), "min", z1.min().item())
            print("z2 max", z2.abs().max().item(), "min", z2.min().item())
            print("x[nmask1] NaNs", torch.isnan(batch.x[nmask1]).any())
            print("x[nmask2] NaNs", torch.isnan(batch.x[nmask2]).any())
        return z_all
    
    def step(self, batch, stage: str):
        # ── safety checks ─────────────────────────────────────────────
        assert not torch.isnan(batch.x).any() and not torch.isinf(batch.x).any()
        assert not torch.isnan(batch.y).any() and not torch.isinf(batch.y).any()

        # ── forward ───────────────────────────────────────────────────
        y_pred  = self(batch)                       # (N, 3)
        y_flat  = batch.y.view(batch.num_graphs, -1)
        N       = batch.num_graphs // 2
        assert torch.allclose(y_flat[0::2], y_flat[1::2])
        y_true  = y_flat[0::2]                      # one row per pair

        # ── primary loss (first metric) ───────────────────────────────
        loss_mean = self.criterion(y_pred, y_true)  # e.g. MSE

        # ── optional pinball loss blend ───────────────────────────────
        pinball_alias = next(
                        (a for a in self.metrics if a.startswith("pinball_q")),
                        None
                    )
        loss_total    = loss_mean                   # default
        loss_pinball  = None

        if pinball_alias in self.metrics:
            loss_pinball  = self.metrics[pinball_alias](y_pred, y_true)
            loss_total    = ((1 - self.pinball_weight) * loss_mean +
                            self.pinball_weight      * loss_pinball)

        # ── logging losses ────────────────────────────────────────────
        self.log(f"{stage}_loss",      loss_total,  batch_size=batch.num_graphs,
                prog_bar=True,  on_step=False, on_epoch=True)
        self.log(f"{stage}_loss_mean", loss_mean,   batch_size=batch.num_graphs,
                prog_bar=False, on_step=False, on_epoch=True)
        if loss_pinball is not None:
            self.log(f"{stage}_loss_pinball", loss_pinball,
                    batch_size=batch.num_graphs, prog_bar=False,
                    on_step=False, on_epoch=True)

        # ── extra metrics ─────────────────────────────────────────────
        for alias, metric in self.metrics.items():
            if alias in (self.criterion_alias, pinball_alias):
                continue                                # already handled
            val = metric(y_pred, y_true)                # updates + batch value
            self.log(f"{stage}_{alias}", val, batch_size=batch.num_graphs,
                    prog_bar=False, on_step=False, on_epoch=True)

        return loss_total

    def training_step  (self, batch, _): return self.step(batch, "train")
    def validation_step(self, batch, _): return self.step(batch, "val")
    def test_step      (self, batch, _): return self.step(batch, "test")


    def on_fit_start(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_start(self):
        if not self.jitter:
            return

        ds    = self.trainer.datamodule.train_dataset   # SubsetWithTransform
        tr    = ds.transform                            # your jittering transform
        epoch = self.current_epoch
        total = max(self.trainer.max_epochs - 1, 1)
        frac  = epoch / total

        tr.jitter_rad = tr.max_jitter_rad * frac
        self.log("jitter_deg", tr.jitter_rad * 180/math.pi, prog_bar=True)
    
    # def predict_step(self, batch, batch_idx):
    #     y_pred = self(batch)  # shape (N, 2)
    #     sin_vals = y_pred[:, 0]
    #     cos_vals = y_pred[:, 1]

    #     angles = torch.atan2(sin_vals, cos_vals)  # radians

    #     # Ensure per_mean is a tensor on the same device
    #     per_mean = self.mean_dir.to(angles.device) if hasattr(self, 'mean_dir') else torch.tensor(0.0, device=angles.device)

    #     angles = (angles + per_mean) % (2 * math.pi)
    #     angles = angles * 180 / math.pi  # convert to degrees

    #     return angles

    def predict_step(self, batch, batch_idx):
        """
        Return network output in the same scaled space that was used for training.
        """
        return self(batch)          # shape (N, 3)

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.cpu().item())

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return 
        loss = self.trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_losses.append(loss.cpu().item())    

    def on_train_end(self):
        print(">> on_fit_end triggered")
        import matplotlib.pyplot as plt
        import os
        if len(self.train_losses) == 0 or len(self.val_losses) == 0:
            print("No training or validation losses to plot.")
            return
        print(">> on_fit_end triggered")

        
        min_len = min(len(self.train_losses), len(self.val_losses))
        train_losses = self.train_losses[:min_len]
        val_losses   = self.val_losses[:min_len]
        epochs       = range(1, min_len + 1)

        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")

        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train vs Val Loss")
        plt.legend()

        log_dir = self.trainer.logger.log_dir if self.trainer.logger else "."
        img_dir = os.path.join(log_dir, "images")
        print(f"Saving loss curve to {img_dir}")
        os.makedirs(img_dir, exist_ok=True)
        plt.savefig(os.path.join(img_dir, "loss_curve.png"))
        plt.close()

    @torch.no_grad()
    def inverse_transform(self, y_scaled: torch.Tensor) -> torch.Tensor:
        """
        Convert scaled predictions back to original units.

        Usage
        -----
        y_hat   = trainer.predict(model, dataloaders)[0]   # still scaled
        y_unsc  = model.inverse_transform(y_hat)
        """
        return y_scaled * self.t_std + self.t_mean

    def configure_optimizers(self):
        params = list(self.head.parameters())
        for enc in self.encoders:
            params += list(enc.parameters())

        opt = torch.optim.Adam(params, lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)
        sch = {
            "scheduler": ReduceLROnPlateau(
                opt, mode="min",
                factor=self.hparams.scheduler_factor,
                patience=self.hparams.scheduler_patience,
            ),
            "monitor": f"val_loss",
        }
        return {"optimizer": opt, "lr_scheduler": sch}
        # return opt