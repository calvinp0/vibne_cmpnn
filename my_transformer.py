import torch
from torch_geometric.data import Data

import math
import torch

class PeriodicToSinCos:
    def __init__(self, cols, specs, radians=True,
                 ignore_value=None):
        self.cont_idx = [i for i,c in enumerate(cols)
                         if specs[c]=="continuous"]
        self.per_idx  = [i for i,c in enumerate(cols)
                         if specs[c]=="periodic"]
        # if your CSV is in degrees and you want radians:
        self.ignore_val = ignore_value
        self.rad = math.pi/180 if not radians else 1.0

    def __call__(self, data):
        data = data.clone()                  # shape [1, M]
        # flatten any shape to 1D
        y = data.y.view(-1)                     # shape [M]
        # continuous targets
        cont = y[self.cont_idx]                 # shape [#cont]
        # raw angles (possibly in degrees) → radians
        ang  = y[self.per_idx] * self.rad       # shape [#per]
        
        # compute sin & cos for *all* angles
        sinv = ang.sin()
        cosv = ang.cos()

        # override sin/cos for any “fluff” angles
        if self.ignore_val is not None:
            # mask of length P where original y==ignore_val
            mask = (y[self.per_idx] == self.ignore_val)
            sinv[mask] = self.ignore_val
            cosv[mask] = self.ignore_val

        # stack into P×2, then flatten to 2P
        sinc = torch.stack([sinv, cosv], dim=1)  # shape [P,2]
        flat = sinc.view(-1)                     # shape [2P]

        # concat continuous + periodic
        out  = torch.cat([cont, flat], dim=0)    # shape [M + 2P]
        data.y = out.unsqueeze(0)                # row vector for batching
        return data


class PeriodicToSinCosGaussianJitter:
    def __init__(
        self,
        cols: list[str],
        specs: dict[str, str],
        radians: bool = True,
        ignore_val: float | None = None,
        max_jitter_deg: float = 10.0
    ):
        """
        Gaussian jitter transform with linearly scheduled magnitude.
        Args:
            cols:            list of target column names
            specs:           dict mapping each col to "continuous" or "periodic"
            radians:         if False, raw angles are degrees → convert to radians
            ignore_val:      sentinel (e.g. -10) to leave unmodified
            max_jitter_deg:  maximum jitter at final epoch (degrees)
        """
        # index separation
        self.cont_idx = [i for i,c in enumerate(cols) if specs[c]=="continuous"]
        self.per_idx  = [i for i,c in enumerate(cols) if specs[c]=="periodic"]
        # convert raw→radians if needed
        self.rad = (math.pi/180) if not radians else 1.0
        self.ignore_val = ignore_val
        # schedule parameters
        self.max_jitter_rad = max_jitter_deg * math.pi/180
        self.jitter_rad     = 0.0      # will be updated each epoch

    def __call__(self, data: Data) -> Data:
        data = data.clone()  # shape [1, M]
        y = data.y.view(-1)
        cont    = y[self.cont_idx]
        raw_ang = y[self.per_idx]
        ang     = raw_ang * self.rad

        # Gaussian noise with std=self.jitter_rad, only on valid angles
        if self.ignore_val is not None:
            valid = raw_ang != self.ignore_val
            noise = torch.randn_like(ang) * self.jitter_rad
            ang[valid] = ang[valid] + noise[valid]
        else:
            ang = ang + torch.randn_like(ang) * self.jitter_rad

        # sin/cos
        sinv = ang.sin()
        cosv = ang.cos()
        # restore ignore_val
        if self.ignore_val is not None:
            mask = raw_ang == self.ignore_val
            sinv[mask] = self.ignore_val
            cosv[mask] = self.ignore_val

        flat = torch.stack([sinv, cosv], dim=1).view(-1)
        out  = torch.cat([cont, flat], dim=0)
        data.y = out.unsqueeze(0)
        return data

import math
import torch
from torch_geometric.data import Data

class PeriodicAndNormalize:
    """
    data.y is assumed to be shape [T] in the same order as `col_names`.
     - first we apply (y - μ) / σ to every 'continuous' column;
     - then we replace every 'periodic' column θ  →  (sin θ, cos θ) pair.
    """

    def __init__(
        self,
        col_names: list[str],
        col_types: dict[str,str],
        assume_degrees: bool = False
    ):
        self.col_names      = col_names
        self.col_types      = col_types
        self.assume_degrees = assume_degrees

        # build index lists once
        self.cont_idx = [i for i,name in enumerate(col_names)
                         if col_types[name] == "continuous"]
        self.per_idx  = [i for i,name in enumerate(col_names)
                         if col_types[name] == "periodic"]

    def __call__(self, data: Data) -> Data:
        y = data.y.clone()           # shape [T]
        # --- normalize continuous dims ---
        if hasattr(data, "mean") and hasattr(data, "std"):
            # your dataset should attach mean/std per-dimension onto each Data…
            mean = data.mean[self.cont_idx]
            std = data.std[self.cont_idx]
        else:
            # fallback: normalize to unit – but really you want mean/std from train‐split
            mean = torch.zeros(len(self.cont_idx))
            std = torch.ones (len(self.cont_idx))

        y[self.cont_idx] = (y[self.cont_idx] - mean) / std

        # --- convert periodic dims to sin/cos pairs ---
        θ = y[self.per_idx]
        if self.assume_degrees:
            θ = θ * (math.pi / 180)
        sc = torch.stack([θ.sin(), θ.cos()], dim=1).view(-1)  # (2 * len(per_idx),)

        # now reassemble y′ = [ all continous dims in order ] + [ all sin/cos pairs ]
        cont = y[self.cont_idx]
        data.y = torch.cat([cont, sc], dim=0)

        return data


from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor, nn


class _ScaleTransformMixin(nn.Module):
    def __init__(self, mean: ArrayLike, scale: ArrayLike, pad: int = 0):
        super().__init__()

        mean = torch.cat([torch.zeros(pad), torch.tensor(mean, dtype=torch.float)])
        scale = torch.cat([torch.ones(pad), torch.tensor(scale, dtype=torch.float)])

        if mean.shape != scale.shape:
            raise ValueError(
                f"uneven shapes for 'mean' and 'scale'! got: mean={mean.shape}, scale={scale.shape}"
            )

        self.register_buffer("mean", mean.unsqueeze(0))
        self.register_buffer("scale", scale.unsqueeze(0))

    @classmethod
    def from_standard_scaler(cls, scaler: StandardScaler, pad: int = 0):
        return cls(scaler.mean_, scaler.scale_, pad=pad)

    def to_standard_scaler(self, anti_pad: int = 0) -> StandardScaler:
        scaler = StandardScaler()
        scaler.mean_ = self.mean[anti_pad:].numpy()
        scaler.scale_ = self.scale[anti_pad:].numpy()
        return scaler


class ScaleTransform(_ScaleTransformMixin):
    def forward(self, X: Tensor) -> Tensor:
        if self.training:
            return X

        return (X - self.mean) / self.scale


class UnscaleTransform(_ScaleTransformMixin):
    def forward(self, X: Tensor) -> Tensor:
        if self.training:
            return X

        return X * self.scale + self.mean

    def transform_variance(self, var: Tensor) -> Tensor:
        if self.training:
            return var
        return var * (self.scale**2)


