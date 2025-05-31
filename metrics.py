from __future__ import annotations
import math
from typing import Sequence, Dict, Callable, Type, Any, Iterable, Union, TypeVar
from numpy.typing import ArrayLike
import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics import Metric
import torchmetrics
T = TypeVar("T")
# Registiry
class ClassRegistry(dict[str, Type[T]]):
    def register(self, alias: Any | Iterable[Any] | None = None):
        def decorator(cls):
            if alias is None:
                keys = [cls.__name__.lower()]
            elif isinstance(alias, str):
                keys = [alias]
            else:
                keys = alias

            cls.alias = keys[0]
            for k in keys:
                self[k] = cls

            return cls

        return decorator

    __call__ = register

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}: {super().__repr__()}"

    def __str__(self) -> str:  # pragma: no cover
        INDENT = 4
        items = [f"{' ' * INDENT}{repr(k)}: {repr(v)}" for k, v in self.items()]

        return "\n".join([f"{self.__class__.__name__} {'{'}", ",\n".join(items), "}"])



class CMPNNMetric(Metric):
    """
    Base-class that behaves like a torchmetrics.Metric *and* like a
    torch.nn.Module loss function (callable returns reduced scalar).

    • `task_weights`: 1-D tensor `[T]`. Default = ones  (equal weight).  
    • `alias`:        short name used in logs & registries.
    """

    is_differentiable = True
    higher_is_better  = False
    full_state_update = False

    def __init__(self, task_weights: Sequence[float] | Tensor | None = None, alias: str = "metric", ignore_value=None, **kwargs):
        """
        Args:
            num_tasks (int): Number of tasks.
            task_weights (Sequence[float] | Tensor | None, optional): Task weights. Defaults to None.
            alias (str, optional): Alias for the metric. Defaults to "metric".
        """
        super().__init__(**kwargs)
        self.alias = alias
        self.ignore_value = ignore_value

        self.register_buffer("task_weights", None, persistent=False)   # init later
        self.add_state("tot_loss", default=torch.zeros(0), dist_reduce_fx="sum")
        self.add_state("n_obs",    default=torch.tensor(0.), dist_reduce_fx="sum")

        if task_weights is not None:
            self._initial_task_weights = torch.as_tensor(task_weights, dtype=torch.float)
        else:
            self._initial_task_weights = None

    def _init_states(self, num_tasks: int):
        if self.tot_loss.numel() == 0:                        # first batch only
            self.tot_loss = torch.zeros(num_tasks, device=self.device)
            if self._initial_task_weights is None:
                self.task_weights = torch.ones(num_tasks, device=self.device)
            else:
                if len(self._initial_task_weights) != num_tasks:
                    raise ValueError("task_weights length does not match `num_tasks`")
                self.task_weights = self._initial_task_weights.to(self.device)

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        Calculate the unreduced loss. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def update(self, preds: Tensor, target: Tensor, weights: Tensor | None = None):

        """
        Update the metric state with new predictions and targets.

        Args:
            preds (Tensor): Predictions.
            target (Tensor): Targets.
            weights (Tensor | None, optional): Weights for the predictions. Defaults to None.
        """
        self._init_states(preds.size(1))
        if self.ignore_value is not None:
            mask = ~(target == self.ignore_value).any(dim=1)
            preds = preds[mask]
            target = target[mask]

            if preds.size(0) == 0:
                print("[DEBUG] Empty angular loss — skipping update")
                return
        losses = self._calc_unreduced_loss(preds, target)
        if weights is not None:
            weights.reshape_as(losses) if weights.ndim == 1 else weights
            losses = losses * weights

        losses = losses * self.task_weights
        self.tot_loss += losses.sum(dim=0)
        self.n_obs += losses.size(0)
        if losses.numel() == 0:
            print("[DEBUG] Empty angular loss — skipping update")

    def compute(self) -> Tensor:
        """
        Compute the final metric value.
        """
        return self.tot_loss / self.n_obs.clamp(min=1)

    def forward(self, preds: Tensor, targets: Tensor, **kw) -> Tensor:
        self.update(preds.detach(), targets.detach(), **kw)
        return self._calc_unreduced_loss(preds, targets).mean()

    def extra_repr(self) -> str:
        if self.task_weights is None:
            return f"alias={self.alias}, task_weights=NotInitialised"
        return f"alias={self.alias}, task_weights={self.task_weights.tolist()}"

MetricRegistry = ClassRegistry[CMPNNMetric]()
LossRegistry = ClassRegistry[CMPNNMetric]()

@MetricRegistry.register("mse")
@LossRegistry.register("mse")
class MSE(CMPNNMetric):
    def __init__(self, task_weights=None, **kw):
        super().__init__(task_weights=task_weights, alias="mse", **kw)

    def _calc_unreduced_loss(self, preds, targets):
        return F.mse_loss(preds, targets, reduction="none")

@MetricRegistry.register("rmse")
@LossRegistry.register("rmse")
class RMSE(CMPNNMetric):
    def __init__(self, task_weights=None, **kw):
        super().__init__(task_weights=task_weights, alias="rmse", **kw)
    def _calc_unreduced_loss(self, preds, targets):
        return F.mse_loss(preds, targets, reduction="none").sqrt()
    
@MetricRegistry.register("mae")
@LossRegistry.register("mae")
class MAE(CMPNNMetric):
    def __init__(self, task_weights=None, **kw):
        super().__init__(task_weights=task_weights, alias="mae", **kw)

    def _calc_unreduced_loss(self, preds, targets):
        return F.l1_loss(preds, targets, reduction="none")

@MetricRegistry.register("r2")
class R2Score(torchmetrics.R2Score):
    def __init__(self, task_weights: ArrayLike = 1.0, **kwargs):
        """
        Parameters
        ----------
        task_weights :  ArrayLike = 1.0
            .. important::
                Ignored. Maintained for compatibility with :class:`ChempropMetric`
        """
        super().__init__()
        task_weights = torch.as_tensor(task_weights, dtype=torch.float).view(1, -1)
        self.register_buffer("task_weights", task_weights)

    def update(self, preds: Tensor, targets: Tensor):
        super().update(preds, targets)
    


# Periodic Metrics & Losses

@MetricRegistry.register("angular_error")
@LossRegistry.register("angular_error")
class AngularError(CMPNNMetric):
    """
    Mean angular error (in radians) between predicted and target unit-vectors. Expects pred, target of shape (B, 2) if multiple angles
    Reduces to shape (B, T) of per-angle errors.
    """

    def __init__(self, task_weights=None, alias="angular_error", ignore_value=None, gamma_init=0.1, **kwargs):
        super().__init__(task_weights=task_weights, alias=alias, ignore_value=ignore_value, **kwargs)
        self.ignore_value = ignore_value
        self._gamma_unconstrained = torch.nn.Parameter(torch.tensor(float(gamma_init)).log().exp().log())


    def _init_states(self, num_tasks: int):
        """
        num_tasks here is preds.size(1) == 2*T.
        We want T = num_tasks // 2.
        """
        super()._init_states(num_tasks // 2)

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        preds, targets: (B, 2) sin/cos
        returns:       (B,) per-sample angular losses, with ignore_value masked out as zero
        """
        # Extract sin & cos
        sin_pred, cos_pred = preds[:, 0], preds[:, 1]
        sin_true, cos_true = targets[:, 0], targets[:, 1]

        # Normalize predictions (skip normalization of targets since they should already be unit)
        norm = torch.clamp(torch.sqrt(sin_pred**2 + cos_pred**2), min=1e-7)
        sin_pred, cos_pred = sin_pred/norm, cos_pred/norm

        # Build mask of valid entries
        if self.ignore_value is not None:
            valid = (targets != self.ignore_value).all(dim=1)  # shape (B,)
        else:
            valid = torch.ones(preds.size(0), dtype=torch.bool, device=preds.device)

        # Compute dot only for valid entries
        dot = sin_pred* sin_true + cos_pred*cos_true        # (B,)
        dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)

        # For invalid entries, force dot=1 so loss=0
        dot_valid = dot[valid]
        if dot_valid.numel() == 0:
            # no valid samples: return zero‐tensor so upstream skips them
            return torch.zeros_like(dot)

        # Final per-sample loss
        # loss = (1.0 - dot)                                 # shape (B,)
        loss_valid = torch.acos(dot_valid)
        loss = torch.zeros_like(dot)
        loss[valid] = loss_valid
        return loss


        # B, D = preds.shape
        # T = D // 2
        # # Reshape to (B, T, 2)
        # preds = preds.view(B, T, 2)
        # targets = targets.view(B, T, 2)
        # # Normalize to unit vectors
        # preds = F.normalize(preds, dim=-1, eps=1e-7)
        # targets = F.normalize(targets, dim=-1, eps=1e-7)

        # # Compute dot product in last dim (B, T)
        # dot_product = torch.sum(preds * targets, dim=-1)
        # # Clamp to avoid NaN
        # dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # return torch.acos(dot_product)


    def _sin_cos(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts sine and cosine components from the last dimension of a tensor.

        Args:
            x (torch.Tensor): Tensor of shape (batch, 2) containing [sin, cos].
        Returns:
            sin_x, cos_x: Separate 1D tensors for sine and cosine.
        """
        # Assume x has two columns: sin and cos
        sin_x = x[:, 0]
        cos_x = x[:, 1]
        return sin_x, cos_x    

    # def forward(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
    #     self.update(preds.detach(), targets.detach(), **kwargs)
    #     core_loss = self._calc_unreduced_loss(preds, targets).mean()
    #     return core_loss + self._unit_circle_penalty(preds, targets)

    def forward(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        self.update(preds.detach(), targets.detach(), **kwargs)
        core_loss = self._calc_unreduced_loss(preds, targets).mean()
        return core_loss + self._unit_circle_penalty(preds, targets)

    
    def _unit_circle_penalty(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Penalizes deviation from the unit circle in (sin, cos) outputs.

        Args:
            preds: Tensor of shape (B, 2 * T), where each angle is (sin, cos)
            targets: Tensor of same shape, with ignore values possibly present

        Returns:
            Scalar penalty term
        """
        B, D = preds.shape
        T = D // 2
        pred_sin = preds[:, 0::2]
        pred_cos = preds[:, 1::2]
        target_sin = targets[:, 0::2]
        target_cos = targets[:, 1::2]

        if self.ignore_value is not None:
            valid = (target_sin != self.ignore_value) & (target_cos != self.ignore_value)
        else:
            valid = torch.ones(B, dtype=torch.bool, device=preds.device)
        if not valid.any():
            return torch.tensor(0.0, device=preds.device)

        # Compute magnitude squared of each (sin, cos) pair
        mag_sq = pred_sin**2 + pred_cos**2

        # Compute penalty only on valid entries
        penalty = ((mag_sq[valid] - 1.0) ** 2).mean()

        # Learnable scale via softplus or fixed gamma
        gamma = F.softplus(self._gamma_unconstrained) if hasattr(self, "_gamma_unconstrained") else self.gamma
        return gamma * penalty



@MetricRegistry.register("SinCosMSELoss")
@LossRegistry.register("SinCosMSELoss")
class SinCosMSELoss(CMPNNMetric):
    """
    Mean squared error loss for sin/cos predictions.
    """

    def __init__(self, task_weights=None, alias="SinCosMSELoss", ignore_value=None, **kwargs):
        super().__init__(task_weights=task_weights, alias=alias, ignore_value=ignore_value, **kwargs)
        self.ignore_value = ignore_value

    def _init_states(self, num_tasks: int):
        """
        num_tasks here is preds.size(1) == 2*T.
        We want T = num_tasks // 2.
        """
        super()._init_states(num_tasks // 2)

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        Calculate the unreduced loss. Must be implemented by subclasses.
        """
        # Extract sin & cos
        sin_pred, cos_pred = preds[:, 0], preds[:, 1]
        sin_true, cos_true = targets[:, 0], targets[:, 1]


        sin_err = F.mse_loss(sin_pred, sin_true, reduction='none')  # shape (B,)
        cos_err = F.mse_loss(cos_pred, cos_true, reduction='none')  # shape (B,)
        return sin_err + cos_err                                   # shape (B,)
    

@MetricRegistry.register("AngularMSELoss")
@LossRegistry.register("AngularMSELoss")
class AngularMSELoss(CMPNNMetric):
    """
    Mean squared error loss for angular predictions.
    """

    def __init__(self, task_weights=None, alias="AngularMSELoss", ignore_value=None, **kwargs):
        super().__init__(task_weights=task_weights, alias=alias, ignore_value=ignore_value, **kwargs)
        self.ignore_value = ignore_value

    def _init_states(self, num_tasks: int):
        """
        num_tasks here is preds.size(1) == 2*T.
        We want T = num_tasks // 2.
        """
        super()._init_states(num_tasks // 2)
    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        Calculate the unreduced loss. Must be implemented by subclasses.
        """
        pred_theta = torch.atan2(preds[:, 0], preds[:, 1])
        target_theta = torch.atan2(targets[:, 0], targets[:, 1])

        delta = pred_theta - target_theta
        delta = torch.remainder(delta + math.pi, 2 * math.pi) - math.pi

        return delta ** 2

# ─── metrics.py (or wherever your metric classes live) ────────────
@MetricRegistry.register("pinball")
@LossRegistry.register("pinball")
class PinballLoss(CMPNNMetric):
    """
    Quantile (pinball) loss for regression.
    
    Parameters
    ----------
    q : float          desired quantile ∈ (0,1)
    task_weights : …   per-target weights, same semantics as MSE/MAE
    """
    def __init__(self, q: float = 0.9, task_weights=None, **kw):
        alias = f"pinball_q{int(q*100)}"   # → pinball_q90
        super().__init__(task_weights=task_weights, alias=alias, **kw)
        if not 0 < q < 1:
            raise ValueError("q must be inside (0,1)")
        self.q = q

    def _calc_unreduced_loss(self, preds: torch.Tensor, targets: torch.Tensor):
        diff = targets - preds
        left  = self.q * torch.clamp(diff, min=0)          # under-estimation
        right = (1 - self.q) * torch.clamp(-diff, min=0)   # over-estimation
        return left + right
