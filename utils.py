
import torch
import torch.nn as nn
import torch.nn.functional as F

def rmse_torch(y_true, y_pred):
    """
    Compute RMSE. Accepts torch tensors or NumPy arrays.
    Returns a Python float.
    """
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

def mse_torch(y_true, y_pred):
    """
    Compute Mean Squared Error. Accepts torch tensors or NumPy arrays.
    Returns a Python float.
    """
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)
    return torch.mean((y_true - y_pred) ** 2).item()

def mae_torch(y_true, y_pred):
    """
    Compute Mean Absolute Error. Accepts torch tensors or NumPy arrays.
    Returns a Python float.
    """
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)
    return torch.mean(torch.abs(y_true - y_pred)).item()

def r2_score_torch(y_true, y_pred):
    """
    Compute Coefficient of Determination (R^2). Accepts torch tensors or NumPy arrays.
    Returns a Python float.
    """
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return (1 - ss_res / ss_tot).item()


class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # compute MSE
        loss = self.mse(pred, target)
        # add small eps for numerical stability, then sqrt
        return torch.sqrt(loss.clamp(min=self.eps))
