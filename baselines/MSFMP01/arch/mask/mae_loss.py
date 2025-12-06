import torch
import numpy as np
import torch.nn as nn  # 导入 nn 模块以便使用 nn.Module


# ====================================================================
# 1. 自动加权损失模块
# ====================================================================
class AutomaticWeightedLoss(nn.Module):
    """
    自动加权多任务损失 (Uncertainty Weighting)
    通过学习参数 sigma_i 来动态分配每个损失项的权重。

    Args:
        num: int, 损失函数的数量。

    Returns:
        torch.Tensor: 加权后的总损失。
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        # 初始化 num 个可学习参数，作为标准差 sigma_i
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *x):
        # 接收任意数量的损失张量
        loss_sum = 0
        for i, loss in enumerate(x):
            # 自动加权公式： (1 / (2 * sigma_i^2)) * L_i + log(1 + sigma_i^2)
            # 其中 self.params[i] 就是 sigma_i
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def mae_loss_cl(prediction: torch.Tensor, target: torch.Tensor, loss_cl: torch.Tensor, awl_module: AutomaticWeightedLoss = None, scaler = None, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Mean Absolute Error (MAE) between the predicted and target values,
    while ignoring the entries in the target tensor that match the specified null value.

    This function is particularly useful for scenarios where the dataset contains missing or irrelevant
    values (denoted by `null_val`) that should not contribute to the loss calculation. It effectively
    masks these values to ensure they do not skew the error metrics.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor.
            Default is `np.nan`. The function will mask all `NaN` values in the target.
        loss_cl (torch.Tensor, optional): The contrastive learning loss to be added to the reconstruction loss.
            Should be a scalar tensor. Defaults to None, which is treated as 0.0.
        awl_module (AutomaticWeightedLoss, optional): An instantiated AutomaticWeightedLoss module (nn.Module).
            If provided, it calculates the weighted sum of MAE and loss_cl. If None, the losses are simply summed.
    Returns:
        torch.Tensor: A scalar tensor representing the masked mean absolute error.

    """
    if loss_cl is None:
        loss_cl = torch.tensor(0.0, device=prediction.device)

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    prediction_norm = prediction.clone()
    target_norm = target.clone()
    prediction_norm = scaler.transform(prediction_norm)
    target_norm = scaler.transform(target_norm)
    loss = torch.abs(prediction_norm - target_norm)
    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero
    loss_mae_mean = torch.mean(loss)
    total_loss = awl_module(loss_mae_mean, loss_cl)

    return total_loss
