import torch
import numpy as np


def masked_temporal_frequency_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        null_val: float = np.nan,
        alpha: float = 0.95,  # 时域损失权重
        beta: float = 0.05 # 频域损失权重
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算带掩码的时域平方损失和频域绝对值损失，同时忽略目标张量中的空值。

    Args:
        prediction (torch.Tensor): 预测值张量
        target (torch.Tensor): 目标值张量（与预测值形状相同）
        null_val (float, optional): 视为空值/缺失值的数值，默认为np.nan
        alpha (float, optional): 时域损失的权重，默认为1.0
        beta (float, optional): 频域损失的权重，默认为1.0

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            总损失、时域损失、频域损失（均为标量张量）
    """
    # 1. 创建掩码（与原逻辑完全一致）
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        null_tensor = torch.tensor(null_val, device=target.device).expand_as(target)
        mask = ~torch.isclose(target, null_tensor, atol=eps, rtol=0.0)

    # 掩码归一化，避免有效样本数对损失的影响
    mask = mask.float()
    mask = mask / torch.mean(mask) if torch.mean(mask) > 0 else mask
    mask = torch.nan_to_num(mask)  # 处理可能的NaN

    # 2. 计算时域平方损失（带掩码）
    temporal_error = torch.abs(prediction - target)
    temporal_error = temporal_error * mask
    temporal_error = torch.nan_to_num(temporal_error)
    loss_tmp = torch.mean(temporal_error)

    # 3. 计算频域绝对值损失（带掩码）
    # 先对预测值和目标值应用掩码，再做傅里叶变换
    pred_masked = prediction * mask
    target_masked = target * mask

    pred_fft = torch.fft.rfft(pred_masked, dim=1)
    target_fft = torch.fft.rfft(target_masked, dim=1)

    frequency_error = (pred_fft - target_fft).abs()
    frequency_error = torch.nan_to_num(frequency_error)
    loss_feq = torch.mean(frequency_error)

    # 4. 计算总损失（可通过alpha/beta调节权重）
    total_loss = alpha * loss_tmp + beta * loss_feq

    return total_loss