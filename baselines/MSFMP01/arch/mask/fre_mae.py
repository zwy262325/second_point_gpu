import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
from basicts.metrics import masked_mae

class FreDFLossBNTD(nn.Module):
    def __init__(self, alpha=0.1):
        super(FreDFLossBNTD, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # 1. 计算时域损失 (始终计算)
        loss_time = masked_mae(pred, target)

        # 2. 课程学习逻辑判断 (步长是否达到 12)
        if pred.shape[1] < 12:
            return loss_time
        print("使用频域损失")
        # 3. 频域损失计算 (只有步长 >= 12 才执行)
        pred_filled = torch.nan_to_num(pred, nan=0.0)
        target_filled = torch.nan_to_num(target, nan=0.0)

        # 执行 rfft
        pred_fft = fft.rfft(pred_filled, dim=1, norm='ortho')
        target_fft = fft.rfft(target_filled, dim=1, norm='ortho')

        # 移除直流分量 (DC Component)
        pred_fft = pred_fft[:, 1:, :, :]
        target_fft = target_fft[:, 1:, :, :]

        loss_freq_real = torch.mean(torch.abs(pred_fft.real - target_fft.real))
        loss_freq_imag = torch.mean(torch.abs(pred_fft.imag - target_fft.imag))
        loss_freq = loss_freq_real + loss_freq_imag

        # 4. 联合损失
        total_loss = (1 - self.alpha) * loss_time + self.alpha * loss_freq
        return total_loss


def masked_fredf_loss(prediction: torch.Tensor, target: torch.Tensor):
    criterion = FreDFLossBNTD(alpha=0.1).to(prediction.device)
    return criterion(prediction, target)