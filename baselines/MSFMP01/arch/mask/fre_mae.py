# baselines/MSFMP01/arch/mask/fre_mae.py
import torch
import torch.nn as nn
import torch.fft as fft


class FreDFLossBNTD(nn.Module):
    def __init__(self, alpha=0.1):
        super(FreDFLossBNTD, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, pred, target):
        # 1. 时域 L1 损失
        loss_time = self.l1_loss(pred, target)

        # 2. 频域变换：仅对时间维度 T (dim=2) 做 rfft
        pred_fft = fft.rfft(pred, dim=1, norm='ortho')
        target_fft = fft.rfft(target, dim=1, norm='ortho')

        # 3. 移除直流分量（第0个频率点）
        pred_fft = pred_fft[:, 1:, :, :]
        target_fft = target_fft[:, 1:, :, :]

        # 4. 频域 L1 损失：实部+虚部
        loss_freq_real = self.l1_loss(pred_fft.real, target_fft.real)
        loss_freq_imag = self.l1_loss(pred_fft.imag, target_fft.imag)
        loss_freq = loss_freq_real + loss_freq_imag

        # 5. 联合损失（返回标量张量）
        total_loss = (1 - self.alpha) * loss_time + self.alpha * loss_freq
        return total_loss


def masked_fredf_loss(prediction: torch.Tensor, target: torch.Tensor, ):
    loss_fn = FreDFLossBNTD(alpha=0.1)
    loss = loss_fn(prediction, target)
    return loss