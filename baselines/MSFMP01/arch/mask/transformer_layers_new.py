import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            hidden_dim * mlp_ratio,
            dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        # 输入维度：BN × T × D (如 6624 × 72 × 96)
        BN, T, D = src.shape

        # 按原Transformer的初始化缩放（可选，但建议保留）
        src = src * math.sqrt(self.d_model)

        # 由于输入已经是三维 (batch_first=True 要求的 [batch, seq_len, feature])
        # 无需额外重塑，直接传入transformer
        output = self.transformer_encoder(src, mask=None)

        # 输出维度保持 BN × T × D，无需reshape
        return output