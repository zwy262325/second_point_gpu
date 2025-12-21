import torch
from torch import nn

from .patch import PatchEmbedding
from .positional_encoding import PositionalEncoding
# from .transformer_layers import TransformerLayers
from .agcrn1 import AVWGCN
import numpy as np  # 新增：用于掩蔽生成
from .augmentations import masked_data, geom_noise_mask_single # 新增：导入augmentations中的掩蔽函数
from .transformer_layers_new import TransformerLayers
from .tools import ContrastiveWeight, AggregationRebuild, DataEmbedding


class Pooler_Head(nn.Module):
    def __init__(self, seq_len, d_model, compression_ratio,head_dropout=0):
        super().__init__()

        pn = seq_len * d_model
        hidden_dim = int(pn * compression_ratio)
        dimension = 32
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dimension),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):  # [(bs * n_vars) x seq_len x d_model]
        x = self.pooler(x) # [(bs * n_vars) x dimension]
        return x

class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len*d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x seq_len x d_model]
        x = self.flatten(x) # [bs x n_vars x (seq_len * d_model)]
        x = self.linear(x) # [bs x n_vars x seq_len]
        x = self.dropout(x) # [bs x n_vars x seq_len]
        return x

class Mask(nn.Module):

    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, mask_ratio, encoder_depth,
                 decoder_depth, dim_in, dim_out, agcn_embed_dim, cheb_k, num_node, input_len,mask_distribution='geometric', lm=3, positive_nums=3, temperature =0.1, compression_ratio = 0.1, mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio


        # 新增：初始化augmentations掩蔽参数、
        self.mask_distribution = mask_distribution  # 掩蔽分布类型
        self.lm = lm  # 几何分布的平均掩蔽长度（仅当distribution='geometric'时使用）
        self.positive_nums = positive_nums # 数据的副本数量
        self.masked_data = masked_data
        # Embedding
        self.enc_embedding = DataEmbedding(1, 32)
        # encoder_new
        self.encoder_new = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        # for series-wise representation
        self.pooler = Pooler_Head(input_len, embed_dim, compression_ratio,head_dropout=dropout)
        self.contrastive = ContrastiveWeight(temperature,positive_nums)
        self.aggregation = AggregationRebuild(temperature, positive_nums)
        self.projection = Flatten_Head(input_len, embed_dim, input_len, head_dropout=dropout)

    def encoding_decoding(self, long_term_history):

        long_term_history = long_term_history.permute(0, 2, 1, 3)
        x_enc, mask_index = self.masked_data(long_term_history, self.mask_ratio, self.lm, self.positive_nums, distribution='geometric')
        batch_size, num_nodes,_, _ = long_term_history.shape
        bs, seq_len, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)  # x_enc: [bs x n_vars x seq_len]
        x_enc = x_enc.reshape(-1, seq_len, 1)  # x_enc: [(bs * n_vars) x seq_len x 1]
        enc_out = self.enc_embedding(x_enc)  # enc_out: [(bs * n_vars) x seq_len x d_model]
        p_enc_out = self.encoder_new(enc_out)  # p_enc_out: [(bs * n_vars) x seq_len x d_model]
        s_enc_out = self.pooler(p_enc_out)  # s_enc_out: [(bs * n_vars) x dimension]
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out)  # similarity_matrix: [(bs * n_vars) x (bs * n_vars)]
        rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out)  # agg_enc_out: [(bs * n_vars) x seq_len x d_model]
        _, seq_len, nums_dim = agg_enc_out.shape
        agg_enc_out = agg_enc_out.reshape(batch_size * (self.positive_nums + 1), num_nodes, seq_len, nums_dim)  # agg_enc_out: [bs x n_vars x seq_len x d_model]
        dec_out = self.projection(agg_enc_out)  # dec_out: [bs x n_vars x seq_len]
        dec_out = dec_out.view(batch_size * (self.positive_nums + 1), num_nodes, 1, -1) # dec_out: [bs x seq_len x n_vars](8,207,24,12)
        dec_out = dec_out[:batch_size]
        dec_out = dec_out.permute(0, 1, 3, 2)
        return dec_out, long_term_history, loss_cl

    def encoding(self, long_term_history):
        long_term_history = long_term_history.permute(0, 2, 1, 3)
        batch_size, num_nodes, seq_len, _ = long_term_history.shape
        x_enc = long_term_history.reshape(batch_size * num_nodes, long_term_history.shape[2], long_term_history.shape[3])  # [(bs * n_vars) x seq_len x d_model]
        x_enc = x_enc.permute(0, 2, 1)  # x_enc: [bs x n_vars x seq_len]
        x_enc = x_enc.reshape(-1, seq_len, 1)  # x_enc: [(bs * n_vars) x seq_len x 1]
        enc_out = self.enc_embedding(x_enc)  # enc_out: [(bs * n_vars) x seq_len x d_model]
        p_enc_out = self.encoder_new(enc_out)  # p_enc_out: [(bs * n_vars) x seq_len x d_model]
        _, _, nums_dim = p_enc_out.shape
        p_enc_out = p_enc_out.reshape(batch_size, num_nodes, seq_len, nums_dim)  # agg_enc_out: [bs x n_vars x seq_len x d_model]
        # dec_out = self.projection(p_enc_out)  # dec_out: [bs x n_vars x seq_len]
        # dec_out = dec_out.view(batch_size , num_nodes, 1, -1)  # dec_out: [bs x seq_len x n_vars](8,207,24,12)

        return p_enc_out

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None,
                epoch: int = None, **kwargs) -> torch.Tensor:
        if self.mode == "pre-train":
            predict_result, original_result, loss_cl = self.encoding_decoding(history_data)
            return predict_result, original_result, loss_cl
        else:
            hidden_states_full = self.encoding(history_data)
            return hidden_states_full
