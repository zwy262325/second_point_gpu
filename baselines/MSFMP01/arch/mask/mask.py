import torch
from torch import nn

from .patch import PatchEmbedding
from .positional_encoding import PositionalEncoding
# from .transformer_layers import TransformerLayers
from .agcrn1 import AVWGCN
import numpy as np  # 新增：用于掩蔽生成
from .augmentations import masked_data, geom_noise_mask_single # 新增：导入augmentations中的掩蔽函数
from .transformer_layers_new import TransformerLayers
from .tools import ContrastiveWeight, AggregationRebuild


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
    def __init__(self, d_model, patch_size, head_dropout=0):
        super().__init__()
        self.linear = nn.Linear(d_model, patch_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
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
        # encoder_new
        self.encoder_new = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        # for series-wise representation
        self.pooler = Pooler_Head(input_len, embed_dim, compression_ratio,head_dropout=dropout)
        self.contrastive = ContrastiveWeight(temperature,positive_nums)
        self.aggregation = AggregationRebuild(temperature, positive_nums)
        self.projection = Flatten_Head(embed_dim, patch_size, head_dropout=dropout)


        self.node_embeddings = nn.Parameter(torch.randn(num_node, agcn_embed_dim), requires_grad=True)
        self.AVWGCN = AVWGCN(dim_in, dim_out, cheb_k, agcn_embed_dim)
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        # # positional encoding
        self.positional_encoding = PositionalEncoding()

    def encoding_decoding(self, long_term_history):
        mid_patches = self.patch_embedding(long_term_history)  # B, N, d, P (8,207,1,864)
        mid_patches = mid_patches.transpose(-1, -2)  # B, N, P, d (8,207,72,96)

        # batch_size, num_nodes, num_time, num_dim = mid_patches.shape
        agcrn_hidden_states = self.AVWGCN(mid_patches, self.node_embeddings)  # (8,207,72,96)
        patches = self.positional_encoding(agcrn_hidden_states)  # BNTD(8,207,72,96)

        # 1.生成多个掩蔽副本/掩蔽矩阵 并与原始序列拼接 batch_x_om_3d(6624,72,96) B * (positive_num + 1)DT, mask_om_3d(6624,72,96)
        x_enc, mask_index = self.masked_data(patches, self.mask_ratio, self.lm, self.positive_nums, distribution='geometric')

        # 2.x_enc BTD(8,48,7)
        batch_size, num_nodes, _, _ = long_term_history.shape

        # 5.节点特征提取 encoder point-wise representation p_enc_out(56,48,128) 使用Transformer
        p_enc_out = self.encoder_new(x_enc)  # p_enc_out: [(bs * n_vars) x seq_len x d_model]

        # 6.序列特征提取 series-wise representation s_enc_out(56,128) 使用MLP将48个时间步的信息压缩到一个固定长度的向量中
        s_enc_out = self.pooler(p_enc_out)  # s_enc_out: [(bs * n_vars) x dimension]

        # 7.对比学习(序列相似度计算+对比损失计算)
        # loss_cl KL散度的损失值,惩罚和奖励 similarity_matrix(56,56)相似度矩阵 logits(56,55)合并正负样本相似度 positives_mask(56,56)正样本掩码
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out)  # similarity_matrix: [(bs * n_vars) x (bs * n_vars)]

        # 8.节点聚合(基于相似度矩阵对节点特征进行加权平均,每个样本用其他样本特征增强自己)
        # 重建权重矩阵(56,56)  加强后的序列agg_enc_out(56,48,128)
        rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out)  # agg_enc_out: [(bs * n_vars) x seq_len x d_model]

        # # 9.agg_enc_out(8,7,48,128)
        _, seq_len, nums_dim = agg_enc_out.shape
        agg_enc_out = agg_enc_out.reshape(batch_size * (self.positive_nums + 1), num_nodes, seq_len, nums_dim)  # agg_enc_out: [bs x n_vars x seq_len x d_model]

        # 10.序列重建 decoder dec_out(8,7,48) agg_enc_out.permute(0,3,2,1)(8,128,48,7)
        dec_out = self.projection(agg_enc_out)  # dec_out: [bs x n_vars x seq_len]
        dec_out = dec_out.view(batch_size * (self.positive_nums + 1), num_nodes, 1, -1) # dec_out: [bs x seq_len x n_vars](8,207,24,12)

        # 11.从重建的8个样本中，取出前2个pred_batch_x (2,48,7)
        dec_out = dec_out[:batch_size]

        # 12.重建损失计算
        # loss_rb = self.mse(pred_batch_x, patches.detach())

        # 13.总体损失计算
        # loss = self.awl(loss_cl, loss_rb)

        return dec_out, long_term_history, loss_cl

    def encoding(self, long_term_history):
        mid_patches = self.patch_embedding(long_term_history)  # B, N, d, P (8,207,1,864)
        mid_patches = mid_patches.transpose(-1, -2)  # B, N, P, d (8,207,72,96)

        # batch_size, num_nodes, num_time, num_dim = mid_patches.shape
        agcrn_hidden_states = self.AVWGCN(mid_patches, self.node_embeddings)  # (8,207,72,96)
        patches = self.positional_encoding(agcrn_hidden_states)  # BNTD(8,207,72,96)

        # 2.x_enc BTD(8,48,7)
        batch_size, num_nodes, _, _ = long_term_history.shape

        patches_reshaped = patches.reshape(batch_size * num_nodes, patches.shape[2], patches.shape[3])  # [(bs * n_vars) x seq_len x d_model]

        # 5.节点特征提取 encoder point-wise representation p_enc_out(56,48,128) 使用Transformer
        p_enc_out = self.encoder_new(patches_reshaped)  # p_enc_out: [(bs * n_vars) x seq_len x d_model]

        # # 9.agg_enc_out(8,7,48,128)
        _, seq_len, nums_dim = p_enc_out.shape
        p_enc_out = p_enc_out.reshape(batch_size, num_nodes, seq_len, nums_dim)  # agg_enc_out: [bs x n_vars x seq_len x d_model]

        # 10.序列重建 decoder dec_out(8,7,48) agg_enc_out.permute(0,3,2,1)(8,128,48,7)
        dec_out = self.projection(p_enc_out)  # dec_out: [bs x n_vars x seq_len]
        dec_out = dec_out.view(batch_size , num_nodes, 1, -1)  # dec_out: [bs x seq_len x n_vars](8,207,24,12)

        return dec_out

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None,
                epoch: int = None, **kwargs) -> torch.Tensor:
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)  # B, N, 1, L * P
        # feed forward
        if self.mode == "pre-train":
            predict_result, original_result, loss_cl = self.encoding_decoding(history_data)
            return predict_result, original_result, loss_cl
        else:
            hidden_states_full = self.encoding(history_data)
            return hidden_states_full
