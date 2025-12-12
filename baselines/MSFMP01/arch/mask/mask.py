import torch
from torch import nn

from .patch import PatchEmbedding
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers
from .agcrn1 import AVWGCN
import numpy as np  # 新增：用于掩蔽生成
from .augmentations import masked_data
# from .transformer_layers_new import TransformerLayers
from .tools import ContrastiveWeight, AggregationRebuild
from .tools import DataEmbedding
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from .layers.SelfAttention_Family import DSAttention, AttentionLayer, ProbAttention
from .layers.losses import AutomaticWeightedLoss


class Pooler_Head(nn.Module):
    def __init__(self, seq_len, d_model, compression_ratio, head_dropout=0):
        super().__init__()

        pn = seq_len * d_model
        hidden_dim = int(pn * compression_ratio)
        dimension = 96
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dimension),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):  # [(bs * n_vars) x seq_len x d_model]
        B_D, N, T, C = x.shape
        x = x.reshape(B_D * N, T, C)
        x = self.pooler(x)
        x = x.reshape(B_D, N, -1)
        return x



class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        # self.linear = nn.Linear(seq_len*d_model, pred_len) # 使用通道
        self.linear = nn.Linear(seq_len, pred_len) # 不使用通道
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x seq_len x d_model]
        x = self.flatten(x) # [bs x n_vars x (seq_len * d_model)]
        x = self.linear(x) # [bs x n_vars x seq_len]
        x = self.dropout(x) # [bs x n_vars x seq_len]
        return x

class Mask(nn.Module):

    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, mask_ratio, encoder_depth,
                 decoder_depth, dim_in, dim_out, agcn_embed_dim, cheb_k, num_node, input_len, simMTM_args, transformer_args, mode="pre-train"):
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
        self.input_len = input_len


        # 新增：初始化augmentations掩蔽参数
        self.mask_distribution = simMTM_args.mask_distribution  # 掩蔽分布类型
        self.lm = simMTM_args.lm  # 几何分布的平均掩蔽长度（仅当distribution='geometric'时使用）
        self.positive_nums = simMTM_args.positive_nums # 数据的副本数量
        self.masked_data = masked_data
        self.mse = torch.nn.MSELoss()
        self.awl = AutomaticWeightedLoss(2)

        # encoder_new
        self.encoder_new = TransformerLayers(16, encoder_depth, mlp_ratio, num_heads, dropout)

        #Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, transformer_args.factor, attention_dropout=transformer_args.dropout,
                                    output_attention=transformer_args.output_attention), transformer_args.d_model, transformer_args.n_heads),
                    transformer_args.d_model,
                    transformer_args.d_ff,
                    dropout=transformer_args.dropout,
                    activation=transformer_args.activation
                ) for l in range(transformer_args.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(transformer_args.d_model),
        )

        # for series-wise representation
        self.pooler = Pooler_Head(input_len, transformer_args.d_model, simMTM_args.compression_ratio,head_dropout=dropout)
        self.contrastive = ContrastiveWeight(simMTM_args.temperature,simMTM_args.positive_nums)
        self.aggregation = AggregationRebuild(simMTM_args.temperature, simMTM_args.positive_nums)
        self.projection = Flatten_Head(input_len, transformer_args.d_model, input_len, head_dropout=dropout)
        # 新加
        self.fc_patch_size = nn.Sequential(nn.Linear(self.embed_dim, patch_size))
        # Embedding
        self.enc_embedding = DataEmbedding(1, 32)

        self.node_embeddings = nn.Parameter(torch.randn(num_node, agcn_embed_dim), requires_grad=True)
        self.AVWGCN = AVWGCN(dim_in, dim_out, cheb_k, agcn_embed_dim)
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        # # positional encoding
        self.positional_encoding = PositionalEncoding()

    def encoding_decoding(self, long_term_history):
        batch_size, num_nodes, _, _ = long_term_history.shape

        mid_patches = self.patch_embedding(long_term_history)  # B, N, d, P (8,207,1,864)
        mid_patches = mid_patches.transpose(-1, -2)  # B, N, P, d (8,207,72,96)
        agcrn_hidden_states = self.AVWGCN(mid_patches, self.node_embeddings)  # (8,207,72,96)
        patches = self.positional_encoding(agcrn_hidden_states)  # BNTD(8,207,72,96)

        # 1.生成多个掩蔽副本/掩蔽矩阵 并与原始序列拼接 batch_x_om_3d(6624,72,96) B * (positive_num + 1)DT, mask_om_3d(6624,72,96)
        x_enc, mask_index = self.masked_data(patches, self.mask_ratio, self.lm, self.positive_nums, distribution='geometric')

        x_enc = x_enc.float()
        mask_index = mask_index.float()

        # # ===================== 步骤1：输入数据基础校验 =====================
        # print("===== 输入数据基础校验 =====")
        # # 检查输入是否含NaN/inf
        # has_nan_x = torch.isnan(x_enc).any().item()
        # has_inf_x = torch.isinf(x_enc).any().item()
        # has_nan_mask = torch.isnan(mask_index).any().item()
        # has_inf_mask = torch.isinf(mask_index).any().item()
        # print(f"x_enc含NaN: {has_nan_x}, 含inf: {has_inf_x}")
        # print(f"mask_index含NaN: {has_nan_mask}, 含inf: {has_inf_mask}")
        # # 打印mask_index的统计信息（关键：有效数是否为0）
        # mask_sum = torch.sum(mask_index == 1, dim=2)  # 形状(16,207,32)
        # mask_min = mask_sum.min().item()
        # mask_max = mask_sum.max().item()
        # mask_zero_count = (mask_sum == 0).sum().item()
        # print(f"mask_sum{mask_sum}")
        # print(f"mask_min{mask_min}")
        # print(f"mask_max{mask_max}")
        #
        # print(f"mask有效数(mask_sum)统计 - 最小值: {mask_min}, 最大值: {mask_max}, 全0数量: {mask_zero_count}")

        #归一化,处理缺失值x_enc(8,48,7)
        means = torch.sum(x_enc, dim=2) / torch.sum(mask_index == 1, dim=2)
        means = means.unsqueeze(2).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask_index == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=2) / torch.sum(mask_index == 1, dim=2) + 1e-5)
        stdev = stdev.unsqueeze(2).detach()
        x_enc /= stdev
        # # ===================== 步骤2：归一化校验=====================
        # print("x_enc1:", x_enc.max())
        # print("x_enc1:", x_enc.min())
        # print("x_enc1:", x_enc.mean())
        # print("stdev:", stdev.max())
        # print("stdev:", stdev.min())
        # print("stdev:", stdev.mean())
        # print("means:", means.max())
        # print("means:", means.min())
        # print("means:", means.mean())

        bs, node, seq_len, n_vars = x_enc.shape

        # # 通道独立处理x_enc(56,48,1) 批次和特征相乘,48为时间步长
        # x_enc = x_enc.permute(0, 3, 1, 2)
        # x_enc = x_enc.unsqueeze(-1)
        # x_enc = x_enc.reshape(-1, node, seq_len, 1)
        #
        # # 特征维度转换1->128
        # enc_out = self.enc_embedding(x_enc)
        #
        # p_enc_out, _ = self.encoder(enc_out) # 使用通道
        p_enc_out, _ = self.encoder(x_enc) # 不使用通道
        # p_enc_out = self.encoder_new(x_enc)

        # 6.序列特征提取 series-wise representation s_enc_out(56,128) 使用MLP将48个时间步的信息压缩到一个固定长度的向量中
        s_enc_out = self.pooler(p_enc_out)

        # 7.对比学习(序列相似度计算+对比损失计算)
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out)  # similarity_matrix: [(bs * n_vars) x (bs * n_vars)]

        # 8.节点聚合(基于相似度矩阵对节点特征进行加权平均,每个样本用其他样本特征增强自己)
        # 重建权重矩阵(56,56)  加强后的序列agg_enc_out(56,48,128)
        rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out)

        #  9.agg_enc_out(8,7,48,128)
        agg_enc_out = agg_enc_out.reshape(bs, node, n_vars, seq_len, -1)

        # 10.序列重建 decoder dec_out(8,7,48) agg_enc_out.permute(0,3,2,1)(8,128,48,7)
        dec_out = self.projection(agg_enc_out)
        dec_out = dec_out.permute(0, 1, 3, 2)

        # 逆标准化dec_out(8,48,7),将重建的序列还原到原始数据的尺度
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(2).repeat(1, 1, self.input_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(2).repeat(1, 1, self.input_len, 1))

        dec_out = self.fc_patch_size(dec_out)
        dec_out = dec_out.view(batch_size * (self.positive_nums + 1), num_nodes, 1, -1)

        dec_out = dec_out[:batch_size]

        # 重建损失计算
        loss_rb = self.mse(dec_out, long_term_history.detach())

        # 总体损失计算
        loss = self.awl(loss_cl, loss_rb)

        return dec_out, long_term_history, loss

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
        # reshape->(8,207,1,864)
        history_data = history_data.permute(0, 2, 3, 1)  # B, N, 1, L * P
        # feed forward
        if self.mode == "pre-train":
            predict_result, original_result, loss_cl = self.encoding_decoding(history_data)
            return predict_result, original_result, loss_cl

        else:
            hidden_states_full = self.encoding(history_data)
            return hidden_states_full
