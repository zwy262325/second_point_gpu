import sys

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
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
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
        self.mask_distribution = simMTM_args.mask_distribution
        self.lm = simMTM_args.lm
        self.positive_nums = simMTM_args.positive_nums
        self.masked_data = masked_data
        self.mse = torch.nn.MSELoss()
        self.awl = AutomaticWeightedLoss(2)

        # encoder_new
        self.encoder_new = TransformerLayers(32, encoder_depth, mlp_ratio, num_heads, dropout)

        #Encoder
        # self.encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 DSAttention(False, transformer_args.factor, attention_dropout=transformer_args.dropout,
        #                             output_attention=transformer_args.output_attention), transformer_args.d_model, transformer_args.n_heads),
        #             transformer_args.d_model,
        #             transformer_args.d_ff,
        #             dropout=transformer_args.dropout,
        #             activation=transformer_args.activation
        #         ) for l in range(transformer_args.e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(transformer_args.d_model),
        # )

        # for series-wise representation
        self.pooler = Pooler_Head(input_len, transformer_args.d_model, simMTM_args.compression_ratio,head_dropout=dropout)
        self.contrastive = ContrastiveWeight(simMTM_args.temperature,simMTM_args.positive_nums)
        self.aggregation = AggregationRebuild(simMTM_args.temperature, simMTM_args.positive_nums)
        self.projection = Flatten_Head(input_len, transformer_args.d_model, input_len, head_dropout=dropout)
        # 新加
        self.fc_patch_size = nn.Sequential(nn.Linear(self.embed_dim, patch_size))
        # Embedding
        # self.enc_embedding = DataEmbedding(1, 96)

        self.node_embeddings = nn.Parameter(torch.randn(num_node, agcn_embed_dim), requires_grad=True)
        self.AVWGCN = AVWGCN(dim_in, dim_out, cheb_k, agcn_embed_dim)
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        # # positional encoding
        self.positional_encoding = PositionalEncoding()

    import torch
    import sys

    def encoding_decoding(self, long_term_history):
        # 定义核心校验函数：检测NaN/inf，若存在则打印信息并立即终止
        def check_tensor_and_terminate(title, tensor, print_shape=False, print_stats=False):
            """
            张量校验函数（含终止逻辑）
            :param title: 校验标题（用于区分不同阶段）
            :param tensor: 要校验的张量
            :param print_shape: 是否打印张量形状
            :param print_stats: 是否打印张量的极值、均值统计
            :raises SystemExit: 当检测到NaN/inf时终止程序
            """
            # 基础NaN/inf校验
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            # 格式化输出标题
            print(f"\n{'=' * 20} {title} {'=' * 20}")
            if print_shape:
                print(f"张量形状: {tensor.shape}")
            print(f"含NaN: {has_nan}, 含inf: {has_inf}")

            # 检测到NaN/inf时，打印错误信息并立即终止
            if has_nan or has_inf:
                print(f"❌ 错误：在【{title}】中检测到NaN/inf，程序立即终止！")
                # 方式1：抛出异常（可被try-except捕获）
                # raise ValueError(f"Detected NaN/inf in {title}")
                # 方式2：直接终止程序（不可捕获，强制停止）
                sys.exit(1)

            # 可选的统计信息打印（仅当无NaN/inf时执行）
            if print_stats and tensor.numel() > 0:
                print(
                    f"最大值: {tensor.max().item():.4f}, 最小值: {tensor.min().item():.4f}, 均值: {tensor.mean().item():.4f}")

        # 定义普通打印函数（用于非张量的统计信息，无终止逻辑）
        def print_stats_info(title, stats_dict):
            """
            打印统计信息的辅助函数
            :param title: 统计标题
            :param stats_dict: 统计信息字典（key为名称，value为数值）
            """
            print(f"\n{'=' * 20} {title} {'=' * 20}")
            for key, value in stats_dict.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

        # ===================== 第一阶段：输入及中间张量基础校验（含终止）=====================
        check_tensor_and_terminate("long_term_history基础校验", long_term_history, print_shape=True)

        batch_size, num_nodes, _, _ = long_term_history.shape

        mid_patches = self.patch_embedding(long_term_history)  # B, N, d, P (8,207,1,864)
        mid_patches = mid_patches.transpose(-1, -2)  # B, N, P, d (8,207,72,96)
        check_tensor_and_terminate("mid_patches基础校验", mid_patches, print_shape=True)

        agcrn_hidden_states = self.AVWGCN(mid_patches, self.node_embeddings)  # (8,207,72,96)
        check_tensor_and_terminate("agcrn_hidden_states校验", agcrn_hidden_states, print_shape=True)

        patches = self.positional_encoding(agcrn_hidden_states)  # BNTD(8,207,72,96)
        check_tensor_and_terminate("patches校验", patches, print_shape=True)

        # 1.生成多个掩蔽副本/掩蔽矩阵 并与原始序列拼接
        x_enc, mask_index = self.masked_data(patches, self.mask_ratio, self.lm, self.positive_nums,
                                             distribution='geometric')
        x_enc = x_enc.float()
        mask_index = mask_index.float()

        # ===================== 第二阶段：掩蔽数据校验（含终止）=====================
        print(f"\n{'=' * 20} 输入数据基础校验 {'=' * 20}")
        # 校验x_enc（含终止）
        has_nan_x = torch.isnan(x_enc).any().item()
        has_inf_x = torch.isinf(x_enc).any().item()
        print(f"x_enc - 形状: {x_enc.shape}, 含NaN: {has_nan_x}, 含inf: {has_inf_x}")
        if has_nan_x or has_inf_x:
            print(f"❌ 错误：在【x_enc】中检测到NaN/inf，程序立即终止！")
            sys.exit(1)
        # 校验mask_index（含终止）
        has_nan_mask = torch.isnan(mask_index).any().item()
        has_inf_mask = torch.isinf(mask_index).any().item()
        print(f"mask_index - 形状: {mask_index.shape}, 含NaN: {has_nan_mask}, 含inf: {has_inf_mask}")
        if has_nan_mask or has_inf_mask:
            print(f"❌ 错误：在【mask_index】中检测到NaN/inf，程序立即终止！")
            sys.exit(1)

        # mask_index的统计信息（无终止逻辑）
        mask_sum = torch.sum(mask_index == 1, dim=2)  # 形状(16,207,32)
        mask_min = mask_sum.min().item()
        mask_max = mask_sum.max().item()
        mask_zero_count = (mask_sum == 0).sum().item()
        stats_dict = {
            "mask有效数-最小值": mask_min,
            "mask有效数-最大值": mask_max,
            "mask有效数-全0数量": mask_zero_count
        }
        print_stats_info("mask_index统计信息", stats_dict)

        # ===================== 第三阶段：归一化处理及校验（含终止）=====================
        # 归一化,处理缺失值
        mask_valid_count = torch.sum(mask_index == 1, dim=2)
        means = torch.sum(x_enc, dim=2) / mask_valid_count
        means = means.unsqueeze(2).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask_index == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=2) / mask_valid_count + 1e-5)
        stdev = stdev.unsqueeze(2).detach()
        x_enc /= stdev

        # 归一化后校验（含终止）
        print_stats_info("归一化后校验", {
            "x_enc-最大值": x_enc.max().item(),
            "x_enc-最小值": x_enc.min().item(),
            "x_enc-均值": x_enc.mean().item(),
            "stdev-最大值": stdev.max().item(),
            "stdev-最小值": stdev.min().item(),
            "stdev-均值": stdev.mean().item(),
            "means-最大值": means.max().item(),
            "means-最小值": means.min().item(),
            "means-均值": means.mean().item()
        })
        # 校验归一化后的x_enc（含终止）
        check_tensor_and_terminate("归一化后x_enc校验", x_enc, print_shape=True)

        bs, node, seq_len, n_vars = x_enc.shape

        # ===================== 第四阶段：编码器及后续层校验（含终止）=====================
        # ... 在 x_enc 校验通过之后， encoder 运行之前插入 ...

        # === 新增：检查模型权重是否健康 ===
        print(f"\n{'=' * 20} 检查 encoder_new 权重状态 {'=' * 20}")
        is_weight_bad = False
        for name, param in self.encoder_new.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"❌ 致命错误: 参数 {name} 已经损坏 (含 NaN/Inf)！")
                is_weight_bad = True
                break

        if not is_weight_bad:
            print("✅ Encoder 权重正常")
        else:
            sys.exit(1)  # 如果权重坏了，说明是上一个 Batch 的反向传播炸了

        p_enc_out = self.encoder_new(x_enc)  # 不使用通道
        check_tensor_and_terminate("encoder输出校验", p_enc_out, print_shape=True)

        # 序列特征提取
        s_enc_out = self.pooler(p_enc_out)
        check_tensor_and_terminate("pooler输出校验", s_enc_out, print_shape=True)

        # 对比学习(序列相似度计算+对比损失计算)
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out)
        # 校验similarity_matrix（含终止）
        check_tensor_and_terminate("contrastive-similarity_matrix校验", similarity_matrix, print_shape=True)
        # 校验logits（含终止）
        check_tensor_and_terminate("contrastive-logits校验", logits, print_shape=True)
        # 校验loss_cl（含终止）
        print(f"\n{'=' * 20} contrastive损失校验 {'=' * 20}")
        if torch.isnan(loss_cl) or torch.isinf(loss_cl):
            print(f"❌ 错误：在【contrastive-loss_cl】中检测到NaN/inf，程序立即终止！")
            sys.exit(1)
        print(f"对比损失loss_cl: {loss_cl.item():.6f}")

        # 节点聚合
        rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out)
        check_tensor_and_terminate("aggregation输出校验", agg_enc_out, print_shape=True)

        # ===================== 第五阶段：序列重建及最终输出（含终止）=====================
        # 形状重塑
        agg_enc_out = agg_enc_out.view(bs, node, n_vars, seq_len, -1)

        # 序列重建
        dec_out = self.projection(agg_enc_out)
        dec_out = dec_out.permute(0, 1, 3, 2)
        check_tensor_and_terminate("projection输出校验", dec_out, print_shape=True)

        # 逆标准化
        dec_out = dec_out * (stdev[:, :, 0, :].unsqueeze(2).repeat(1, 1, self.input_len, 1))
        dec_out = dec_out + (means[:, :, 0, :].unsqueeze(2).repeat(1, 1, self.input_len, 1))
        # 校验逆标准化后的dec_out（含终止）
        check_tensor_and_terminate("逆标准化后dec_out校验", dec_out, print_shape=True)

        # 最终处理
        dec_out = self.fc_patch_size(dec_out)
        dec_out = dec_out.view(batch_size * (self.positive_nums + 1), num_nodes, 1, -1)
        dec_out = dec_out[:batch_size]

        # 最终输出校验（含终止+统计）
        check_tensor_and_terminate("最终dec_out校验", dec_out, print_shape=True, print_stats=True)

        # ===================== 第六阶段：损失计算及统计（含终止）=====================
        # 重建损失计算
        loss_rb = self.mse(dec_out, long_term_history.detach())
        # 总体损失计算
        loss = self.awl(loss_cl, loss_rb)

        # 损失统计（含终止）
        print(f"\n{'=' * 20} 损失统计 {'=' * 20}")
        # 校验loss_rb
        if torch.isnan(loss_rb) or torch.isinf(loss_rb):
            print(f"❌ 错误：在【loss_rb】中检测到NaN/inf，程序立即终止！")
            sys.exit(1)
        # 校验loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ 错误：在【总损失loss】中检测到NaN/inf，程序立即终止！")
            sys.exit(1)
        print(f"重建损失loss_rb: {loss_rb.item():.6f}")
        print(f"对比损失loss_cl: {loss_cl.item():.6f}")
        print(f"总损失loss: {loss.item():.6f}")

        return dec_out, long_term_history, loss

    def encoding(self, long_term_history):
        batch_size, num_nodes, _, _ = long_term_history.shape

        mid_patches = self.patch_embedding(long_term_history)
        mid_patches = mid_patches.transpose(-1, -2)

        agcrn_hidden_states = self.AVWGCN(mid_patches, self.node_embeddings)

        patches = self.positional_encoding(agcrn_hidden_states)
        p_enc_out, _ = self.encoder(patches)

        _, _, seq_len, nums_dim = p_enc_out.shape
        p_enc_out = p_enc_out.permute(0, 1, 3, 2)
        p_enc_out = p_enc_out.view(batch_size, num_nodes, nums_dim, seq_len, -1)

        dec_out = self.projection(p_enc_out)
        dec_out = dec_out.permute(0, 1, 3, 2)
        dec_out = self.fc_patch_size(dec_out)
        dec_out = dec_out.view(batch_size, num_nodes, 1, -1)

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
