import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

plt.switch_backend('agg')

class ContrastiveWeight(nn.Module):

    def __init__(self, temperature, positive_nums):
        super(ContrastiveWeight, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.positive_nums = positive_nums

    import torch
    import torch.nn.functional as F
    # 注意：移除了 import numpy as np

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size):
        # 确保 device 从输入张量获取
        device = similarity_matrix.device

        # 1. 生成对角线掩码 (mask)
        mask = torch.eye(cur_batch_size, dtype=torch.bool, device=device)

        oral_batch_size = cur_batch_size // (self.positive_nums + 1)

        # 2. 初始化正样本掩码 (positives_mask) - 必须是 float 或 bool，这里用 bool 效率更高
        positives_mask = torch.zeros(cur_batch_size, cur_batch_size, dtype=torch.bool, device=device)

        for i in range(self.positive_nums + 1):
            k_val = oral_batch_size * i

            # 确保偏移量不超过矩阵大小
            if k_val >= cur_batch_size:
                break

            # 使用纯 PyTorch 操作生成偏移对角线 (相当于 np.eye(..., k=...))
            diag_len = cur_batch_size - abs(k_val)

            # 创建对角线上的值
            ones_diag = torch.ones(diag_len, device=device, dtype=torch.bool)

            # 上偏移 k (ll)
            ll = torch.diag(ones_diag, diagonal=k_val)
            # 下偏移 -k (lr)
            lr = torch.diag(ones_diag, diagonal=-k_val)

            # 累加掩码 (使用逻辑或 | 代替加法)
            positives_mask = positives_mask | ll | lr

        # 3. 移除自身 (对角线)
        # positives_mask[mask] = 0 的 PyTorch 等价形式
        positives_mask[mask] = False

        # 4. 生成负样本掩码
        # 1 - positives_mask ( float/int ) 替换为逻辑非 ~
        # 确保负样本掩码中也排除了对角线
        negatives_mask = (~positives_mask) & (~mask)

        return positives_mask, negatives_mask

    def forward(self, batch_emb_om):
        # batch_emb_om形状: (T, N, D)
        cur_batch_shape = batch_emb_om.shape  # (T, N, D)

        # 对每个节点的特征进行归一化
        norm_emb = F.normalize(batch_emb_om, dim=2)  # (T, N, D)

        # 转置以便向量化计算
        # 形状: (N, T, D)
        norm_emb_transposed = norm_emb.transpose(0, 1)

        # 计算批处理相似度矩阵
        # 使用torch.matmul的批处理功能
        similarity_matrix = torch.matmul(
            norm_emb_transposed,  # (N, T, D)
            norm_emb_transposed.transpose(1, 2)  # (N, D, T)
        )  # 结果形状: (N, T, T)

        # 初始化存储
        batch_size_T = cur_batch_shape[0]
        batch_size_N = cur_batch_shape[1]

        total_loss = 0
        all_logits = []

        # 对每个节点独立处理mask和loss计算
        for n in range(batch_size_N):
            node_sim_matrix = similarity_matrix[n]  # (T, T)

            # 获取正负样本mask
            positives_mask, negatives_mask = self.get_positive_and_negative_mask(
                node_sim_matrix, batch_size_T)

            # 提取正负样本
            num_positives = positives_mask.sum() // batch_size_T
            num_negatives = negatives_mask.sum() // batch_size_T

            positives = node_sim_matrix[positives_mask].view(batch_size_T, num_positives)
            negatives = node_sim_matrix[negatives_mask].view(batch_size_T, num_negatives)

            # 构建logits和labels
            logits = torch.cat((positives, negatives), dim=-1)  # (T, num_pos+num_neg)
            y_true = torch.cat((
                torch.ones(batch_size_T, num_positives),
                torch.zeros(batch_size_T, num_negatives)
            ), dim=-1).to(batch_emb_om.device).float()

            # 计算loss
            predict = self.log_softmax(logits / self.temperature)
            node_loss = self.kl(predict, y_true)

            total_loss += node_loss
            all_logits.append(logits.unsqueeze(0))

        # 平均loss
        avg_loss = total_loss / batch_size_N

        # 合并logits
        logits_all = torch.cat(all_logits, dim=0)  # (N, T, num_samples)

        return avg_loss, similarity_matrix, logits_all, None


class AggregationRebuild(torch.nn.Module):

    def __init__(self, temperature, positive_nums):
        super(AggregationRebuild, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.positive_nums = positive_nums

    def forward(self, similarity_matrix, batch_emb_om):

        cur_batch_shape = batch_emb_om.shape

        # 1.缩放相似度similarity_matrix(56,56)
        similarity_matrix /= self.temperature

        device = similarity_matrix.device

        # 2.排除自身相似度影响(对角线置为极小值) rebuild_weight_matrix(56,56)
        eye_mask = torch.eye(cur_batch_shape[0], device=device, dtype=similarity_matrix.dtype)
        eye_mask = eye_mask.unsqueeze(0)
        similarity_matrix = similarity_matrix - eye_mask * 1e12

        rebuild_weight_matrix = self.softmax(similarity_matrix)

        # 3.重塑三维batch_emb_om(56,48,128)->二维batch_emb_om(56,6144)
        batch_emb_om = batch_emb_om.reshape(cur_batch_shape[1],cur_batch_shape[0], -1)

        # 4.矩阵乘法加权重建batch_emb_om(56,6144)
        rebuild_batch_emb = torch.bmm(rebuild_weight_matrix, batch_emb_om)

        # 5.恢复为原来的三维形状
        rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1],cur_batch_shape[2], -1)

        return rebuild_weight_matrix, rebuild_oral_batch_emb


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.d_model = d_model
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        B_D, N, L, C = x.shape
        x_reshaped = x.reshape(B_D * N, L, C)
        x_conv = self.tokenConv(x_reshaped.permute(0, 2, 1))
        x_out = x_conv.transpose(1, 2).reshape(B_D, N, L, self.d_model)
        return x_out


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x)
        return self.dropout(x)