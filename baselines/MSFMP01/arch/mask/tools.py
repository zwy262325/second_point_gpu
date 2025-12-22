import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.data_preparation.CA.generate_training_data import feature_description


class ContrastiveWeight(nn.Module):
    def __init__(self, temperature, positive_nums):
        super(ContrastiveWeight, self).__init__()
        self.temperature = temperature
        self.positive_nums = positive_nums
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size):
        device = similarity_matrix.device

        # 1. 生成对角线掩码 (mask) - 直接在 GPU 上生成
        mask = torch.eye(cur_batch_size, dtype=torch.bool, device=device)

        oral_batch_size = cur_batch_size // (self.positive_nums + 1)

        # 2. 生成正样本掩码 (positives_mask) - 使用 torch.diag 直接在 GPU 生成
        # 初始化为全 False
        positives_mask = torch.zeros(similarity_matrix.size(), dtype=torch.bool, device=device)

        for i in range(self.positive_nums + 1):
            k_val = oral_batch_size * i
            if k_val >= cur_batch_size:
                break

            # 生成偏移对角线 (相当于 np.eye(..., k=...))
            # PyTorch 使用 torch.diag 配合 torch.ones 来实现
            diag_len = cur_batch_size - abs(k_val)
            ones_diag = torch.ones(diag_len, device=device, dtype=torch.bool)

            # 上偏移 k
            ll = torch.diag(ones_diag, diagonal=k_val)
            # 下偏移 -k
            lr = torch.diag(ones_diag, diagonal=-k_val)

            # 累加掩码 (逻辑或)
            positives_mask = positives_mask | ll | lr

        # 3. 移除自身 (对角线)
        positives_mask = positives_mask & (~mask)

        # 4. 生成负样本掩码
        negatives_mask = (~positives_mask) & (~mask)

        return positives_mask, negatives_mask

    def forward(self, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape

        device = batch_emb_om.device

        # 1.计算序列的相似度矩阵
        norm_emb = F.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))

        # 2.获取正样本和负样本的掩码 (现在是纯 GPU 操作)
        positives_mask, negatives_mask = self.get_positive_and_negative_mask(similarity_matrix, cur_batch_shape[0])

        # 3.根据掩码获取正样本和负样本的相似度值
        # 注意：使用 bool 索引会展平数据，需要 reshape
        positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)

        # 4.合并正负样本相似度
        logits = torch.cat((positives, negatives), dim=-1)

        # 5.构建标签
        y_true = torch.cat(
            (torch.ones(cur_batch_shape[0], positives.shape[-1], device=device),
             torch.zeros(cur_batch_shape[0], negatives.shape[-1], device=device)),
            dim=-1
        ).float()

        # 6.计算损失
        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)

        return loss, similarity_matrix, logits, positives_mask

class AggregationRebuild(torch.nn.Module):
    def __init__(self, temperature, positive_nums):
        super(AggregationRebuild, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.positive_nums = positive_nums

    def forward(self, similarity_matrix, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape
        device = similarity_matrix.device  # 获取设备

        # 1.缩放相似度
        # 避免原地操作修改原始 similarity_matrix，如果外部还需要用它的话
        similarity_matrix = similarity_matrix / self.temperature

        # 2.排除自身相似度影响 (直接在 GPU 创建 eye)
        eye_matrix = torch.eye(cur_batch_shape[0], device=device, dtype=torch.float)
        similarity_matrix = similarity_matrix - eye_matrix * 1e12

        rebuild_weight_matrix = self.softmax(similarity_matrix)

        # 3.重塑三维
        batch_emb_om = batch_emb_om.reshape(cur_batch_shape[0], -1)

        # 4.矩阵乘法加权重建
        rebuild_batch_emb = torch.matmul(rebuild_weight_matrix, batch_emb_om)

        # 5.恢复为原来的三维形状
        rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1], -1)

        return rebuild_weight_matrix, rebuild_oral_batch_emb


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, input_len,  dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.input_len = input_len
        self.add_time_in_day = "True"
        self.add_day_in_week = "True"
        self.minute_size = 288
        self.weekday_size = 7
        self.feature_dim = 1
        if self.add_time_in_day:
            self.daytime_embedding = nn.init.xavier_uniform_(nn.Parameter(torch.empty(self.minute_size, d_model)))  # 一天中不同分钟索引（0-1440）映射到embed_dim维的向量中
        if self.add_day_in_week:
            self.weekday_embedding = nn.init.xavier_uniform_(nn.Parameter(torch.empty(self.weekday_size, d_model))) # 一周中不同天索引（0-7）映射到embed_dim维的向量中

    def forward(self, x):
        origin_x = x
        # feature_x = origin_x[:, :, :self.feature_dim]
        # feature_x = feature_x.permute(0, 2, 1)  # x_enc: [bs x n_vars x seq_len]
        # feature_x = feature_x.reshape(-1, self.input_len, 1)  # x_enc: [(bs * n_vars) x seq_len x 1]
        # x = self.value_embedding(feature_x)  # （B,T,N,d） Xdata
        x = self.value_embedding(origin_x[:, :, :self.feature_dim])
        if self.add_time_in_day:
            x += self.daytime_embedding[(origin_x[:, :, self.feature_dim] *self.minute_size).type(torch.LongTensor)]
        if self.add_day_in_week:
            x += self.weekday_embedding[(origin_x[:, :, self.feature_dim + 1] *self.weekday_size).type(torch.LongTensor) ]# 输入是的B*T*N的张量
        x = self.dropout(x)
        return x

# class TokenEmbedding(nn.Module):
#     def __init__(self, input_dim, embed_dim, norm_layer=None):
#         super().__init__()
#         self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
#         self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
#
#     def forward(self, x):
#         x = self.token_embed(x)
#         x = self.norm(x)
#         return x


# class PositionalEncoding(nn.Module):
#     def __init__(self, embed_dim, max_len=100):
#         # 这个地方可以不写类名
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, embed_dim).float()  # (M,E)
#         # 确保在反向传播时不会计算这个矩阵的梯度，因为位置编码是固定的，不需要学习。
#         pe.require_grad = False
#
#         position = torch.arange(0, max_len).float().unsqueeze(1)  # (M,1)
#         div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()  # (E//2,)
#
#         pe[:, 0::2] = torch.sin(position * div_term)  # 填充偶数列 (M,E//2)*(E//2,)
#         pe[:, 1::2] = torch.cos(position * div_term)  # 填充奇数列 (M,E//2)*(E//2,)
#
#         pe = pe.unsqueeze(0)  # (1,M,E)
#         self.register_buffer('pe', pe)  # pe 注册为不需要梯度的缓冲区 不计算梯度
#
#     def forward(self, x):
#         return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


# class DataEmbedding(nn.Module):
#     def __init__(self, feature_dim, embed_dim, drop_out=0.1, add_time_in_day=True, add_day_in_week=True,):
#         super().__init__()
#
#         self.add_time_in_day = add_time_in_day
#         self.add_day_in_week = add_day_in_week
#         self.minute_size = 288
#         self.weekday_size = 7
#         self.embed_dim = embed_dim
#         self.feature_dim = feature_dim
#         self.value_embedding = TokenEmbedding(feature_dim, embed_dim)  # 创建原始数据的转换层
#         #
#         # self.position_encoding = PositionalEncoding(embed_dim)  # 创建位置嵌入层
#         if self.add_time_in_day:
#             self.daytime_embedding = nn.init.xavier_uniform_(nn.Parameter(torch.empty(self.minute_size, embed_dim)))  # 一天中不同分钟索引（0-1440）映射到embed_dim维的向量中
#         if self.add_day_in_week:
#             self.weekday_embedding = nn.init.xavier_uniform_(nn.Parameter(torch.empty(self.weekday_size, embed_dim)))
#
#         self.dropout = nn.Dropout(drop_out)
#
#     def forward(self, x):
#         origin_x = x
#         x = self.value_embedding(origin_x[ :, :, :self.feature_dim])  # （B,T,N,d） Xdata
#         if self.add_time_in_day:
#             x += self.daytime_embedding[(origin_x[:, :, self.feature_dim] *self.minute_size).type(torch.LongTensor)]
#         if self.add_day_in_week:
#             x += self.weekday_embedding[(origin_x[:, :, self.feature_dim + 1]*self.weekday_size).type(torch.LongTensor) ]
#         x = self.dropout(x)
#         return x