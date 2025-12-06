
import torch
import torch.nn as nn
import torch.nn.functional as F

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