import numpy as np
import torch
import math



def masked_data(sample, masking_ratio, lm, positive_nums=1, distribution='geometric'):
    """Masked time series in time dimension"""
    # 原始代码输入数据sampleBTD(2,48,7) 现输入代码BNTD(8,207,72,96)

    # 1.新增：维度转换(B, N, T, D) -> (B*N, T, D)
    B_orig, N_orig, T, D = sample.shape
    sample_3d = sample.reshape(B_orig * N_orig, T, D)

    sample_3d = sample_3d.permute(0, 2, 1)
    # 对原始序列进行复制,生成 positive_nums 个副本,sample_repeat(6,7,48)
    sample_3d_repeat = sample_3d.repeat(positive_nums, 1, 1)
    # 为每个副本生成随机掩蔽（基于几何分布或随机掩蔽）mask(6,7,48)
    mask_index = noise_mask(sample_3d_repeat, masking_ratio, lm, distribution=distribution)
    mask_index = mask_index.to('cuda:0')
    # 应用掩蔽：将掩蔽位置的值置为 0
    x_masked = mask_index * sample_3d_repeat

    # 2.新增：原始序列和掩蔽序列拼接  # batch_x_om_3d(6624,96,72) B * (positive_num + 1)DT, mask_om_3d(6624,96,72)
    mask_o = torch.ones(size=sample_3d.shape, device=sample.device, dtype=sample.dtype)
    batch_x_om_3d = torch.cat([sample_3d, x_masked], dim=0)
    mask_om_3d = torch.cat([mask_o, mask_index], dim=0)

    # 3.新增：还原输出
    # 新的总批次大小: (P+1) * B_orig
    B_total_new = B_orig * (positive_nums + 1)
    batch_x_om = batch_x_om_3d.reshape(B_total_new, N_orig, T, D)
    mask_om = mask_om_3d.reshape(B_total_new, N_orig, T, D)
    # batch_x_om(32,207,72,96), mask_om(32,207,72,96)
    return batch_x_om, mask_om
    # return batch_x_om_3d.permute(0,2,1).float(), mask_om_3d.permute(0,2,1).float()

import torch

def geom_noise_mask_vectorized_gpu(L, lm, masking_ratio, device, batch_size=1):
    """
    向量化GPU版本：无逐元素循环，批量生成多个样本的掩码
    Args:
        L: 单个样本的掩码长度
        lm: 几何分布平均长度
        masking_ratio: 掩码比例
        device: 执行设备
        batch_size: 一次性生成多少个样本的掩码（M）
    Returns:
        (batch_size, L) torch.bool: 批量掩码
    """
    p_m = 1.0 / lm
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    rand_init = torch.rand(batch_size, device=device)
    rand_switch = torch.rand(batch_size, L, device=device)
    init_state = (rand_init > masking_ratio).int()
    state = init_state.unsqueeze(1).repeat(1, L)
    p_m_tensor = torch.full((batch_size, L), p_m, device=device)
    p_u_tensor = torch.full((batch_size, L), p_u, device=device)
    p = torch.where(state == 0, p_m_tensor, p_u_tensor)
    switch = rand_switch < p
    switch_cumsum = torch.cumsum(switch.int(), dim=1)
    final_state = (state + switch_cumsum) % 2
    return final_state.bool()

# 升级noise_mask的geometric分支
def noise_mask(X, masking_ratio=0.25, lm=3, distribution='geometric', exclude_feats=None):
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)
    device = X.device
    M, D, T = X.shape
    sample_len = D * T
    if distribution == 'geometric':
        # 向量化GPU生成所有M个样本的掩码（无循环）
        batch_masks = geom_noise_mask_vectorized_gpu(
            L=sample_len,
            lm=lm,
            masking_ratio=masking_ratio,
            device=device,
            batch_size=M
        )
        mask = batch_masks.reshape(M, D, T)  # (M, D, T)
    return mask


def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b


def DataTransform(sample, config):
    """Weak and strong augmentations"""
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    # weak_aug = permutation(sample, max_segments=config.augmentation.max_seg)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug


def remove_frequency(x, pertub_ratio=0.0):
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > pertub_ratio # maskout_ratio are False
    mask = mask.to(x.device)
    return x*mask


def add_frequency(x, pertub_ratio=0.0):

    mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x+pertub_matrix


def generate_binomial_mask(B, T, D, p=0.5): # p is the ratio of not zero
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)


def masking(x, keepratio=0.9, mask= 'binomial'):
    global mask_id
    nan_mask = ~x.isnan().any(axis=-1)
    x[~nan_mask] = 0
    # x = self.input_fc(x)  # B x T x Ch

    if mask == 'binomial':
        mask_id = generate_binomial_mask(x.size(0), x.size(1), x.size(2), p=keepratio).to(x.device)
    # elif mask == 'continuous':
    #     mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
    # elif mask == 'all_true':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    # elif mask == 'all_false':
    #     mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
    # elif mask == 'mask_last':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    #     mask[:, -1] = False

    # mask &= nan_mask
    x[~mask_id] = 0
    return x

