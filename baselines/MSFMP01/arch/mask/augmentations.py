import numpy as np
import torch
import math

from numpy.lib.twodim_base import mask_indices


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
    # mask_index = mask_index.to('cuda:0')
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



def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (
            1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def noise_mask(X, masking_ratio=0.25, lm=3, distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)
    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        """
           优化后：对每个样本独立生成掩码
           X: 形状 (M, D, T)，其中 M = positive_nums * B * N
           """
        M, D, T = X.shape
        sample_len = D * T
        mask = np.zeros((M, D, T), dtype=bool)

        # 对每个样本独立生成掩码
        for m in range(M):
            # 为第m个样本生成独立的掩码
            single_mask = geom_noise_mask_single(sample_len, lm, masking_ratio)
            mask[m] = single_mask.reshape(D, T)

        # mask = geom_noise_mask_single(X.shape[0] * X.shape[1] * X.shape[2], lm, masking_ratio)
        # mask = mask.reshape(X.shape[0], X.shape[1], X.shape[2])
    elif distribution == 'masked_tail':
        mask = np.ones(X.shape, dtype=bool)
        for m in range(X.shape[0]):  # feature dimension

            keep_mask = np.zeros_like(mask[m, :], dtype=bool)
            n = math.ceil(keep_mask.shape[1] * (1 - masking_ratio))
            keep_mask[:, :n] = True
            mask[m, :] = keep_mask  # time dimension
    elif distribution == 'masked_head':
        mask = np.ones(X.shape, dtype=bool)
        for m in range(X.shape[0]):  # feature dimension

            keep_mask = np.zeros_like(mask[m, :], dtype=bool)
            n = math.ceil(keep_mask.shape[1] * masking_ratio)
            keep_mask[:, n:] = True
            mask[m, :] = keep_mask  # time dimension
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                p=(1 - masking_ratio, masking_ratio))
    return torch.tensor(mask)


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

