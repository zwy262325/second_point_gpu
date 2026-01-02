import numpy as np
import torch
import math




def masked_data(sample, masking_ratio, lm, positive_nums,distribution):
    """Masked time series in time dimension"""
    # 原始代码输入数据sampleBTD(2,48,7) 现输入代码BNTD(8,207,72,96)

    # 1.新增：维度转换(B, N, T, D) -> (B*N, T, D)
    B_orig, N_orig, T, D = sample.shape
    sample_3d = sample.reshape(B_orig * N_orig, T, D)

    sample_3d = sample_3d.permute(0, 2, 1)
    # 对原始序列进行复制,生成 positive_nums 个副本,sample_repeat(6,7,48)
    sample_3d_repeat = sample_3d.repeat(positive_nums, 1, 1)
    # 为每个副本生成随机掩蔽（基于几何分布或随机掩蔽）mask(6,7,48)
    mask_index = noise_mask(sample_3d_repeat, masking_ratio, lm, distribution,positive_nums,
                            device=sample_3d_repeat.device)
    # 应用掩蔽：将掩蔽位置的值置为 0
    mask_index = mask_index.to('cuda:0')
    x_masked = mask_index * sample_3d_repeat

    # 2.新增：原始序列和掩蔽序列拼接  # batch_x_om_3d(6624,96,72) B * (positive_num + 1)DT, mask_om_3d(6624,96,72)
    mask_o = torch.ones(size=sample_3d.shape, device=sample.device, dtype=sample.dtype)
    batch_x_om_3d = torch.cat([sample_3d, x_masked], dim=0)
    mask_om_3d = torch.cat([mask_o, mask_index], dim=0)

    return batch_x_om_3d.permute(0, 2, 1), mask_om_3d.permute(0, 2, 1)


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


def noise_mask(X, masking_ratio, lm, distribution, positive_nums, exclude_feats=None, device=None):
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

    is_tensor = isinstance(X, torch.Tensor)
    device = X.device if is_tensor else torch.device('cpu')

    # ... (其他 distribution 逻辑不变) ...

    if distribution == 'geometric':  # stateful (Markov chain)
        # 仅在 X 是 PyTorch Tensor 时，使用 GPU 加速版本
        if is_tensor and device.type != 'cpu':
            L = X.shape[0] * X.shape[1] * X.shape[2]  # 总元素数
            mask_1d = torch_geom_noise_mask_single(L, lm, masking_ratio, device=device)
            mask = mask_1d.reshape(X.shape)
        else:
            # CPU/NumPy fallback for compatibility
            # 这里是原有的 CPU 密集型调用
            mask_1d = geom_noise_mask_single(X.shape[0] * X.shape[1] * X.shape[2], lm, masking_ratio)
            mask_1d = mask_1d.reshape(X.shape[0], X.shape[1], X.shape[2])
            mask = mask_1d
    # elif distribution == 'Mix':
    #     B_total, C, T = X.shape
    #     B_per = B_total // positive_nums
    #     mask = torch.ones_like(X, dtype=torch.bool)  # 默认全1（保留）
    #     # 第一个副本：随机掩蔽
    #     mask_random = (torch.rand(B_per, C, T, device=device) < masking_ratio)
    #     mask[:B_per] = mask_random
    #     # 第二个副本：几何掩蔽（连续块，原有逻辑）
    #     L_geo = B_per * (positive_nums -1) * C * T
    #     if device.type != 'cpu' and is_tensor:
    #         mask_geo_1d = torch_geom_noise_mask_single(L_geo, lm, masking_ratio, device=device)
    #     else:
    #         mask_geo_1d = geom_noise_mask_single(L_geo, lm, masking_ratio)
    #         mask_geo_1d = torch.from_numpy(mask_geo_1d).to(device)
    #     mask_geo = mask_geo_1d.reshape(B_per* (positive_nums -1), C, T).bool()
    #     mask[B_per:] = mask_geo
    # elif distribution == 'Mix': # 全部为随机掩蔽
    #     B_total, C, T = X.shape
    #     B_per = B_total // positive_nums
    #     mask = torch.ones_like(X, dtype=torch.bool)  # 默认全1（保留）
    #     mask_random = (torch.rand(B_per, C, T, device=device) < masking_ratio)
    #     mask[:B_per] = mask_random
    #     B_remaining = B_total - B_per
    #     mask_random_remaining = (torch.rand(B_remaining, C, T, device=device) < masking_ratio)
    #     mask[B_per:] = mask_random_remaining
    elif distribution == 'Mix':   # 全部为几何掩蔽
        B_total, C, T = X.shape
        L_geo_total = B_total * C * T
        if device.type != 'cpu' and is_tensor:
            mask_geo_1d_total = torch_geom_noise_mask_single(L_geo_total, lm, masking_ratio, device=device)
        else:
            mask_geo_1d_total = geom_noise_mask_single(L_geo_total, lm, masking_ratio)
            mask_geo_1d_total = torch.from_numpy(mask_geo_1d_total).to(device)
        mask_geo_total = mask_geo_1d_total.reshape(B_total, C, T).bool()
        mask = mask_geo_total
        return mask
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
    if not is_tensor:
        return torch.tensor(mask).to(device)
    else:
        # 如果 X 本身就是 Tensor，且我们已在 GPU 上生成了 mask，直接返回
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
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > pertub_ratio  # maskout_ratio are False
    mask = mask.to(x.device)
    return x * mask


def add_frequency(x, pertub_ratio=0.0):
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1 - pertub_ratio)  # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape) * (max_amplitude * 0.1)
    pertub_matrix = mask * random_am
    return x + pertub_matrix


def generate_binomial_mask(B, T, D, p=0.5):  # p is the ratio of not zero
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)


def masking(x, keepratio=0.9, mask='binomial'):
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


import torch
import math


def torch_geom_noise_mask_single(L: int, lm: float, masking_ratio: float, device: torch.device) -> torch.Tensor:
    """
    使用 PyTorch 张量操作并行生成长度为 L 的几何分布掩码。
    （GPU 加速版本，已增加鲁棒性）
    """
    if L == 0:
        return torch.empty(0, dtype=torch.bool, device=device)

    # 1. 计算几何分布的成功概率 P (即一个段结束的概率)
    p_m = 1.0 / lm

    # 鲁棒性：防止 1 - masking_ratio = 0
    if 1.0 - masking_ratio < 1e-6:
        p_u = 1.0
    else:
        p_u = p_m * masking_ratio / (1.0 - masking_ratio)

    # 鲁棒性：防止 p_m/p_u >= 1.0 导致 lambda <= 0
    p_m = min(p_m, 0.99999)
    p_u = min(p_u, 0.99999)

    # 确保 lambda > 0，log(lambda) 不为 -Inf
    lambda_m = torch.tensor(1.0 - p_m, device=device)
    lambda_u = torch.tensor(1.0 - p_u, device=device)

    # 2. 估计需要生成的段数：保持不变
    avg_segment_length = (lm * (1 - masking_ratio)) + ((1 - masking_ratio) / p_u * masking_ratio) if p_u > 1e-6 else L
    safe_segment_count = int(L / avg_segment_length) * 2 + 10
    safe_segment_count = max(safe_segment_count, 1000)

    # 3. 逆变换采样生成掩蔽段 (0s) 长度 K_m
    EPS = 1e-6
    # 鲁棒性：确保 U 严格大于 EPS，防止 log(U) 出现 -Inf
    u_m = torch.rand(safe_segment_count, device=device).clamp(min=EPS)
    # 鲁棒性：确保段长度至少为 1
    k_m = torch.ceil(torch.log(u_m) / torch.log(lambda_m)).long().clamp(min=1)

    # 4. 逆变换采样生成非掩蔽段 (1s) 长度 K_u
    u_u = torch.rand(safe_segment_count, device=device).clamp(min=EPS)
    k_u = torch.ceil(torch.log(u_u) / torch.log(lambda_u)).long().clamp(min=1)

    # 5. 确定起始状态并交错合并段 (保持不变)
    if torch.rand(1, device=device) > masking_ratio:
        segments_lengths = torch.empty(2 * safe_segment_count, dtype=k_m.dtype, device=device)
        segments_lengths[0::2] = k_u
        segments_lengths[1::2] = k_m
        segments_lengths = segments_lengths[1:]
        segment_values = torch.tensor([1, 0], device=device).repeat(segments_lengths.shape[0] // 2 + 1)[
                         :segments_lengths.shape[0]]
    else:
        segments_lengths = torch.empty(2 * safe_segment_count, dtype=k_m.dtype, device=device)
        segments_lengths[0::2] = k_m
        segments_lengths[1::2] = k_u
        segment_values = torch.tensor([0, 1], device=device).repeat(segments_lengths.shape[0] // 2 + 1)[
                         :segments_lengths.shape[0]]

    # 6. 鲁棒性：限制单个 segment 长度，防止 SymInt 溢出
    # MAX_SAFE_LENGTH 是防止溢出的关键，设置一个大于 L 的安全值。
    MAX_SAFE_LENGTH = L * 2 + 10
    segments_lengths = segments_lengths.clamp(max=MAX_SAFE_LENGTH)

    full_mask_1d = torch.repeat_interleave(segment_values, segments_lengths)

    # 7. 截断到所需长度 L
    final_mask_1d = full_mask_1d[:L]

    return final_mask_1d.bool()
