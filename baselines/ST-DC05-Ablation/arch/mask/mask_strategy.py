import torch

def spatial_random_masking(x, mask_ratio):
    """
    Perform per-node, per-sample random masking in the spatial (node) dimension.
    x: [B, N, T, D], sequence with batch size B, number of nodes N, time steps T, and feature dimension D
    """
    B, N, T, D = x.shape  # batch size, number of nodes, time steps, feature dimension
    num_keep = int(N * (1 - mask_ratio))  # number of nodes to keep

    # Generate random noise for each node in each sample
    noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

    # Sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove along node dimension
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # to restore the original order later

    # Keep the first subset of nodes for each sample
    ids_keep = ids_shuffle[:, :num_keep]
    ids_nokeep = ids_shuffle[:, num_keep:]

    # Gather the kept nodes for each sample
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D))

    return x_masked, ids_nokeep, ids_restore



def temporal_random_masking(x, mask_ratio):
    """
    Perform per-time-step, per-sample random masking in the temporal dimension.
    x: [B , N, T, D], sequence with batch size B, number of nodes N, time steps T, and feature dimension D
    """
    B, N, T, D = x.shape  # batch size, number of nodes, time steps, feature dimension
    num_keep = int(T * (1 - mask_ratio))  # number of time steps to keep

    # Generate random noise for each time step in each sample
    noise = torch.rand(B, T, device=x.device)  # noise in [0, 1]

    # Sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove along time dimension
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # to restore the original order later

    # Keep the first subset of time steps for each sample
    ids_keep = ids_shuffle[:, :num_keep]
    ids_nokeep = ids_shuffle[:, num_keep:]

    # Gather the kept time steps for each sample
    x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).unsqueeze(-1).transpose(1, 2).repeat(1, N, 1, D))

    return x_masked,ids_nokeep, ids_restore
