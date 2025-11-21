import torch
import torch.nn.functional as F
import torch.nn as nn


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, agcn_embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.dim_in = dim_in
        self.weights_pool = nn.Parameter(torch.FloatTensor(agcn_embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(agcn_embed_dim, dim_out))
        self.init_param()
        self.conv_layer = nn.Conv2d(in_channels=cheb_k * dim_in, out_channels=dim_out, kernel_size=1)


    def init_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x, node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        B, N, T, D =x.shape
        x = x.permute(0, 2, 1, 3).reshape(B*T, N, D)
        node_num = node_embeddings.shape[0]
        supports = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(
                2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 3, 2, 1).reshape(B, T, N, self.cheb_k * self.dim_in).permute(0, 3, 1, 2)  # B, N, cheb_k, dim_in
        x_g = self.conv_layer(x_g)
        x_gconv= x_g.permute(0, 3, 2, 1)
        return x_gconv
