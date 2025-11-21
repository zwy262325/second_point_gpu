import torch
import torch.nn.functional as F
import torch.nn as nn


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, agcn_embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        # self.weights_pool = nn.Parameter(torch.FloatTensor(agcn_embed_dim,  dim_in, dim_out))
        # self.bias_pool = nn.Parameter(torch.FloatTensor(agcn_embed_dim, dim_out))
        self.weights = torch.nn.Parameter(torch.FloatTensor(self.cheb_k,dim_in, dim_out))
        self.bias = torch.nn.Parameter(torch.FloatTensor(dim_out))
        self.init_param()


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
        # support_set = [torch.eye(node_num).to(supports.device), supports]
        # for k in range(2, self.cheb_k):
        #     support_set.append(torch.matmul(
        #         2 * supports, support_set[-1]) - support_set[-2])
        # supports = torch.stack(support_set, dim=0)
        # weights = torch.einsum(
        #     'nd,dkio->nkio', node_embeddings, self.weights_pool)
        # bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        # x_g = torch.einsum("knm,bmc->bknc", supports,
        #                    x)  # B, cheb_k, N, dim_in
        # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_gconv = torch.einsum('bnki,nkio->bno', x_g,
        #                        weights)+bias  # b, N, dim_out
        # x_gconv= x_gconv.reshape(B, T, N, D).permute(0, 2, 1, 3)

       #去掉单位矩阵
        # weights = torch.einsum(
        #     'nd,dio->nio', node_embeddings, self.weights_pool)
        # bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        # x_g = torch.einsum("nm,bmc->bnc", supports,
        #                    x)  # B, cheb_k, N, dim_in
        # x_gconv = torch.einsum('bni,nio->bno', x_g,
        #                        weights) + bias  # b, N, dim_out
        # x_gconv = x_gconv.reshape(B, T, N, D).permute(0, 2, 1, 3)


        #去掉可学习的
        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(
                2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        x_g = torch.einsum("knm,bmc->bknc", supports,
                           x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,kio->bno', x_g,
                               self.weights)+self.bias  # b, N, dim_out
        x_gconv= x_gconv.reshape(B, T, N, D).permute(0, 2, 1, 3)
        return x_gconv
