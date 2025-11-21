import torch
import torch.nn as nn


class Adj(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(Adj, self).__init__()
        self.weights = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        self.init_param()

    def init_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x, supports):
        B, N, T, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * T, N, D)
        weights_transformed = self.weights
        supports1 = supports[0].to('cuda:0')
        x_g = torch.einsum("nm,bmc->bnc", supports1, x)
        x_gconv = torch.einsum('bnc,ci->bni', x_g, weights_transformed) + self.bias
        x_gconv = x_gconv.reshape(B, T, N, D).permute(0, 2, 1, 3)
        return x_gconv

