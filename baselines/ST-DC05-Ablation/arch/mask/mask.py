import torch
from torch import nn

from .patch import PatchEmbedding
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers
from .mask_strategy import spatial_random_masking
from .mask_strategy import temporal_random_masking
from .agcrn1 import AVWGCN
from .adj import Adj


def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


class Mask(nn.Module):

    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, mask_ratio, encoder_depth,
                 decoder_depth, dim_in, dim_out, agcn_embed_dim, cheb_k, num_node,supports,spatial=False, mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.spatial = spatial
        self.selected_feature = 0
        self.supports = supports

        self.node_embeddings = nn.Parameter(torch.randn(num_node, agcn_embed_dim), requires_grad=True)
        self.AVWGCN = AVWGCN(dim_in, dim_out, cheb_k, agcn_embed_dim)
        self.Adj = Adj(dim_in,dim_out)

        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.pos_mat = None
        # encoder specifics
        # # patchify & embedding
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        # # positional encoding
        self.positional_encoding = PositionalEncoding()

        # encoder
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

        # decoder specifics
        # transform layer
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        # # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # # decoder
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)

        # # prediction (reconstruction) layer
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()

    def initialize_weights(self, timm=None):
        # import here to fix bugs related to set visible device
        from timm.models.vision_transformer import trunc_normal_
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history, mask=True):

        mid_patches = self.patch_embedding(long_term_history)  # B, N, d, P
        mid_patches = mid_patches.transpose(-1, -2)  # B, N, P, d
        # patchify and embed input
        if mask:
            batch_size, num_nodes, num_time, num_dim = mid_patches.shape
            agcrn_hidden_states = self.AVWGCN(mid_patches,self.node_embeddings)
            patches = self.positional_encoding(agcrn_hidden_states)
            if self.spatial:
                x, ids_nokeep, ids_restore = spatial_random_masking(patches, self.mask_ratio)
                encoder_input = x.transpose(-2, -3)
                hidden_states_unmasked = self.encoder(encoder_input)
                hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_time, -1, self.embed_dim)
            if not self.spatial:
                x, ids_nokeep, ids_restore = temporal_random_masking(patches, self.mask_ratio)
                encoder_input = x
                hidden_states_unmasked = self.encoder(encoder_input)
                hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1,self.embed_dim)
            return hidden_states_unmasked, ids_nokeep, ids_restore, patches.shape[2], patches.shape[1]

        else:
            batch_size, num_nodes, _, _ = long_term_history.shape
            patches = self.positional_encoding(mid_patches)  # B, N, P, d
            encoder_input = patches  # B, N, P, d
            if self.spatial:
                encoder_input = encoder_input.transpose(-2, -3)  # B,  P,N, d
            hidden_states_unmasked = self.encoder(encoder_input)  # B,  P,N, d/# B, N, P, d
            if self.spatial:
                hidden_states_unmasked = hidden_states_unmasked.transpose(-2, -3)  # B, N, P, d
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1,
                                                                                    self.embed_dim)  # B, N, P, d
            return hidden_states_unmasked

    def decoding(self, hidden_states_unmasked, ids_restore, full_T, full_N):

        # encoder 2 decoder layer
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)  # B, N, P, d/# B,P, N,  d
        # B,N*r,P,d
        if self.spatial:
            # TO WORK SPATIAL:
            batch_size, num_time, num_nodes, _ = hidden_states_unmasked.shape  # B,T,N,D
            mask_token = self.mask_token.repeat(batch_size, num_time, full_N - num_nodes, 1)
            x_ = torch.cat([hidden_states_unmasked, mask_token], dim=2)  # no cls token
            ids_restore = ids_restore.unsqueeze(-1).unsqueeze(-1).transpose(1, 2).repeat(1, full_T, 1, x_.shape[3])
            x_ = torch.gather(x_, dim=2, index=ids_restore)
            hidden_states_full = self.positional_encoding(x_)

            # decoding
            hidden_states_full = self.decoder(hidden_states_full)  # B, P, N, d
            hidden_states_full = self.decoder_norm(hidden_states_full)  # B, P, N, d
            # prediction (reconstruction)
            reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_time, -1, self.embed_dim))  # B, P, N, L
        else:
            batch_size, num_nodes, num_time, _ = hidden_states_unmasked.shape
            mask_token = self.mask_token.repeat(batch_size, num_nodes, full_T - num_time, 1)
            x_ = torch.cat([hidden_states_unmasked, mask_token], dim=2)  # no cls token
            ids_restore = ids_restore.unsqueeze(-1).unsqueeze(-1).transpose(1, 2).repeat(1, full_N, 1, x_.shape[3])
            x_ = torch.gather(x_, dim=2, index=ids_restore)
            hidden_states_full = self.positional_encoding(x_)

            # decoding
            hidden_states_full = self.decoder(hidden_states_full)
            hidden_states_full = self.decoder_norm(hidden_states_full)

            # prediction (reconstruction)
            reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))

        return reconstruction_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, ids_nokeep, full_T, full_N):

        if self.spatial:
            batch_size, num_time, num_nodes, D_states = reconstruction_full.shape  # B, P, N, L
            label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :,self.selected_feature, :]  # BTND

            ids_nokeep = ids_nokeep.unsqueeze(-1).unsqueeze(-1).transpose(1, 2).repeat(1, full_T, 1, D_states)
            reconstruction_masked_tokens = torch.gather(reconstruction_full, dim=2, index=ids_nokeep)
            label_masked_tokens = torch.gather(label_full, dim=2, index=ids_nokeep)
            reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_time, -1)
            label_masked_tokens = label_masked_tokens.reshape(batch_size, num_time, -1)
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            batch_size, num_nodes, num_time, D_states = reconstruction_full.shape  # BTND
            label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :,self.selected_feature, :].transpose(1, 2)  # B, N, P, L

            ids_nokeep = ids_nokeep.unsqueeze(-1).unsqueeze(-1).transpose(1, 2).repeat(1, full_N, 1, D_states)
            reconstruction_masked_tokens = torch.gather(reconstruction_full, dim=2, index=ids_nokeep)
            label_masked_tokens = torch.gather(label_full, dim=2, index=ids_nokeep)

            reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1)
            label_masked_tokens = label_masked_tokens.reshape(batch_size, num_nodes, -1)
            return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None,
                epoch: int = None, **kwargs) -> torch.Tensor:
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)  # B, N, 1, L * P
        # feed forward
        if self.mode == "pre-train":
            # encoding
            hidden_states_unmasked, ids_nokeep, ids_restore, full_T, full_N = self.encoding(history_data)
            # decoding
            reconstruction_full = self.decoding(hidden_states_unmasked, ids_restore, full_T, full_N)
            # for subsequent loss computing
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, ids_nokeep, full_T, full_N)

            return reconstruction_masked_tokens.unsqueeze(-1), label_masked_tokens.unsqueeze(-1)
        else:
            hidden_states_full = self.encoding(history_data, mask=False)
            return hidden_states_full
