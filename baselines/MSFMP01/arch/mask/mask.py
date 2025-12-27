import torch
from torch import nn
from ..layers.Transformer_EncDec import Encoder, EncoderLayer
from ..layers.SelfAttention_Family import DSAttention, AttentionLayer
from .augmentations import masked_data
from .transformer_layers_new import TransformerLayers
from .tools import ContrastiveWeight, AggregationRebuild, DataEmbedding


class Pooler_Head(nn.Module):
    def __init__(self, seq_len, d_model, compression_ratio,head_dropout=0):
        super().__init__()

        pn = seq_len * d_model
        hidden_dim = int(pn * compression_ratio)
        dimension = 32
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dimension),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):  # [(bs * n_vars) x seq_len x d_model]
        x = self.pooler(x) # [(bs * n_vars) x dimension]
        return x

class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len*d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x seq_len x d_model]
        x = self.flatten(x) # [bs x n_vars x (seq_len * d_model)]
        x = self.linear(x) # [bs x n_vars x seq_len]
        x = self.dropout(x) # [bs x n_vars x seq_len]
        return x

class Mask(nn.Module):

    def __init__(self,  embed_dim, num_heads, mlp_ratio, dropout, mask_ratio, encoder_depth, input_len,mask_distribution, lm, positive_nums, temperature, compression_ratio, attention_configs, mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        # 初始化augmentations掩蔽参数、
        self.mask_distribution = mask_distribution  # 掩蔽分布类型
        self.lm = lm  # 几何分布的平均掩蔽长度（仅当distribution='geometric'时使用）
        self.positive_nums = positive_nums # 数据的副本数量
        self.masked_data = masked_data
        # Embedding
        self.enc_embedding = DataEmbedding(1, 32, input_len)
        # encoder_new
        #self.encoder_new = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        # encoder_original
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, attention_configs.factor, attention_dropout=attention_configs.dropout,
                                    output_attention=attention_configs.output_attention), attention_configs.d_model, attention_configs.n_heads),
                    attention_configs.d_model,
                    attention_configs.d_ff,
                    dropout=attention_configs.dropout,
                    activation=attention_configs.activation
                ) for l in range(attention_configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(attention_configs.d_model),
        )
        # for series-wise representation
        self.pooler = Pooler_Head(input_len, embed_dim, compression_ratio,head_dropout=dropout)
        self.contrastive = ContrastiveWeight(temperature,positive_nums)
        self.aggregation = AggregationRebuild(temperature, positive_nums)
        self.projection = Flatten_Head(input_len, embed_dim, input_len, head_dropout=dropout)

    def encoding_decoding(self, long_term_history):

        long_term_history = long_term_history.permute(0, 2, 1, 3)
        x_enc, mask_index = self.masked_data(long_term_history, self.mask_ratio, self.lm, self.positive_nums, distribution='geometric')
        batch_size, num_nodes,_, _ = long_term_history.shape
        enc_out = self.enc_embedding(x_enc)
        #p_enc_out = self.encoder_new(enc_out)
        p_enc_out, _= self.encoder(enc_out)
        s_enc_out = self.pooler(p_enc_out)
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out)
        rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out)
        _, seq_len, nums_dim = agg_enc_out.shape
        agg_enc_out = agg_enc_out.reshape(batch_size * (self.positive_nums + 1), num_nodes, seq_len, nums_dim)
        dec_out = self.projection(agg_enc_out)
        dec_out = dec_out.view(batch_size * (self.positive_nums + 1), num_nodes, 1, -1)
        dec_out = dec_out[:batch_size]
        dec_out = dec_out.permute(0, 1, 3, 2)
        return dec_out, long_term_history[:, :, :, 0:1], loss_cl

    def encoding(self, long_term_history):
        long_term_history = long_term_history.permute(0, 2, 1, 3)
        batch_size, num_nodes, seq_len, _ = long_term_history.shape
        x_enc = long_term_history.reshape(batch_size * num_nodes, long_term_history.shape[2], long_term_history.shape[3])  # [(bs * n_vars) x seq_len x d_model]
        enc_out = self.enc_embedding(x_enc)
        # p_enc_out = self.encoder_new(enc_out)
        p_enc_out, _ = self.encoder(enc_out)
        _, _, nums_dim = p_enc_out.shape
        p_enc_out = p_enc_out.reshape(batch_size, num_nodes, seq_len, nums_dim)

        return p_enc_out

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        if self.mode == "pre-train":
            predict_result, original_result, loss_cl = self.encoding_decoding(history_data)
            return predict_result, original_result, loss_cl
        else:
            hidden_states_full = self.encoding(history_data)
            return hidden_states_full
