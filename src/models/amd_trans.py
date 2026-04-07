import torch
import torch.nn as nn
from src.layers.Transformer_EncDec import Encoder, EncoderLayer
from src.layers.SelfAttention_Family import FullAttention, AttentionLayer
from src.layers.Embed import PatchEmbedding
from src.norm import Normalize


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [B x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class HybridLoss(nn.Module):
    """
    Hybrid Loss = alpha * MSE + beta * Directional Loss

    MSE minimizes numerical error; DirectionalLoss penalizes
    predictions that move in the wrong direction vs. the target.
    """
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred, true):
        mse_loss = self.mse(pred, true)
        # Differentiable directional penalty: relu(-pred_diff * true_diff)
        # is > 0 only when signs disagree
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
        true_diff = true[:, 1:, :] - true[:, :-1, :]
        dir_loss = torch.mean(torch.relu(-pred_diff * true_diff))
        return self.alpha * mse_loss + self.beta * dir_loss


class Model(nn.Module):
    """
    Global-Aware AMD-Trans: Adaptive Multi-Dimensional Transformer for Asset Pricing

    Architecture (follows readme.md spec):
        1. RevIN  — per-variate reversible instance normalization
        2. Temporal Patch Attention  — channel-independent patch-wise self-attention
        3. Cross-Attention  — local patch tokens attend to global temporal context (x_mark)
        4. Variate-wise Attention  — cross-channel relationship learning
        5. FlattenHead projection + RevIN denormalization
    """

    def __init__(self, configs, patch_len=16, stride=8, structure='rev-1-2-3'):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.structure = structure
        padding = stride

        # ------------------------------------------------------------------
        # Stage 0: RevIN (Reversible Instance Normalization)
        # ------------------------------------------------------------------
        self.revin = Normalize(configs.enc_in, affine=True)

        # ------------------------------------------------------------------
        # Stage 1: Temporal Patch Embedding + Self-Attention
        #          Channel-independent: each variate is treated separately
        # ------------------------------------------------------------------
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)

        self.temporal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        # ------------------------------------------------------------------
        # Stage 2: Cross-Attention — local patch tokens (Q) attend to
        #          global temporal context derived from x_mark (K, V)
        #          x_mark encodes (month, day) → 2 features → project to d_model
        # ------------------------------------------------------------------
        self.global_proj = nn.Linear(2, configs.d_model)
        self.cross_attn = AttentionLayer(
            FullAttention(False, configs.factor,
                          attention_dropout=configs.dropout,
                          output_attention=False),
            configs.d_model, configs.n_heads
        )
        self.cross_norm = nn.LayerNorm(configs.d_model)
        self.cross_dropout = nn.Dropout(configs.dropout)

        # ------------------------------------------------------------------
        # Stage 3: Variate-wise Self-Attention — cross-channel interaction
        # ------------------------------------------------------------------
        self.variate_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # ------------------------------------------------------------------
        # Output Head
        # ------------------------------------------------------------------
        head_nf = configs.d_model * self.patch_num
        self.head = FlattenHead(configs.enc_in, head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast_notused(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, T, N = x_enc.shape

        # Stage 0: RevIN normalization
        x_enc = self.revin(x_enc, 'norm')

        # Stage 1: Temporal patch attention (channel-independent)
        # [B, T, N] -> [B, N, T] -> patch_embedding -> [B*N, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc.permute(0, 2, 1))
        enc_out, _ = self.temporal_encoder(enc_out)  # [B*N, patch_num, d_model]

        # Stage 2: Cross-attention with global temporal context
        # x_mark: [B, T, 2] -> [B, T, d_model] -> [B*N, T, d_model]
        global_ctx = self.global_proj(x_mark_enc.float())           # [B, T, d_model]
        global_ctx = (global_ctx
                      .unsqueeze(1)
                      .expand(-1, n_vars, -1, -1)
                      .reshape(B * n_vars, T, -1))                  # [B*N, T, d_model]

        cross_out, _ = self.cross_attn(enc_out, global_ctx, global_ctx, attn_mask=None)
        enc_out = self.cross_norm(enc_out + self.cross_dropout(cross_out))

        # Stage 3: Variate-wise attention
        # Reshape: [B*N, patch_num, d_model] -> [B, N, patch_num, d_model]
        P = enc_out.shape[-2]
        enc_out = enc_out.reshape(B, n_vars, P, enc_out.shape[-1])

        # Per-variate summary token (mean over patches): [B, N, d_model]
        variate_tokens = enc_out.mean(dim=2)
        variate_out, _ = self.variate_encoder(variate_tokens)       # [B, N, d_model]

        # Broadcast variate context back and add as residual
        enc_out = enc_out + variate_out.unsqueeze(2)                # [B, N, patch_num, d_model]

        # Output: [B, N, d_model, patch_num] -> FlattenHead -> [B, N, pred_len]
        dec_out = self.head(enc_out.permute(0, 1, 3, 2))           # [B, N, pred_len]
        dec_out = dec_out.permute(0, 2, 1)                         # [B, pred_len, N]

        # Stage 0: RevIN denormalization
        dec_out = self.revin(dec_out, 'denorm')
        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        select which stages to execute for ablation:
            - 'rev-1-2-3': full model with all stages
            - '1': only temporal patch attention
            - '1-2': temporal patch attention + cross-attention
            - '1-3': temporal patch attention + variate-wise attention
            - '1-2-3': all stages except RevIN
            - 'rev-1': RevIN + temporal patch attention only (no cross-attention or variate-wise attention)
            - 'rev-1-2': RevIN + temporal patch attention + cross-attention
            - 'rev-1-3': RevIN + temporal patch attention + variate-wise attention
        :param x_enc:
        :param x_mark_enc:
        :param x_dec:
        :param x_mark_dec:
        :param structure_selection:
        :return:
        """
        structure = self.structure

        B, T, N = x_enc.shape

        # Stage 0: RevIN normalization
        if 'rev' in structure:
            x_enc = self.revin(x_enc, 'norm')
        else:
            x_enc = x_enc

        # Stage 1: Temporal patch attention (channel-independent)
        # [B, T, N] -> [B, N, T] -> patch_embedding -> [B*N, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc.permute(0, 2, 1))
        enc_out, _ = self.temporal_encoder(enc_out)  # [B*N, patch_num, d_model]
        if '2' in structure:
            # Stage 2: Cross-attention with global temporal context
            # x_mark: [B, T, 2] -> [B, T, d_model] -> [B*N, T, d_model]
            global_ctx = self.global_proj(x_mark_enc.float())           # [B, T, d_model]
            global_ctx = (global_ctx
                          .unsqueeze(1)
                          .expand(-1, n_vars, -1, -1)
                          .reshape(B * n_vars, T, -1))                  # [B*N, T, d_model]

            cross_out, _ = self.cross_attn(enc_out, global_ctx, global_ctx, attn_mask=None)
            enc_out = self.cross_norm(enc_out + self.cross_dropout(cross_out))

        if '3' in structure:
            # Stage 3: Variate-wise attention
            # Reshape: [B*N, patch_num, d_model] -> [B, N, patch_num, d_model]
            P = enc_out.shape[-2]
            enc_out = enc_out.reshape(B, n_vars, P, enc_out.shape[-1])

            # Per-variate summary token (mean over patches): [B, N, d_model]
            variate_tokens = enc_out.mean(dim=2)
            variate_out, _ = self.variate_encoder(variate_tokens)       # [B, N, d_model]
            # Broadcast variate context back and add as residual
            enc_out = enc_out + variate_out.unsqueeze(2)                # [B, N, patch_num, d_model]

        # Output: [B, N, d_model, patch_num] -> FlattenHead -> [B, N, pred_len]
        dec_out = self.head(enc_out.permute(0, 1, 3, 2))           # [B, N, pred_len]
        dec_out = dec_out.permute(0, 2, 1)                         # [B, pred_len, N]

        # Stage 0: RevIN denormalization
        if 'rev' in structure:
            dec_out = self.revin(dec_out, 'denorm')

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, N]
        return None