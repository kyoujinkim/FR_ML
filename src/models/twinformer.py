"""
TwinFormer: A Dual-Level Transformer for Long-Sequence Time-Series Forecasting
Paper: arXiv:2512.12301

Hierarchical Transformer:
  1. Non-overlapping patching
  2. Local Informer (top-k sparse MHA + FFN) within each patch
  3. Mean-pool each patch
  4. Global Informer (top-k sparse MHA + FFN) across patches
  5. GRU aggregation across patch tokens
  6. Linear head on final hidden state -> pred_len
Channel-independent (PatchTST-style).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSparseAttention(nn.Module):
    """Multi-head attention keeping only top-k logits per query row."""
    def __init__(self, d_model, n_heads, top_k=5, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.top_k = top_k
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, _ = x.shape
        H, Dk = self.n_heads, self.d_k
        q = self.q_proj(x).view(B, N, H, Dk).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, Dk).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, Dk).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dk)  # [B,H,N,N]

        k_eff = min(self.top_k, N)
        if k_eff < N:
            topk_vals, _ = torch.topk(logits, k_eff, dim=-1)
            thresh = topk_vals[..., -1:].expand_as(logits)
            logits = torch.where(logits < thresh,
                                 torch.full_like(logits, float('-inf')),
                                 logits)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B,H,N,Dk]
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(out)


class InformerBlock(nn.Module):
    """MultiHead top-k sparse attention + FFN with residual/LayerNorm."""
    def __init__(self, d_model, n_heads, d_ff, top_k=5, dropout=0.1, activation='gelu'):
        super().__init__()
        self.attn = TopKSparseAttention(d_model, n_heads, top_k=top_k, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(x))
        x = x + self.dropout(self.ffn(self.norm1(x)))
        return self.norm2(x)


class Model(nn.Module):
    """
    TwinFormer for long-term forecasting.
    Expects configs with: task_name, seq_len, pred_len, enc_in, d_model, n_heads,
                          e_layers, d_ff, dropout, activation.
    Optional: patch_len (default 16), top_k (default 5).
    """
    def __init__(self, configs, patch_len=16, top_k=5):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.patch_len = patch_len
        self.top_k = top_k

        # pad so seq_len is divisible by patch_len
        self.pad_len = (patch_len - self.seq_len % patch_len) % patch_len
        self.num_patches = (self.seq_len + self.pad_len) // patch_len

        # Token embedding: project scalar time step to d_model (channel-independent)
        self.token_embed = nn.Linear(1, configs.d_model)
        self.pos_embed_local = nn.Parameter(torch.zeros(1, patch_len, configs.d_model))
        self.pos_embed_global = nn.Parameter(torch.zeros(1, self.num_patches, configs.d_model))
        nn.init.trunc_normal_(self.pos_embed_local, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_global, std=0.02)

        self.local_blocks = nn.ModuleList([
            InformerBlock(configs.d_model, configs.n_heads, configs.d_ff,
                          top_k=top_k, dropout=configs.dropout, activation=configs.activation)
            for _ in range(configs.e_layers)
        ])
        self.global_blocks = nn.ModuleList([
            InformerBlock(configs.d_model, configs.n_heads, configs.d_ff,
                          top_k=top_k, dropout=configs.dropout, activation=configs.activation)
            for _ in range(configs.e_layers)
        ])

        self.gru = nn.GRU(input_size=configs.d_model,
                          hidden_size=configs.d_model,
                          num_layers=1, batch_first=True)

        self.head = nn.Linear(configs.d_model, configs.pred_len)

    def _minmax_norm(self, x):
        # x: [B, L, C] — per-batch per-variate min-max
        x_min = x.amin(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)
        scale = (x_max - x_min).clamp(min=1e-5)
        return (x - x_min) / scale, x_min, scale

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, C = x_enc.shape
        x, x_min, scale = self._minmax_norm(x_enc)

        if self.pad_len > 0:
            pad = x[:, -1:, :].expand(-1, self.pad_len, -1)
            x = torch.cat([x, pad], dim=1)  # [B, L+pad, C]

        # Channel-independent: fold C into batch
        x = x.permute(0, 2, 1).contiguous().view(B * C, -1, 1)  # [B*C, L', 1]
        x = self.token_embed(x)  # [B*C, L', d]

        # Patching -> [B*C, num_patches, patch_len, d]
        Np, P = self.num_patches, self.patch_len
        x = x.view(B * C, Np, P, self.d_model)

        # Local Informer: process each patch independently
        x_local = x.view(B * C * Np, P, self.d_model)
        for blk in self.local_blocks:
            x_local = blk(x_local)

        # Mean pool within each patch -> [B*C, Np, d]
        pooled = x_local.view(B * C, Np, P, self.d_model).mean(dim=2)

        # Global Informer across patches
        x_global = pooled
        for blk in self.global_blocks:
            x_global = blk(x_global)

        # GRU aggregation; take final hidden state
        _, h_n = self.gru(x_global)  # h_n: [1, B*C, d]
        h_final = h_n.squeeze(0)  # [B*C, d]

        # Forecast head -> [B*C, pred_len]
        y = self.head(h_final)
        y = y.view(B, C, self.pred_len).permute(0, 2, 1).contiguous()  # [B, pred_len, C]

        # Denormalize
        y = y * scale + x_min
        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None