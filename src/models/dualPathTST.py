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


class PyramidNet(nn.Module):
    def __init__(self, depth, alpha, input_dim, output_dim, model_type='add'):
        """
        depth: 잔차 유닛(Residual Units)의 총 개수 (N) [cite: 224, 239]
        alpha: 네트워크의 너비를 결정하는 하이퍼파라미터 [cite: 224]
        model_type: 'add' (가산) 또는 'mul' (승산)
        """
        super(PyramidNet, self).__init__()
        self.depth = depth
        self.alpha = alpha
        self.model_type = model_type

        # 레이어 리스트 초기화
        self.layers = nn.ModuleList()

        # 초기 차원 설정 (k=1)
        current_dim = 16  # 논문 수식 (7), (8)의 초기값 [cite: 227, 231]

        # 입력 차원을 초기 차원(16)으로 맞추는 투영 레이어
        self.input_proj = nn.Linear(input_dim, current_dim)

        # 수식에 따라 레이어별 차원 계산 및 생성
        for k in range(2, depth + 2):
            if self.model_type == 'add':
                # 가산(Additive) 방식: 차원이 선형적으로 증가 [cite: 240]
                # D_k = D_{k-1} + alpha / N
                next_dim = int(current_dim + (self.alpha / self.depth))
            else:
                # 승산(Multiplicative) 방식: 차원이 기하급수적으로 증가 [cite: 241]
                # D_k = D_{k-1} * alpha^(1/N)
                next_dim = int(current_dim * (self.alpha ** (1 / self.depth)))

            self.layers.append(self.make_pyramid_unit(current_dim, next_dim))
            current_dim = next_dim

        # 최종 출력 레이어 (예: de-stationary factor τ, Δ 추출용) [cite: 212]
        self.output_proj = nn.Linear(current_dim, output_dim)

    def make_pyramid_unit(self, in_dim, out_dim):
        """기본적인 잔차 유닛 구성"""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),  # 안정성을 위한 배치 정규화 [cite: 62]
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [batch_size, input_dim]
        out = self.input_proj(x)

        for layer in self.layers:
            out = layer(out)

        return self.output_proj(out)

def DeStationaryAttention(Q, K, V, tau, delta):
    '''
    # 식 (6) 적용: tau와 delta를 어텐션 스코어 계산에 포함 [cite: 216]
    attn_weights = Softmax((tau * (Q @ K.T) + delta) / Sqrt(d_k))
    return attn_weights @ V
    '''
    attn_weights = torch.softmax((tau * (Q @ K.transpose(-2, -1)) + delta) / (Q.size(-1) ** 0.5), dim=-1)
    return attn_weights @ V


class Model(nn.Module):
    """
    DualPathTST 알고리즘 Pseudo CodeInput: Multivariate Time Series $x = \{x^{(1)}, \dots, x^{(M)}\}$ of length $L$
    Output: Predicted sequences $\hat{x}$ of length $T$

    단계 1: 데이터 전처리 및 패칭 (Patching Module)Channel Independence: 다변량 데이터를 $M$개의 독립적인 단변량 시퀀스로 분리한다.
    Instance Normalization: 각 채널별로 Reversible Instance Normalization(RevIN)을 적용하여 평균 0, 표준편차 1로 정규화한다.
    $\mu_x, \sigma_x$ 계산 및 저장 (이후 역정규화 및 De-stationary Factor 추출에 사용).
    Patching: 정규화된 시퀀스를 길이 $P$, 스트라이드 $S$인 패치(Patch)들로 분할하여 토큰화한다.
    Embedding: 패치들을 차원 $D$로 선형 투영(Linear Projection)하고 위치 인코딩(Positional Encoding)을 더한다.

    단계 2: 이중 경로 처리 (Dual-Pathway Backbone)데이터는 병렬적인 두 경로로 입력됩니다:
    경로 A: De-stationary Attention Pathway (Intra-variable)
    Factor Learning: 원본 데이터의 $\mu_x, \sigma_x$를 PyramidNet 기반 MLP에 통과시켜 정규화로 손실된 정보를 복구할 요소($\tau, \Delta$)를 도출한다.
    Attention Calculation: 도출된 요소를 사용하여 'De-stationary Attention'을 수행한다: $$Attn(Q', K', V', \tau, \Delta) = Softmax\left(\frac{\tau Q'K'^T + 1\Delta^T}{\sqrt{d_k}}\right)V'$$
    이 과정은 시퀀스 내부의 시간적 동역학을 포착한다.
    경로 B: Convolutional Pathway (Cross-variable)
    TCN Processing: 여러 변수의 임베딩을 결합하여 Temporal Convolutional Network(TCN)에 입력한다.
    Dilated Convolution: 팽창 컨볼루션을 통해 변수 간 상관관계(Cross-variable dependency)를 학습한다.

    단계 3: 융합 및 출력 (Fusion & Output Module)
    Gated Fusion: 시그모이드(Sigmoid) 함수 기반의 게이트 신호 $z$를 계산하여 두 경로의 정보를 동적으로 통합한다: $$y = z \cdot x_{Attention} + (1-z) \cdot x_{TCN}$$
    Batch Normalization: 출력 직전 배치 정규화를 적용하여 내부 공변량 변화(Internal Covariate Shift)를 방지하고 학습 안정성을 높인다.
    Linear Head: 데이터를 평탄화(Flatten)하고 선형 레이어를 통과시켜 최종 예측값 $\hat{x}$를 산출한다.
    Denormalization: RevIN의 역과정을 통해 저장해둔 $\mu_x, \sigma_x$를 다시 적용하여 원래 스케일로 복원한다.
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
        # Stage 2: Dual-Pathway Backbone
        #          Path A: De-stationary Attention (Intra-variable)
        #          Path B: Convolutional Pathway (Cross-variable)
        self.mlp_pyramid = PyramidNet(depth=4, alpha=48, input_dim=configs.enc_in * 2, output_dim=configs.d_model * 2, model_type='add')

        # ------------------------------------------------------------------
        # Output Head
        # ------------------------------------------------------------------
        head_nf = configs.d_model * self.patch_num
        self.head = FlattenHead(configs.enc_in, head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, N]
        return None