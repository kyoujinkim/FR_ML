import torch
import torch.nn as nn
import numpy as np
from src.layers import DataEmbedding

class Model(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, c_in, c_out):
        super(Model, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.c_in = c_in
        self.c_out = c_out

        # Define the model layers here
        self.embedding = DataEmbedding(c_in, d_model)
        self.transformer_blocks = nn.Transformer(d_model=d_model
                                                 , nhead=n_heads
                                                 , num_encoder_layers=n_layers
                                                 , num_decoder_layers=n_layers
                                                 , batch_first=True
                                                 )
        self.fc_out = nn.Linear(d_model, c_out, bias=True)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Define the forward pass
        src = self.embedding(src) * np.sqrt(self.d_model)
        tgt = self.embedding(tgt) * np.sqrt(self.d_model)

        x = self.transformer_blocks(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        x = self.fc_out(x)
        return x

    def generate_square_subsequent_mask(self, sz):
        """Generates a square subsequent mask for the transformer model."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_padding_mask(self, seq):
        """Generates a padding mask for the transformer model."""
        seq = seq.permute(0, 2, 1)
        mask = (seq == 0).unsqueeze(1).unsqueeze(2)
        return mask