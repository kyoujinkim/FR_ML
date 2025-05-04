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
                                                 )
        self.fc_out = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        # Define the forward pass
        x = self.embedding(x)
        x = self.transformer_blocks(x)
        x = self.fc_out(x)
        return x