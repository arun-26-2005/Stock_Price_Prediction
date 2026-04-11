import torch 
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class MTRAN(nn.Module):

    def __init__(self, input_size, d_model=64, num_heads=4, num_layers=2):
        super().__init__()

        # Feature embedding
        self.embedding = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=0.2,
            batch_first=True
        )

        # Multi-layer transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layer
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):

        # (batch, seq, features)

        x = self.embedding(x)

        x = self.pos_enc(x)

        x = self.transformer(x)

        # take last timestep
        x = x[:, -1, :]

        out = self.fc(x)

        return out