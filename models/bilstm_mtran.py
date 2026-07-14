import torch
import torch.nn as nn
from models.mtran import PositionalEncoding

class BiLSTM_MTRAN(nn.Module):
    def __init__(self, input_size, hidden_size=64, embed_dim=64, num_heads=4):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_size, embed_dim)

        # Positional Encoding
        self.pos_enc = PositionalEncoding(embed_dim)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # Residual projection (to match dimensions)
        self.res_lstm = nn.Linear(embed_dim, hidden_size * 2)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # Output
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):

        # Input embedding
        x = self.input_proj(x)
        x = self.pos_enc(x)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)

        # Residual
        lstm_out = lstm_out + self.res_lstm(x)

        # Transformer
        trans_out = self.transformer(lstm_out)

        # Residual
        trans_out = trans_out + lstm_out

        # Output
        out = trans_out[:, -1, :]
        out = self.fc(out)

        return out