import torch
import torch.nn as nn
from models.mtran import PositionalEncoding
from .mtran_tcn import TCNBlock
class BiLSTM_TCN(nn.Module):
    def __init__(self, input_size, hidden_size=64, embed_dim=64):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_size, embed_dim)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # Match dimension for residual (embed → hidden*2)
        self.res_lstm = nn.Linear(embed_dim, hidden_size * 2)

        # TCN 
        self.tcn = TCNBlock(hidden_size * 2)

        # Output
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):

        # -------- Input Projection --------
        x = self.input_proj(x)

        # -------- BiLSTM --------
        lstm_out, _ = self.bilstm(x)

        # Residual connection
        lstm_out = lstm_out + self.res_lstm(x)

        # -------- TCN --------
        tcn_out = self.tcn(lstm_out)

        # -------- Output --------
        out = tcn_out[:, -1, :]
        out = self.fc(out)

        return out