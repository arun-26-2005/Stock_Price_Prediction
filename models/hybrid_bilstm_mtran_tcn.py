
import torch.nn as nn
from .mtran import PositionalEncoding

class TCNBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=2):
        super(TCNBlock, self).__init__()

        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            padding=dilation,
            dilation=dilation
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0,2,1)
        return x

class BiLSTM_MTRAN_TCN(nn.Module):
    def __init__(self, input_size, hidden_size=64, embed_dim=64, num_heads=4):
        super(BiLSTM_MTRAN_TCN, self).__init__()

        # Input projection
        self.input_proj = nn.Linear(input_size, embed_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(embed_dim)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # Match dimension for residual (embed_dim → hidden*2)
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

        # TCN
        self.tcn = TCNBlock(
            input_channels=hidden_size * 2,
            output_channels=hidden_size * 2
        )

        # Final layer
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):

        # ---------- Input + Positional Encoding ----------
        x = self.input_proj(x)
        x = self.pos_enc(x)

        # ---------- BiLSTM + Residual ----------
        lstm_out, _ = self.bilstm(x)

       
        res = self.res_lstm(x)

        lstm_out = lstm_out + res   # Residual connection

        # ---------- Transformer + Residual ----------
        trans_out = self.transformer(lstm_out)

        trans_out = trans_out + lstm_out   # Residual connection

        # ---------- TCN + Residual ----------
        tcn_out = self.tcn(trans_out)

        tcn_out = tcn_out + trans_out   # Residual connection

        # ---------- Output ----------
        out = tcn_out[:, -1, :]
        out = self.fc(out)

        return out