
import torch.nn as nn
from .mtran import PositionalEncoding

class TCNBlock(nn.Module):
    def __init__(self, input_channels, num_layers=4, channels=32, kernel_size=7):
        super().__init__()

        layers = []
        in_ch = input_channels

        for i in range(num_layers):
            dilation = 2 ** i

            layers.append(
                nn.Conv1d(
                    in_ch,
                    channels,
                    kernel_size,
                    padding=(kernel_size - 1) * dilation,
                    dilation=dilation
                )
            )
            layers.append(nn.ReLU())

            in_ch = channels

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)

        # remove extra padding(causal trimming)
        out = out[:, :, :x.size(2)]

        return out.permute(0, 2, 1)

class BiLSTM_MTRAN_TCN(nn.Module):
    def __init__(self, input_size, hidden_size=64, embed_dim=64, num_heads=8):
        super(BiLSTM_MTRAN_TCN, self).__init__()

        # Input projection
        self.input_proj = nn.Linear(input_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)

        # BiLSTM (3 layers)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=3,                     
            batch_first=True,
            bidirectional=True
        )

        self.res_lstm = nn.Linear(embed_dim, hidden_size * 2)

        # Transformer (6 layers, 8 heads, d_model=128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )

        # Reduce dimension BEFORE TCN (128 → 32)
        self.downsample = nn.Linear(hidden_size * 2, 32)

        # TCN (4 layers, 32 neurons, kernel=7)
        self.tcn = TCNBlock(
            input_channels=32,
            num_layers=4,
            channels=32,
            kernel_size=7
        )

        # Final layer
        self.fc = nn.Linear(32, 1)

    def forward(self, x):

        # Input + Pos Encoding
        x = self.input_proj(x)
        x = self.pos_enc(x)

        # BiLSTM + Residual
        lstm_out, _ = self.bilstm(x)
        res = self.res_lstm(x)
        lstm_out = lstm_out + res

        # Transformer + Residual
        trans_out = self.transformer(lstm_out)
        trans_out = trans_out + lstm_out

        # Reduce dimension before TCN
        tcn_in = self.downsample(trans_out)

        # TCN + Residual
        tcn_out = self.tcn(tcn_in)
        tcn_out = tcn_out + tcn_in

        # Output
        out = tcn_out[:, -1, :]
        out = self.fc(out)

        return out