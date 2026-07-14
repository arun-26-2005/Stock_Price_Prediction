import torch.nn as nn
import torch
from models.mtran import PositionalEncoding

class MTRAN(nn.Module):

    def __init__(self, input_size, d_model=64, num_heads=4, num_layers=2):
        super().__init__()

        self.embedding = nn.Linear(input_size, d_model)

        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True,
            dropout=0.2
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):

        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.transformer(x)

        return x   # return full sequence
    
class TCNBlock(nn.Module):

        def __init__(self, channels):
            super().__init__()

            self.conv1 = nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )

            self.conv2 = nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2
            )

            self.relu = nn.ReLU()

        def forward(self, x):

            residual = x

            x = x.permute(0,2,1)

            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))

            x = x.permute(0,2,1)

            return x + residual
class MTRAN_TCN(nn.Module):

    def __init__(self, input_size, d_model=64):

        super().__init__()

        self.mtran = MTRAN(input_size, d_model)

        self.tcn = TCNBlock(d_model)

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):

        x = self.mtran(x)

        x = self.tcn(x)

        x = torch.mean(x, dim=1)

        out = self.fc(x)

        return out