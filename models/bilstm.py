import torch.nn as nn
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(BiLSTMModel, self).__init__()

        self.bilstm = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    batch_first=True,
    bidirectional=True
)

        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out