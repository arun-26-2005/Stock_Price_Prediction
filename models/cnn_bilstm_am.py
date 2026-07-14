import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.attn = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.v = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):

        scores = torch.tanh(self.attn(x))
        weights = torch.softmax(self.v(scores), dim=1)

        context = torch.sum(weights * x, dim=1)

        return context
class CNN_BiLSTM_AM(nn.Module):
    def __init__(self, input_size, hidden_size=64, cnn_filters=32):
        super(CNN_BiLSTM_AM, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_filters,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.bilstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.attn = Attention(hidden_size)

        self.fc = nn.Linear(hidden_size*2, 1)

    def forward(self, x):

        x = x.permute(0,2,1)

        x = self.conv1(x)
        x = self.relu(x)

        x = x.permute(0,2,1)

        out, _ = self.bilstm(x)

        context = self.attn(out)

        out = self.fc(context)

        return out