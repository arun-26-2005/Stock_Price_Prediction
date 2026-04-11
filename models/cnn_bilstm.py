import torch.nn as nn

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, cnn_filters=32):
        super(CNN_BiLSTM, self).__init__()

        # CNN layer
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_filters,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected output
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Convert for CNN: (batch, features, seq_len)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu(x)

        # Convert back for LSTM: (batch, seq_len, filters)
        x = x.permute(0, 2, 1)

        out, _ = self.bilstm(x)

        out = out[:, -1, :]   # last timestep

        out = self.fc(out)

        return out