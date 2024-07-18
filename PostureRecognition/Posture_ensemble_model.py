import torch
import torch.nn as nn

# 定義 LSTM 模型
class PostureLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(PostureLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # print(f'LSTMModel input shape: {x.shape}')
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        # print(f'LSTMModel output shape: {out.shape}')
        return out

# 定義 GRU 模型
class PostureGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(PostureGRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # print(f'GRUModel input shape: {x.shape}')
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        # print(f'GRUModel output shape: {out.shape}')
        return out

# 定義 TCN 模型
class PostureTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=2, dropout=0.2):
        super(PostureTemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, 
                                 stride=1, padding=(kernel_size-1) * dilation_size, dilation=dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        # print(f'TemporalConvNet input shape: {x.shape}')
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        y1 = self.network(x)
        out = self.fc(y1[:, :, -1])
        # print(f'TemporalConvNet output shape: {out.shape}')
        return out

# 定義集成模型
class PostureEnsembleModel(nn.Module):
    def __init__(self, lstm_model, gru_model, tcn_model, num_classes):
        super(PostureEnsembleModel, self).__init__()
        self.lstm_model = lstm_model
        self.gru_model = gru_model
        self.tcn_model = tcn_model
        # self.fc = nn.Linear(num_classes * 3, num_classes)  # 3 是模型的數量

    def forward(self, x):
        # print(f'EnsembleModel input shape: {x.shape}')
        lstm_out = self.lstm_model(x)
        gru_out = self.gru_model(x)
        tcn_out = self.tcn_model(x)
        # combined = torch.cat((lstm_out, gru_out, tcn_out), dim=1)
        # print(f'EnsembleModel combined shape: {combined.shape}')
        # output = self.fc(combined)
        # print(f'EnsembleModel output shape: {output.shape}')
        combined_out = (lstm_out + gru_out + tcn_out) / 3  # 取平均
        return combined_out