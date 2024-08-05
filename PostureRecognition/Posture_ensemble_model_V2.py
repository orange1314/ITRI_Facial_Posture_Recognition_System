import torch
import torch.nn as nn

# 定義 Temporal Convolutional Network (TCN) 模型
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
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        y1 = self.network(x)
        out = self.fc(y1[:, :, -1])
        return out

# 定義 BiLSTM 模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 乘以2是因為雙向

    def forward(self, x):
        h0 = torch.zeros(self.bilstm.num_layers * 2, x.size(0), self.bilstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.bilstm.num_layers * 2, x.size(0), self.bilstm.hidden_size).to(x.device)
        out, _ = self.bilstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 定義 Self-Attention 模型
class SelfAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output

class AttentionModel(nn.Module):
    def __init__(self, input_dim, num_classes, n_heads=4):
        super(AttentionModel, self).__init__()
        self.attention = SelfAttention(input_dim, n_heads)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        attn_output = self.attention(x)
        out = self.fc(attn_output[:, -1, :])  # 取最後一個時間步的輸出
        return out

# 定義集成模型
class CombinedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, num_channels, n_heads=4):
        super(CombinedModel, self).__init__()
        self.bilstm = BiLSTMModel(input_dim, hidden_dim, num_layers, num_classes)
        self.tcn = PostureTemporalConvNet(input_dim, num_channels, num_classes)
        self.attention = AttentionModel(input_dim, num_classes, n_heads)
        self.fc = nn.Linear(num_classes * 3, num_classes)  # 結合三個模型的輸出

    def forward(self, x):
        bilstm_out = self.bilstm(x)
        tcn_out = self.tcn(x)
        attention_out = self.attention(x)
        combined = torch.cat((bilstm_out, tcn_out, attention_out), dim=1)
        out = self.fc(combined)
        return out
