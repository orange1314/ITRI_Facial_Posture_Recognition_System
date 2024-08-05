# import torch
# import torch.nn as nn
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from torchvision import transforms

# # 定義數據增強類
# class DataAugmentation:
#     def __init__(self):
#         self.transform = transforms.Compose([
#             transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
#         ])
    
#     def __call__(self, skeleton):
#         # skeleton: (seq_len, keypoints, coords)
#         skeleton = skeleton.permute(2, 0, 1)  # (coords, seq_len, keypoints)
#         skeleton = self.transform(skeleton)
#         skeleton = skeleton.permute(1, 2, 0)  # (seq_len, keypoints, coords)
#         return skeleton

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # x: (seq_len, batch_size, d_model)
#         x = x + self.pe[:x.size(0), :]  # (seq_len, batch_size, d_model)
#         return self.dropout(x)

# class PostureTransformerModel(nn.Module):
#     def __init__(self, input_dim, num_classes, nhead=8, num_encoder_layers=6, dim_feedforward=512, dropout=0.1):
#         super(PostureTransformerModel, self).__init__()
#         self.model_type = 'Transformer'
#         self.pos_encoder = PositionalEncoding(input_dim, dropout)
#         encoder_layers = TransformerEncoderLayer(input_dim, nhead, dim_feedforward, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
#         self.input_dim = input_dim
#         self.decoder = nn.Linear(input_dim, num_classes)

#     def forward(self, src):
#         # src: (batch_size, seq_len, input_dim)
#         src = src.permute(1, 0, 2)  # (seq_len, batch_size, input_dim)
#         src = self.pos_encoder(src)  # (seq_len, batch_size, input_dim)
#         output = self.transformer_encoder(src)  # (seq_len, batch_size, input_dim)
#         output = self.decoder(output[-1, :, :])  # (batch_size, num_classes)
#         return output
    
# # 定義 BiLSTM 模型
# class BiLSTMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
#         super(BiLSTMModel, self).__init__()
#         self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 乘以2是因為雙向

#     def forward(self, x):
#         # x: (batch_size, seq_len, input_dim)
#         h0 = torch.zeros(self.bilstm.num_layers * 2, x.size(0), self.bilstm.hidden_size).to(x.device)
#         c0 = torch.zeros(self.bilstm.num_layers * 2, x.size(0), self.bilstm.hidden_size).to(x.device)
#         out, _ = self.bilstm(x, (h0, c0))  # out: (batch_size, seq_len, hidden_dim * 2)
#         out = self.fc(out[:, -1, :])  # (batch_size, num_classes)
#         return out

# # 定義 Self-Attention 模型
# class SelfAttention(nn.Module):
#     def __init__(self, input_dim, n_heads):
#         super(SelfAttention, self).__init__()
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=n_heads, batch_first=True)

#     def forward(self, x):
#         # x: (batch_size, seq_len, input_dim)
#         attn_output, _ = self.multihead_attn(x, x, x)  # attn_output: (batch_size, seq_len, input_dim)
#         return attn_output

# class AttentionModel(nn.Module):
#     def __init__(self, input_dim, num_classes, n_heads=4):
#         super(AttentionModel, self).__init__()
#         self.attention = SelfAttention(input_dim, n_heads)
#         self.fc = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         # x: (batch_size, seq_len, input_dim)
#         attn_output = self.attention(x)  # (batch_size, seq_len, input_dim)
#         out = self.fc(attn_output[:, -1, :])  # (batch_size, num_classes)
#         return out

# # 定義 TCN 模型
# class PostureTemporalConvNet(nn.Module):
#     def __init__(self, num_inputs, num_channels, num_classes, kernel_size=2, dropout=0.2):
#         super(PostureTemporalConvNet, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [nn.Conv1d(in_channels, out_channels, kernel_size, 
#                                  stride=1, padding=(kernel_size-1) * dilation_size, dilation=dilation_size),
#                        nn.ReLU(),
#                        nn.Dropout(dropout)]
#         self.network = nn.Sequential(*layers)
#         self.fc = nn.Linear(out_channels, num_classes)

#     def forward(self, x):
#         # x: (batch_size, seq_len, input_dim)
#         x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
#         y1 = self.network(x)  # (batch_size, num_channels[-1], seq_len)
#         out = self.fc(y1[:, :, -1])  # (batch_size, num_classes)
#         return out

# class CombinedModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, num_classes, num_channels, n_heads=4):
#         super(CombinedModel, self).__init__()
#         self.bilstm = BiLSTMModel(input_dim, hidden_dim, num_layers, num_classes)
#         self.tcn = PostureTemporalConvNet(input_dim, num_channels, num_classes)
#         self.attention = AttentionModel(input_dim, num_classes, n_heads)
#         self.transformer = PostureTransformerModel(input_dim, num_classes, nhead=n_heads)

#         # 增加更多的全連接層
#         self.fc1 = nn.Linear(num_classes * 4, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, num_classes)

#         self.dropout = nn.Dropout(p=0.5)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         bilstm_out = self.bilstm(x)  # (batch_size, num_classes)
#         tcn_out = self.tcn(x)  # (batch_size, num_classes)
#         attention_out = self.attention(x)  # (batch_size, num_classes)
#         transformer_out = self.transformer(x)  # (batch_size, num_classes)
        
#         # 確保輸出形狀一致
#         bilstm_out = bilstm_out.unsqueeze(1)  # (batch_size, 1, num_classes)
#         tcn_out = tcn_out.unsqueeze(1)  # (batch_size, 1, num_classes)
#         attention_out = attention_out.unsqueeze(1)  # (batch_size, 1, num_classes)
#         transformer_out = transformer_out.unsqueeze(1)  # (batch_size, 1, num_classes)

#         combined = torch.cat((bilstm_out, tcn_out, attention_out, transformer_out), dim=1)  # (batch_size, 4, num_classes)
#         combined = combined.view(combined.size(0), -1)  # (batch_size, 4 * num_classes)
        
#         # 通過更多的全連接層
#         out = self.fc1(combined)  # (batch_size, 512)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.fc2(out)  # (batch_size, 256)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.fc3(out)  # (batch_size, num_classes)

#         return out

# V2
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import transforms
import torch.nn.functional as F

# 定義數據增強類
class DataAugmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        ])
    
    def __call__(self, skeleton):
        # skeleton: (seq_len, keypoints, coords)
        skeleton = skeleton.permute(2, 0, 1)  # (coords, seq_len, keypoints)
        skeleton = self.transform(skeleton)
        skeleton = skeleton.permute(1, 2, 0)  # (coords, seq_len, keypoints) -> (seq_len, keypoints, coords)
        return skeleton

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]  # (seq_len, batch_size, d_model)
        return self.dropout(x)

class PostureTransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, nhead=8, num_encoder_layers=6, dim_feedforward=512, dropout=0.1):
        super(PostureTransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = TransformerEncoderLayer(input_dim, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.input_dim = input_dim
        self.decoder = nn.Linear(input_dim, num_classes)

    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, input_dim)
        src = self.pos_encoder(src)  # (seq_len, batch_size, input_dim)
        output = self.transformer_encoder(src)  # (seq_len, batch_size, input_dim)
        output = self.decoder(output[-1, :, :])  # (batch_size, num_classes)
        return output

# 定義 BiLSTM 模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 乘以2是因為雙向

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.bilstm.num_layers * 2, x.size(0), self.bilstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.bilstm.num_layers * 2, x.size(0), self.bilstm.hidden_size).to(x.device)
        out, _ = self.bilstm(x, (h0, c0))  # out: (batch_size, seq_len, hidden_dim * 2)
        out = self.fc(out[:, -1, :])  # (batch_size, num_classes)
        return out

# 定義 Self-Attention 模型
class SelfAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        attn_output, _ = self.multihead_attn(x, x, x)  # attn_output: (batch_size, seq_len, input_dim)
        return attn_output

class AttentionModel(nn.Module):
    def __init__(self, input_dim, num_classes, n_heads=4):
        super(AttentionModel, self).__init__()
        self.attention = SelfAttention(input_dim, n_heads)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        attn_output = self.attention(x)  # (batch_size, seq_len, input_dim)
        out = self.fc(attn_output[:, -1, :])  # (batch_size, num_classes)
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
        # x: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        y1 = self.network(x)  # (batch_size, num_channels[-1], seq_len)
        out = self.fc(y1[:, :, -1])  # (batch_size, num_classes)
        return out

# 定義帶GRU的殘差模塊
class ResidualGRUBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ResidualGRUBlock, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim)
        out = self.fc(out[:, -1, :])  # (batch_size, num_classes)
        return out

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(out_channels, num_classes)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None
        
    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        residual = x
        out = self.conv1(x)  # (batch_size, out_channels, seq_len)
        out = self.bn1(out)  # (batch_size, out_channels, seq_len)
        out = self.relu(out)  # (batch_size, out_channels, seq_len)
        out = self.conv2(out)  # (batch_size, out_channels, seq_len)
        out = self.bn2(out)  # (batch_size, out_channels, seq_len)

        if self.residual_conv:
            residual = self.residual_conv(residual)  # (batch_size, out_channels, seq_len)
        
        out += residual  # (batch_size, out_channels, seq_len)
        out = self.relu(out)  # (batch_size, out_channels, seq_len)
        out = out.mean(dim=2)  # (batch_size, out_channels) - global average pooling
        out = self.fc(out)  # (batch_size, num_classes)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)  # (num_layers, batch_size, hidden_dim)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)  # (num_layers, batch_size, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_len, hidden_dim)
        out = self.fc(out[:, -1, :])  # (batch_size, num_classes)
        return out

class AttentionCombine(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionCombine, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim, output_dim))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (batch_size, num_models, num_classes)
        batch_size, num_models, num_classes = x.size()
        flattened_dim = num_models * num_classes
        attention_scores = torch.matmul(x.view(batch_size, flattened_dim), self.attention_weights)  # (batch_size, num_models * num_classes) @ (num_models * num_classes, output_dim) -> (batch_size, output_dim)
        attention_scores = self.softmax(attention_scores)  # (batch_size, output_dim)
        attention_scores = attention_scores.view(batch_size, num_models, num_classes)  # (batch_size, num_models, num_classes)
        combined = torch.sum(attention_scores * x, dim=1)  # (batch_size, num_classes)
        return combined

class CombinedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, num_channels, n_heads=4):
        super(CombinedModel, self).__init__()
        self.bilstm = BiLSTMModel(input_dim, hidden_dim, num_layers, num_classes)
        self.tcn = PostureTemporalConvNet(input_dim, num_channels, num_classes)
        self.attention = AttentionModel(input_dim, num_classes, n_heads)
        self.transformer = PostureTransformerModel(input_dim, num_classes, nhead=n_heads)
        self.residual_gru = ResidualGRUBlock(input_dim, hidden_dim, num_classes)
        self.resnet_block = ResNetBlock(input_dim, input_dim, num_classes)
        self.lstm = LSTMModel(input_dim, hidden_dim, num_layers, num_classes)

        self.attention_combine = AttentionCombine(num_classes * 7, num_classes * 7)
        self.fc = nn.Linear(num_classes * 7, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)

        bilstm_out = self.bilstm(x)  # (batch_size, num_classes)

        tcn_out = self.tcn(x)  # (batch_size, num_classes)

        attention_out = self.attention(x)  # (batch_size, num_classes)

        transformer_out = self.transformer(x)  # (batch_size, num_classes)

        residual_gru_out = self.residual_gru(x)  # (batch_size, num_classes)

        resnet_x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        resnet_out = self.resnet_block(resnet_x)  # (batch_size, num_classes)

        lstm_out = self.lstm(x)  # (batch_size, num_classes)


        # Combine the outputs
        combined = torch.cat((bilstm_out, tcn_out, attention_out, transformer_out, residual_gru_out, resnet_out, lstm_out), dim=1)  # (batch_size, num_classes * 7)
        combined = combined.unsqueeze(1)  # (batch_size, 1, num_classes * 7)

        combined = self.attention_combine(combined)  # (batch_size, num_classes * 7)
        combined = combined.squeeze(1)  # (batch_size, num_classes * 7)


        # print(f"bilstm_out shape: {bilstm_out.shape}")
        # print(f"tcn_out shape: {tcn_out.shape}")
        # print(f"attention_out shape: {attention_out.shape}")
        # print(f"transformer_out shape: {transformer_out.shape}")
        # print(f"residual_gru_out shape: {residual_gru_out.shape}")
        # print(f"resnet_out shape: {resnet_out.shape}")
        # print(f"lstm_out shape: {lstm_out.shape}")
        # print(f"combined shape before attention: {combined.shape}")

        out = self.fc(combined)  # (batch_size, num_classes)
        return out
