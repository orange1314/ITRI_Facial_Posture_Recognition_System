# import torch
# import torch.nn as nn
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
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
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src)
#         output = self.decoder(output[-1, :, :])  # 取最後一個時間步的輸出
#         return output

# # 定義集成模型
# class PostureEnsembleTransformerModel(nn.Module):
#     def __init__(self, transformer_model, num_classes):
#         super(PostureEnsembleTransformerModel, self).__init__()
#         self.transformer_model = transformer_model
#         self.fc = nn.Linear(num_classes, num_classes)

#     def forward(self, x):
#         transformer_out = self.transformer_model(x)
#         output = self.fc(transformer_out)
#         return output

#v2

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision.transforms as transforms

# 定義數據增強類
class DataAugmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        ])
    
    def __call__(self, skeleton):
        # 對骨架數據進行增強
        skeleton = skeleton.permute(2, 0, 1)  # (seq_len, keypoints, coords) -> (coords, seq_len, keypoints)
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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
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
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[-1, :, :])  # 取最後一個時間步的輸出
        return output

# 定義集成模型
class PostureEnsembleTransformerModel(nn.Module):
    def __init__(self, transformer_model, num_classes):
        super(PostureEnsembleTransformerModel, self).__init__()
        self.transformer_model = transformer_model
        self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        transformer_out = self.transformer_model(x)
        output = self.fc(transformer_out)
        return output

