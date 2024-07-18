import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, Live
import pickle
from Posture_ensemble_model import PostureLSTMModel, PostureGRUModel, PostureTemporalConvNet, PostureEnsembleModel
import torch.nn as nn
import argparse

def main(time_step):
    """
    主函數，用於訓練姿態集成模型。
    
    參數:
    - time_step: int，滑動窗口的大小。
    
    功能:
    - 加載數據並進行標籤編碼。
    - 分割數據集為訓練集和測試集。
    - 創建數據集和數據加載器。
    - 初始化模型和訓練相關的參數。
    - 訓練和評估模型。
    - 保存訓練好的模型。
    """
    # 加載數據
    loaded = np.load('data_combined.npz')
    data_slices = loaded['data']
    labels = loaded['labels']

    # 創建LabelEncoder並進行編碼
    label_encoder = LabelEncoder()
    int_labels = label_encoder.fit_transform(labels)

    # 保存LabelEncoder到.pkl文件
    with open('Model/posture_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # 打印標籤列表
    print(f'Labels: {label_encoder.classes_}')

    # 分割數據集為訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(data_slices, int_labels, test_size=0.2, random_state=42)

    # 創建數據集和數據加載器
    class SkeletonDataset(Dataset):
        """
        自定義數據集類別，用於加載骨架數據。
        
        參數:
        - data: numpy.ndarray，數據片段。
        - labels: numpy.ndarray，數據標籤。
        """
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

    train_dataset = SkeletonDataset(X_train, y_train)
    test_dataset = SkeletonDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 設置參數
    input_dim = 33 * 4  # 33 keypoints * 4 coords (xyzv)
    hidden_dim = 128
    num_layers = 2
    num_classes = len(label_encoder.classes_)
    tcn_channels = [128, 128]

    lstm_model = PostureLSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes)
    gru_model = PostureGRUModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes)
    tcn_model = PostureTemporalConvNet(num_inputs=input_dim, num_channels=tcn_channels, num_classes=num_classes, dropout=0.2)

    # 創建集成模型
    model = PostureEnsembleModel(lstm_model, gru_model, tcn_model, num_classes=num_classes)

    # 將模型移動到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 訓練函數
    def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
        """
        訓練模型。
        
        參數:
        - model: nn.Module，待訓練的模型。
        - train_loader: DataLoader，訓練數據加載器。
        - criterion: 損失函數。
        - optimizer: 優化器。
        - num_epochs: int，訓練的輪數。
        
        功能:
        - 訓練模型並顯示訓練進度。
        """
        progress = Progress(
            SpinnerColumn(), 
            BarColumn(), 
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"), 
            TextColumn("[progress.description]{task.description}"),
            auto_refresh=True
        )

        with Live(progress, refresh_per_second=10):
            train_task = progress.add_task(
                description="Initializing...", 
                total=num_epochs * len(train_loader)
            )
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.view(inputs.size(0), time_step, -1)  # 形狀為 (batch_size, frames, keypoints * coords)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    progress.update(
                        train_task,
                        description=f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}",
                        advance=1
                    )

    # 評估函數
    def evaluate_model(model, test_loader):
        """
        評估模型。
        
        參數:
        - model: nn.Module，待評估的模型。
        - test_loader: DataLoader，測試數據加載器。
        
        功能:
        - 評估模型並顯示評估進度。
        """
        model.eval()
        correct = 0
        total = 0

        progress = Progress(
            SpinnerColumn(), 
            BarColumn(), 
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"), 
            TextColumn("[progress.description]{task.description}"),
            auto_refresh=True
        )
        
        with Live(progress, refresh_per_second=10):
            eval_task = progress.add_task("Evaluating", total=len(test_loader))
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.view(inputs.size(0), time_step, -1)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    progress.update(eval_task, advance=1)

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

    # 訓練模型
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, num_epochs=128)

    # 評估模型
    evaluate_model(model, test_loader)

    # 保存模型
    torch.save({
        'lstm_model_state_dict': lstm_model.state_dict(),
        'gru_model_state_dict': gru_model.state_dict(),
        'tcn_model_state_dict': tcn_model.state_dict(),
        'ensemble_model_state_dict': model.state_dict()
    }, r"Model/Posture_ensemble_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Posture Ensemble Model.")
    parser.add_argument("--time_step", type=int, default=70, help="The size of the time step for the sliding window.")
    args = parser.parse_args()

    main(args.time_step)
