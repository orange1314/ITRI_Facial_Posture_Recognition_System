import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, Live
import pickle
from Posture_ensemble_model_V2 import CombinedModel
import torch.nn as nn
import argparse
import sys
import matplotlib.pyplot as plt

def main(time_step):
    """
    主函數，用於訓練姿態集成模型。
    
    參數:
    - time_step: int，滑動窗口的大小。
    
    功能:
    - 加載數據並進行標籤編碼。
    - 創建數據集和數據加載器。
    - 初始化模型和訓練相關的參數。
    - 訓練和評估模型。
    - 保存訓練好的模型。
    """
    # 加載數據
    loaded = np.load('data_combined_augmented_sampled.npz')
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

    # 創建數據集和數據加載器
    class SkeletonDataset(Dataset):
        """
        自定義數據集類，用於加載骨架數據。
        
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

    train_dataset = SkeletonDataset(data_slices, int_labels)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 設置參數
    input_dim = 33 * 4  # 33 keypoints * 4 coords (xyzv)
    hidden_dim = 128
    num_layers = 3
    num_classes = len(label_encoder.classes_)
    tcn_channels = [128, 128]
    n_heads = 4

    # 創建組合模型實例
    model = CombinedModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes, num_channels=tcn_channels, n_heads=n_heads)

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

        train_losses = []

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
                
                train_losses.append(running_loss / len(train_loader))
        
        return train_losses

    # 評估函數
    def evaluate_model(model, test_file_path):
        """
        評估模型。
        
        參數:
        - model: nn.Module，待評估的模型。
        - test_file_path: str，測試數據的文件路徑。
        
        功能:
        - 評估模型並顯示評估進度。
        - 計算每一個動作的正確率。
        """
        model.eval()
        total_correct = 0
        total_samples = 0
        action_correct = {}
        action_total = {}

        progress = Progress(
            SpinnerColumn(), 
            BarColumn(), 
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"), 
            TextColumn("[progress.description]{task.description}"),
            auto_refresh=True
        )
        
        # 加載測試數據
        loaded_test = np.load(test_file_path)
        test_data_slices = loaded_test['data']
        test_labels = loaded_test['labels']

        # 創建測試數據集和數據加載器
        test_dataset = SkeletonDataset(test_data_slices, label_encoder.transform(test_labels))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        with Live(progress, refresh_per_second=10):
            eval_task = progress.add_task("Evaluating", total=len(test_loader))
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.view(1, time_step, -1)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    action = label_encoder.inverse_transform([labels.item()])[0]

                    total_samples += 1
                    if action not in action_total:
                        action_total[action] = 0
                        action_correct[action] = 0
                    action_total[action] += 1
                    if predicted.item() == labels.item():
                        total_correct += 1
                        action_correct[action] += 1

                    progress.update(eval_task, advance=1)

        overall_accuracy = 100 * total_correct / total_samples
        print(f'Overall Accuracy of the model on the test set: {overall_accuracy:.2f}%')

        for action in action_correct:
            action_accuracy = 100 * action_correct[action] / action_total[action]
            print(f'Accuracy for action {action}: {action_accuracy:.2f}%')

    # 訓練模型
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=128)

    # 評估模型
    evaluate_model(model, 'test_data.npz')

    # 保存模型
    torch.save({
        'combined_model_state_dict': model.state_dict()
    }, r"Model/Posture_combined_model.pth")

    # 繪製並保存損失曲線
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('training_loss_curve.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Posture Combined Model.")
    parser.add_argument("--time_step", type=int, default=70, help="The size of the time step for the sliding window.")
    # 如果在 Jupyter Notebook 中運行，跳過 argparse
    if "ipykernel" in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    main(args.time_step)
