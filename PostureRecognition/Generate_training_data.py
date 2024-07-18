import os
import numpy as np

# 設定訓練資料夾路徑
train_path = "train"

def combine_data(train_path):
    """
    合併訓練資料夾中的數據片段，並保存為一個 .npz 文件。
    
    參數:
    - train_path: str，包含訓練數據的資料夾路徑。
    
    功能:
    - 遍歷訓練資料夾中的所有子資料夾，讀取數據片段和標籤。
    - 將所有數據片段和標籤合併為 numpy 數組。
    - 保存合併後的數據和標籤到一個 .npz 文件。
    - 打印各類別的資料數量。
    """
    # 初始化數據和標籤列表
    data_slices = []
    labels = []

    # 初始化一個字典來計數各類別的資料數量
    label_counts = {}

    # 讀取訓練資料夾中的所有子資料夾
    labels_folders = os.listdir(train_path)

    for label in labels_folders:
        label_folder = os.path.join(train_path, label)
        if os.path.isdir(label_folder):
            files = os.listdir(label_folder)
            for file_name in files:
                file_path = os.path.join(label_folder, file_name)
                data = np.load(file_path)
                data_slices.append(data)
                labels.append(label)
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1

    # 將數據和標籤轉換為numpy數組
    data_slices = np.array(data_slices)
    labels = np.array(labels)

    # 保存數據和標籤到 .npz 文件
    np.savez('data_combined.npz', data=data_slices, labels=labels)

    # 打印各類別的資料數量
    for label, count in label_counts.items():
        print(f"Category '{label}' has {count} slices.")

    print(f"Data and labels have been saved to data_combined.npz with {len(data_slices)} slices.")

if __name__ == "__main__":
    # 執行數據合併並保存
    combine_data(train_path)
