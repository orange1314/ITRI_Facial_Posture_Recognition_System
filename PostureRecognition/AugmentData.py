import numpy as np
import pickle
from collections import Counter

# 定義水平翻轉函數
def horizontal_flip(skeleton):
    flipped_skeleton = skeleton.copy()
    # 假設 x 座標在偶數索引
    flipped_skeleton[:, ::2] *= -1
    return flipped_skeleton

# 讀取數據
data = np.load('data_combined.npz')
X = data['data']
y = data['labels']

# 對所有幀應用水平翻轉
X_flipped = np.array([horizontal_flip(frame) for frame in X])
y_flipped = y.copy()  # 標籤保持不變

# 合併原始數據和翻轉數據
X_combined = np.concatenate((X, X_flipped), axis=0)
y_combined = np.concatenate((y, y_flipped), axis=0)

# 保存處理後的數據
np.savez('data_combined.npz', data=X_combined, labels=y_combined)

# 計算各類別數量
label_counts = Counter(y_combined)
print("各類別數量:")
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")
