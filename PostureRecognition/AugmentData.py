# import numpy as np
# import pickle
# from collections import Counter

# # 定義水平翻轉函數
# def horizontal_flip(skeleton):
#     flipped_skeleton = skeleton.copy()
#     # 假設 x 座標在偶數索引
#     flipped_skeleton[:, ::2] *= -1
#     return flipped_skeleton

# # 讀取數據
# data = np.load('data_combined.npz')
# X = data['data']
# y = data['labels']

# # 對所有幀應用水平翻轉
# X_flipped = np.array([horizontal_flip(frame) for frame in X])
# y_flipped = y.copy()  # 標籤保持不變

# # 合併原始數據和翻轉數據
# X_combined = np.concatenate((X, X_flipped), axis=0)
# y_combined = np.concatenate((y, y_flipped), axis=0)

# # 保存處理後的數據
# np.savez('data_combined.npz', data=X_combined, labels=y_combined)

# # 計算各類別數量
# label_counts = Counter(y_combined)
# print("各類別數量:")
# for label, count in label_counts.items():
#     print(f"Label {label}: {count} samples")

# V3
import numpy as np
from collections import Counter

# 定義旋轉函數
def rotate_skeleton(data, angle):
    radians = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians), 0],
        [np.sin(radians),  np.cos(radians), 0],
        [0,                0,               1]
    ])
    
    # 提取 x, y, z 坐標
    coords = data[:, :, :, :3]
    # 提取 visibility
    visibility = data[:, :, :, 3:]

    # 旋轉 x, y, z 坐標
    rotated_coords = np.dot(coords, rotation_matrix.T)
    # 拼接旋轉後的坐標和 visibility
    rotated_data = np.concatenate((rotated_coords, visibility), axis=3)
    
    return rotated_data

# 定義水平翻轉函數
def horizontal_flip(skeleton):
    flipped_skeleton = skeleton.copy()
    flipped_skeleton[:, :, 0] *= -1  # 只翻轉 x 坐標
    return flipped_skeleton

# 讀取數據
data = np.load('data_combined.npz')
X = data['data']
y = data['labels']

# 記錄每個類別的原始數據的索引
original_indices = {label: np.where(y == label)[0] for label in np.unique(y)}

# 對原始數據應用旋轉操作，從45度開始，直到360度，每次增量45度
angles = np.arange(45, 360, 45)
rotated_data = [X]
for angle in angles:
    X_rotated = rotate_skeleton(X, angle)
    rotated_data.append(X_rotated)

# 合並所有旋轉後的數據
X_rotated_combined = np.concatenate(rotated_data, axis=0)
y_rotated_combined = np.concatenate([y] * (len(angles) + 1), axis=0)

# 對所有幀應用水平翻轉
X_flipped = np.array([horizontal_flip(sequence) for sequence in X_rotated_combined])
y_flipped = y_rotated_combined.copy()  # 標簽保持不變

# 合並所有數據
X_combined = np.concatenate((X_rotated_combined, X_flipped), axis=0)
y_combined = np.concatenate((y_rotated_combined, y_flipped), axis=0)

# 每一個類別隨機抽取最多5000個樣本，保留原始數據，隨機抽滿到5000
def sample_data(X, y, original_indices, max_samples=5000):
    unique_labels = np.unique(y)
    sampled_X = []
    sampled_y = []
    for label in unique_labels:
        orig_indices = original_indices[label]
        augmented_indices = np.where(y == label)[0]
        # 刪除原始數據的索引
        augmented_indices = np.setdiff1d(augmented_indices, orig_indices)
        if len(orig_indices) >= max_samples:
            final_indices = np.random.choice(orig_indices, max_samples, replace=False)
        else:
            num_needed = max_samples - len(orig_indices)
            sampled_augmented_indices = np.random.choice(augmented_indices, num_needed, replace=False)
            final_indices = np.concatenate((orig_indices, sampled_augmented_indices))
        sampled_X.append(X[final_indices])
        sampled_y.append(y[final_indices])
    return np.concatenate(sampled_X, axis=0), np.concatenate(sampled_y, axis=0)

X_sampled, y_sampled = sample_data(X_combined, y_combined, original_indices, max_samples=6500)

# 保存處理後的數據
np.savez('data_combined_augmented_sampled.npz', data=X_sampled, labels=y_sampled)

# 計算各類別數量
label_counts = Counter(y_sampled)
print("各類別數量:")
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")

