import os
import numpy as np
import argparse
import shutil

# 設定資料夾路徑
npy_path = "action_npy"
train_path = "train"

# 清空 train 資料夾
def clear_directory(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except PermissionError:
            print(f"Permission denied: '{path}'. Attempting to use system command.")
            # 使用系統命令強制刪除
            os.system(f'rd /s /q "{path}"')
        except Exception as e:
            print(f"Error deleting directory '{path}': {e}")

# 定義滑動窗口函數
def sliding_window(data, window_size, step_size, label):
    num_frames, num_keypoints, num_coords = data.shape
    slices = []

    # 使用滑動窗口提取片段
    for start in range(0, num_frames - window_size + 1, step_size):
        end = start + window_size
        slice_data = data[start:end]
        slices.append((slice_data, label))

    # 處理剩餘不足的片段
    if num_frames % step_size != 0 and num_frames > window_size:
        slice_data = data[-window_size:]
        slices.append((slice_data, label))

    return slices

# 處理每個動作的資料並儲存
def process_and_save_data(npy_path, train_path, window_size, step_size=10):
    labels = os.listdir(npy_path)

    for label in labels:
        action_folder = os.path.join(npy_path, label)
        save_folder = os.path.join(train_path, label)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        files = os.listdir(action_folder)
        for file_index, file_name in enumerate(files):
            file_path = os.path.join(action_folder, file_name)
            data = np.load(file_path)
            slices = sliding_window(data, window_size, step_size, label)

            for slice_index, (slice_data, _) in enumerate(slices):
                save_path = os.path.join(save_folder, f"{os.path.splitext(file_name)[0]}_{slice_index}.npy")
                np.save(save_path, slice_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save action data with a sliding window.")
    parser.add_argument("--time_step", type=int, default=70, help="The size of the sliding window.")
    args = parser.parse_args()
    # 清空 train 資料夾
    clear_directory(train_path)
    # 執行處理並儲存資料
    process_and_save_data(npy_path, train_path, window_size=args.time_step)
