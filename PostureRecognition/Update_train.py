# import os
# import numpy as np
# import argparse
# import shutil

# # 設定資料夾路徑
# npy_path = "action_npy"
# train_path = "train"

# # 清空 train 資料夾
# def clear_directory(path):
#     """
#     清空指定目錄。
    
#     參數:
#     - path: str，目錄的路徑。
    
#     功能:
#     - 刪除指定目錄及其內容。如果刪除失敗，則嘗試使用系統命令強制刪除。
#     """
#     if os.path.exists(path):
#         try:
#             shutil.rmtree(path)
#         except PermissionError:
#             print(f"Permission denied: '{path}'. Attempting to use system command.")
#             # 使用系統命令強制刪除
#             os.system(f'rd /s /q "{path}"')
#         except Exception as e:
#             print(f"Error deleting directory '{path}': {e}")

# # 定義滑動窗口函數
# def sliding_window(data, window_size, step_size, label):
#     """
#     使用滑動窗口提取數據片段。
    
#     參數:
#     - data: numpy.ndarray，包含骨架資訊的數據。
#     - window_size: int，滑動窗口的大小。
#     - step_size: int，滑動窗口的步長。
#     - label: str，數據的標籤。
    
#     返回:
#     - slices: list，包含數據片段和相應標籤的列表。
#     """

#     num_frames, num_keypoints, num_coords = data.shape
#     slices = []

#     # 使用滑動窗口提取片段
#     for start in range(0, num_frames - window_size + 1, step_size):
#         end = start + window_size
#         slice_data = data[start:end]
#         slices.append((slice_data, label))

#     # 處理剩餘不足的片段
#     if num_frames % step_size != 0 and num_frames > window_size:
#         slice_data = data[-window_size:]
#         slices.append((slice_data, label))

#     return slices

# # 處理每個動作的資料並儲存
# def process_and_save_data(npy_path, train_path, window_size, step_size=10):
#     """
#     處理每個動作的數據，使用滑動窗口提取片段並保存。
    
#     參數:
#     - npy_path: str，包含原始數據的目錄。
#     - train_path: str，保存處理後數據的目錄。
#     - window_size: int，滑動窗口的大小。
#     - step_size: int，滑動窗口的步長。
#     """
#     labels = os.listdir(npy_path)

#     for label in labels:
#         action_folder = os.path.join(npy_path, label)
#         save_folder = os.path.join(train_path, label)

#         if not os.path.exists(save_folder):
#             os.makedirs(save_folder)

#         files = os.listdir(action_folder)
#         for file_index, file_name in enumerate(files):
#             file_path = os.path.join(action_folder, file_name)
#             data = np.load(file_path)
#             slices = sliding_window(data, window_size, step_size, label)

#             for slice_index, (slice_data, _) in enumerate(slices):
#                 save_path = os.path.join(save_folder, f"{os.path.splitext(file_name)[0]}_{slice_index}.npy")
#                 np.save(save_path, slice_data)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process and save action data with a sliding window.")
#     parser.add_argument("--time_step", type=int, default=70, help="The size of the sliding window.")
#     args = parser.parse_args()
#     # 清空 train 資料夾
#     clear_directory(train_path)
#     # 執行處理並儲存資料
#     process_and_save_data(npy_path, train_path, window_size=args.time_step)

# # V2
# import os
# import numpy as np
# import argparse
# import shutil

# # 設定資料夾路徑
# npy_path = "action_npy"
# train_path = "train"
# test_path = "test"

# # 清空 train 和 test 資料夾
# def clear_directory(path):
#     """
#     清空指定目錄。
    
#     參數:
#     - path: str，目錄的路徑。
    
#     功能:
#     - 刪除指定目錄及其內容。如果刪除失敗，則嘗試使用系統命令強制刪除。
#     """
#     if os.path.exists(path):
#         try:
#             shutil.rmtree(path)
#         except PermissionError:
#             print(f"Permission denied: '{path}'. Attempting to use system command.")
#             os.system(f'rd /s /q "{path}"')
#         except Exception as e:
#             print(f"Error deleting directory '{path}': {e}")

# # 定義滑動窗口函數
# def sliding_window(data, window_size, step_size, label):
#     """
#     使用滑動窗口提取數據片段。
    
#     參數:
#     - data: numpy.ndarray，包含骨架資訊的數據。
#     - window_size: int，滑動窗口的大小。
#     - step_size: int，滑動窗口的步長。
#     - label: str，數據的標籤。
    
#     返回:
#     - slices: list，包含數據片段和相應標籤的列表。
#     """

#     num_frames, num_keypoints, num_coords = data.shape
#     slices = []

#     # 使用滑動窗口提取片段
#     for start in range(0, num_frames - window_size + 1, step_size):
#         end = start + window_size
#         slice_data = data[start:end]
#         slices.append((slice_data, label))

#     # 處理剩餘不足的片段
#     if num_frames % step_size != 0 and num_frames > window_size:
#         slice_data = data[-window_size:]
#         slices.append((slice_data, label))

#     return slices

# # 處理每個動作的資料並儲存
# def process_and_save_data(npy_path, train_path, test_path, window_size, step_size=10):
#     """
#     處理每個動作的數據，使用滑動窗口提取片段並保存。
    
#     參數:
#     - npy_path: str，包含原始數據的目錄。
#     - train_path: str，保存處理後數據的目錄。
#     - test_path: str，保存測試數據的目錄。
#     - window_size: int，滑動窗口的大小。
#     - step_size: int，滑動窗口的步長。
#     """
#     labels = os.listdir(npy_path)

#     for label in labels:
#         action_folder = os.path.join(npy_path, label)
#         train_save_folder = os.path.join(train_path, label)
#         test_save_folder = os.path.join(test_path, label)

#         if not os.path.exists(train_save_folder):
#             os.makedirs(train_save_folder)
        
#         if not os.path.exists(test_save_folder):
#             os.makedirs(test_save_folder)

#         files = os.listdir(action_folder)
#         if not files:
#             continue

#         # 選取檔案大小最小的檔案
#         test_file = min(files, key=lambda x: os.path.getsize(os.path.join(action_folder, x)))
#         files.remove(test_file)

#         # 將測試檔案使用滑動窗口技術處理並保存到 test 資料夾
#         test_file_path = os.path.join(action_folder, test_file)
#         test_data = np.load(test_file_path)
#         test_slices = sliding_window(test_data, window_size, step_size, label)
#         for slice_index, (slice_data, _) in enumerate(test_slices):
#             test_save_path = os.path.join(test_save_folder, f"{os.path.splitext(test_file)[0]}_{slice_index}.npy")
#             np.save(test_save_path, slice_data)

#         for file_index, file_name in enumerate(files):
#             file_path = os.path.join(action_folder, file_name)
#             data = np.load(file_path)
#             slices = sliding_window(data, window_size, step_size, label)

#             for slice_index, (slice_data, _) in enumerate(slices):
#                 save_path = os.path.join(train_save_folder, f"{os.path.splitext(file_name)[0]}_{slice_index}.npy")
#                 np.save(save_path, slice_data)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process and save action data with a sliding window.")
#     parser.add_argument("--time_step", type=int, default=70, help="The size of the sliding window.")
#     args = parser.parse_args()
#     # 清空 train 和 test 資料夾
#     clear_directory(train_path)
#     clear_directory(test_path)
#     # 執行處理並儲存資料
#     process_and_save_data(npy_path, train_path, test_path, window_size=args.time_step)


# V3

import os
import numpy as np
import argparse
import shutil
import random

# 設定資料夾路徑
npy_path = "action_npy"
train_path = "train"
test_path = "test"

# 清空 train 和 test 資料夾
def clear_directory(path):
    """
    清空指定目錄。
    
    參數:
    - path: str，目錄的路徑。
    
    功能:
    - 刪除指定目錄及其內容。如果刪除失敗，則嘗試使用系統命令強制刪除。
    """
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
    """
    使用滑動窗口提取數據片段。
    
    參數:
    - data: numpy.ndarray，包含骨架資訊的數據。
    - window_size: int，滑動窗口的大小。
    - step_size: int，滑動窗口的步長。
    - label: str，數據的標籤。
    
    返回:
    - slices: list，包含數據片段和相應標籤的列表。
    """

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
def process_and_save_data(npy_path, train_path, test_path, window_size, step_size=10):
    """
    處理每個動作的數據，使用滑動窗口提取片段並保存。
    
    參數:
    - npy_path: str，包含原始數據的目錄。
    - train_path: str，保存處理後數據的目錄。
    - test_path: str，保存測試數據的目錄。
    - window_size: int，滑動窗口的大小。
    - step_size: int，滑動窗口的步長。
    """
    labels = os.listdir(npy_path)

    for label in labels:
        action_folder = os.path.join(npy_path, label)
        train_save_folder = os.path.join(train_path, label)
        test_save_folder = os.path.join(test_path, label)

        if not os.path.exists(train_save_folder):
            os.makedirs(train_save_folder)
        
        if not os.path.exists(test_save_folder):
            os.makedirs(test_save_folder)

        files = os.listdir(action_folder)
        if not files:
            continue

        test_file = random.choice(files)
        files.remove(test_file)

        # 將測試檔案使用滑動窗口技術處理並保存到 test 資料夾
        test_file_path = os.path.join(action_folder, test_file)
        test_data = np.load(test_file_path)
        test_slices = sliding_window(test_data, window_size, step_size, label)
        for slice_index, (slice_data, _) in enumerate(test_slices):
            test_save_path = os.path.join(test_save_folder, f"{os.path.splitext(test_file)[0]}_{slice_index}.npy")
            np.save(test_save_path, slice_data)

        for file_index, file_name in enumerate(files):
            file_path = os.path.join(action_folder, file_name)
            data = np.load(file_path)
            slices = sliding_window(data, window_size, step_size, label)

            for slice_index, (slice_data, _) in enumerate(slices):
                save_path = os.path.join(train_save_folder, f"{os.path.splitext(file_name)[0]}_{slice_index}.npy")
                np.save(save_path, slice_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save action data with a sliding window.")
    parser.add_argument("--time_step", type=int, default=70, help="The size of the sliding window.")
    args = parser.parse_args()
    # 清空 train 和 test 資料夾
    clear_directory(train_path)
    clear_directory(test_path)
    # 執行處理並儲存資料
    process_and_save_data(npy_path, train_path, test_path, window_size=args.time_step)