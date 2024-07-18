import os
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
import time


# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

# 定義路徑
input_dir = 'action'
output_dir = 'action_npy'

# 如果輸出目錄不存在則創建
os.makedirs(output_dir, exist_ok=True)

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    total_time = 0
    frame_count = 0
    
    time_read = 0
    time_convert = 0
    time_infer = 0

    while cap.isOpened():
        start_read = time.time()
        ret, frame = cap.read()
        end_read = time.time()
        time_read += (end_read - start_read)
        
        if not ret:
            break
        
        start_convert = time.time()
        # 將 BGR 圖像轉換為 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        end_convert = time.time()
        time_convert += (end_convert - start_convert)

        # 計時開始
        start_infer = time.time()
        results = pose.process(image)
        # 計時結束
        end_infer = time.time()
        time_infer += (end_infer - start_infer)
        
        inference_time = end_infer - start_infer
        total_time += inference_time
        frame_count += 1

        if results.pose_landmarks:
            keypoints = []
            landmarks = results.pose_landmarks.landmark

            # 取得左髖部和右髖部的座標
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            # 計算hip中心點
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            hip_center_z = (left_hip.z + right_hip.z) / 2

            for lm in landmarks:
                keypoints.append([
                    lm.x - hip_center_x,
                    lm.y - hip_center_y,
                    lm.z - hip_center_z,
                    lm.visibility
                ])
            frames_data.append(keypoints)

    cap.release()
    np.save(output_path, np.array(frames_data))
    print(f'Saved: {output_path}')
    
    average_time_per_frame = total_time / frame_count if frame_count > 0 else 0
    # print(f"Time spent reading frames: {time_read:.4f} seconds")
    # print(f"Time spent converting frames: {time_convert:.4f} seconds")
    # print(f"Time spent on inference: {time_infer:.4f} seconds")
    return average_time_per_frame

def is_new_file(video_path, npy_path):
    if not os.path.exists(npy_path):
        return True
    video_time = os.path.getmtime(video_path)
    npy_time = os.path.getmtime(npy_path)
    return video_time > npy_time

# 檢查並處理視頻文件
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
    TextColumn("[progress.completed] {task.completed}/{task.total}")
) as progress:
    tasks = []
    for action in os.listdir(input_dir):
        action_path = os.path.join(input_dir, action)
        if os.path.isdir(action_path):
            output_action_dir = os.path.join(output_dir, action)
            os.makedirs(output_action_dir, exist_ok=True)

            for video_file in os.listdir(action_path):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(action_path, video_file)
                    output_file_path = os.path.join(output_action_dir, f"{os.path.splitext(video_file)[0]}.npy")
                    
                    # 檢查對應的 .npy 文件是否需要更新
                    if is_new_file(video_path, output_file_path):
                        tasks.append((video_path, output_file_path))

    if tasks:
        total_videos = len(tasks)
        task = progress.add_task("[green]Initializing...", total=total_videos)
        total_inference_time = 0
        total_frames = 0

        for i, (video_path, output_file_path) in enumerate(tasks):
            progress.update(task, description=f"Processing {video_path}", advance=1)
            average_time_per_frame = process_video(video_path, output_file_path)
            total_inference_time += average_time_per_frame
            total_frames += 1

        if total_frames > 0:
            overall_average_time = total_inference_time / total_frames
            print(f"Average inference time per frame: {overall_average_time:.4f} seconds")
        
        # 確保所有視頻文件都處理完並保存
        progress.update(task, description="Processing complete.", completed=total_videos)
    else:
        print("沒有可以更新的檔案")
