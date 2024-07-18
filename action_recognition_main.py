import os
import contextlib
import torch
import cv2
import time
import pickle
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
from torch import nn
from rich.progress import Progress
from insightface.app import FaceAnalysis
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
from ultralytics import YOLO
import subprocess
import json
import argparse

from PostureRecognition.Posture_ensemble_model import PostureLSTMModel, PostureGRUModel, PostureTemporalConvNet, PostureEnsembleModel

# 忽略警告
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def get_video_metadata(video_path):
    try:
        result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        metadata = json.loads(result.stdout)
        creation_time = metadata['format']['tags'].get('creation_time', None)
        return creation_time
    except Exception as e:
        print(e)
        return None

def main(video_path):
    time_steps = 70  # 固定 time_steps

    # 加載視頻
    creation_time_str = get_video_metadata(video_path)

    if creation_time_str:
        start_time = datetime.fromisoformat(creation_time_str.replace('Z', '+00:00'))
    else:
        print("Could not retrieve creation time from metadata.")
        start_time = datetime.now()  # fallback to current time

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 初始化人臉檢測和編碼模型
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        app = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))

    # 初始化 DeepSort
    deepsort = DeepSort(
        max_age=50,
        nn_budget=100,
        max_iou_distance=0.7,
        n_init=1
    )

    # 初始化 MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=0)

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 設置參數
    input_dim = 33 * 4  # 33 keypoints * 4 coords (xyzv)
    hidden_dim = 128
    num_layers = 2
    num_channels = [128] * num_layers  # TCN 模型的通道數量

    # 加載姿態標籤編碼器
    with open(r'PostureRecognition\Model\posture_label_encoder.pkl', 'rb') as f:
        posture_label_encoder = pickle.load(f)
    num_classes = len(posture_label_encoder.classes_)

    # 加載動作識別模型
    posture_ensemble_checkpoint = torch.load(r'PostureRecognition\Model\Posture_ensemble_model.pth')

    posture_lstm_model = PostureLSTMModel(input_dim, hidden_dim, num_layers, num_classes).to(device)
    posture_gru_model = PostureGRUModel(input_dim, hidden_dim, num_layers, num_classes).to(device)
    posture_tcn_model = PostureTemporalConvNet(input_dim, num_channels, num_classes).to(device)
    posture_ensemble_model = PostureEnsembleModel(posture_lstm_model, posture_gru_model, posture_tcn_model, num_classes).to(device)

    posture_lstm_model.load_state_dict(posture_ensemble_checkpoint['lstm_model_state_dict'])
    posture_gru_model.load_state_dict(posture_ensemble_checkpoint['gru_model_state_dict'])
    posture_tcn_model.load_state_dict(posture_ensemble_checkpoint['tcn_model_state_dict'])
    posture_ensemble_model.load_state_dict(posture_ensemble_checkpoint['ensemble_model_state_dict'])

    posture_lstm_model.eval()
    posture_gru_model.eval()
    posture_tcn_model.eval()
    posture_ensemble_model.eval()

    # 用於存儲每個 ID 的骨架數據隊列
    skeleton_sequences = defaultdict(lambda: deque(maxlen=time_steps))

    # 用於存儲骨架數據
    skeleton_data = []
    # 用於存儲人臉 ID 和特徵
    face_id_data = []
    # 用於存儲每個 ID 的第一張灰階照片
    id_first_gray_images = {}

    # 初始化 YOLOv8
    model = YOLO('yolov8n.pt')

    # 進度條
    with Progress() as progress:
        task = progress.add_task("[red]Processing Video...", total=total_frames)

        success, frame = cap.read()
        frame_count = 0

        while success:
            # 使用 YOLOv8 進行人體檢測
            with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yolo_results = model(frame, verbose=False)  # 禁用 YOLOv8 的推理輸出

            detections = []
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = box.cls[0]
                    if cls == 0:
                        detections.append([[x1, y1, x2-x1, y2-y1], conf])

            # 更新追踪器
            tracks = deepsort.update_tracks(detections, frame=frame)

            # 檢測每個追踪到的對象
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # 確保邊界框有效
                if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
                    continue

                person = frame[y1:y2, x1:x2]

                # 確保圖像非空
                if person.size == 0:
                    continue

                # 保存第一次檢測到的灰階照片
                if track_id not in id_first_gray_images:
                    gray_person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
                    id_first_gray_images[track_id] = gray_person

                # 偵測人體骨架
                results = pose.process(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))

                predicted_action = "unknown"
                confidence_text = ""

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    keypoints = [(lmk.x, lmk.y, lmk.z, lmk.visibility) for lmk in landmarks]
                    skeleton_sequences[track_id].append(keypoints)

                    # 當累積滿 time_steps 個骨架時，進行動作識別
                    if len(skeleton_sequences[track_id]) == time_steps:
                        skeleton_array = np.array(skeleton_sequences[track_id])

                        # 以 hip 為中心進行標準化
                        keypoints_tensor = torch.tensor(skeleton_array, dtype=torch.float32).to(device)
                        hip_x, hip_y, hip_z = keypoints_tensor[:, mp_pose.PoseLandmark.LEFT_HIP.value, :3].mean(dim=0)
                        keypoints_centered = keypoints_tensor - torch.tensor([hip_x, hip_y, hip_z, 0.0]).to(device)

                        skeleton_input_data = keypoints_centered.reshape(time_steps, -1).unsqueeze(0).to(device)

                        with torch.no_grad():
                            posture_output = posture_ensemble_model(skeleton_input_data)

                            # 獲取模型的輸出概率
                            softmax = nn.Softmax(dim=1)
                            posture_prob = softmax(posture_output).cpu().numpy()

                            # 獲取總概率分數最高的類別
                            predicted_class = np.argmax(posture_prob, axis=1)[0]
                            predicted_action = posture_label_encoder.inverse_transform([predicted_class])[0]

                            # 計算信心
                            confidence = posture_prob[0, predicted_class]
                            confidence_text = f'{confidence:.2f}'

                        # 如果信心分數低於 50，顯示未知
                        if confidence < 0.9:
                            predicted_action = "unknown"
                            confidence_text = ""

                        # 計算當前幀的時間
                        frame_time = start_time + timedelta(seconds=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                        frame_time_str = frame_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                        skeleton_data.append((frame_time_str, track_id, predicted_action))

                # 檢測每個追踪到的對象的人臉
                with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    faces = app.get(person)

                for face in faces:
                    face_feature = face.normed_embedding
                    face_id_data.append((track_id, face_feature.tolist()))

            progress.update(task, advance=1)
            success, frame = cap.read()
            frame_count += 1

    cap.release()
    # 儲存ID、時間和動作數據
    with open(r'IDdata/id_time_action.txt', 'w') as f:
        for record in skeleton_data:
            f.write(f"{record[1]}, {record[0]}, {record[2]}\n")

    # 儲存人臉ID和特徵數據
    with open(r'IDdata/face_id_data.pkl', 'wb') as f:
        pickle.dump(face_id_data, f)

    # 儲存第一次檢測到的灰階照片數據
    with open(r'IDdata/id_first_gray_images.pkl', 'wb') as f:
        pickle.dump(id_first_gray_images, f)

    # 結束
    print("Processing complete.")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Action Recognition Main Program')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()  # 解析命令行參數
    main(args.video_path)  # 將解析的參數傳遞給 main 函數
