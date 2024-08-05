import os
import subprocess
import re
import numpy as np
import pandas as pd
import sys 

# 定義要測試的 time_step 值範圍
time_step_values = [50,70,90,110,130,150]  # 示例範圍

# 輸出文件的路徑
output_file_path = "output.txt"

# 定義結果保存的 CSV 文件路徑
csv_file_path = "experiment_results.csv"

# 創建一個空的 DataFrame 並保存為 CSV
columns = [
    "time_step", "run", "Overall",
    "Bent_Leg_Kickback", "Bent_Over_Row", "Bird_Dog", "Crunch_Floor", 
    "Deadlift", "Dead_Bug", "Farmers_walk", "Front_Plank", 
    "glute-bridge", "Kettlebell_Strict_Press", "Mountain_Climber", 
    "Prone", "push_up", "Resistance_Band_Side_Walk", 
    "Russian_Twist", "Squat"
]
if not os.path.exists(csv_file_path):
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_file_path, index=False)

# 初始化結果字典
results = {}

# 定義運行腳本的函數
def run_script(command):
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=sys.stdout, 
            stderr=sys.stderr, 
            text=True
        )
        process.communicate()  # 確保輸出被正確捕獲並顯示
        if process.returncode != 0:
            print(f"運行命令出錯: {command}")
            return False
        return True
    except Exception as e:
        print(f"運行命令出錯: {command}")
        print(e)
        return False

# 從 output.txt 解析準確率
def parse_accuracy(output):
    accuracies = {}
    lines = output.splitlines()
    for line in lines:
        # 允許包含連字符的標簽
        match = re.match(r'Accuracy for action ([\w-]+): (\d+\.\d+)%', line)
        overall_match = re.match(r'Overall Accuracy of the model on the test set: (\d+\.\d+)%', line)
        if match:
            action, accuracy = match.groups()
            accuracies[action] = float(accuracy)
        elif overall_match:
            accuracies['Overall'] = float(overall_match.group(1))
    return accuracies

# 遍歷每個 time_step 值
for time_step in time_step_values:
    print(f"Testing with time_step = {time_step}\n")
    
    time_step_results = []
    
    for test_run in range(3):  # 每個 time_step 重複三次
        print(f"Run {test_run + 1} for time_step = {time_step}\n")
        
        # 定義腳本命令
        scripts = [
            "python Update_action_npy.py",
            f"python Update_train.py --time_step={time_step}",
            "python Generate_training_data.py",
            "python AugmentData.py",
            f"python TrainEnsembleModel_V3.py --time_step={time_step}"
        ]
        
        # 順序運行每個腳本
        for command in scripts:
            success = run_script(command)
            if not success:
                print("由於錯誤，停止進一步執行。")
                break
        
        # 讀取並記錄 output.txt 的輸出
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r') as file:
                output_content = file.read()
                accuracies = parse_accuracy(output_content)
                time_step_results.append(accuracies)
                
                # 構建行數據並追加到 CSV
                row_data = {'time_step': time_step, 'run': test_run + 1}
                row_data.update(accuracies)
                
                df = pd.DataFrame([row_data], columns=columns)
                df.to_csv(csv_file_path, mode='a', header=False, index=False)
                
                print(f"已記錄 time_step {time_step} run {test_run + 1} 的輸出。\n")
        else:
            print(f"time_step {time_step} 的輸出文件未找到。\n")

    # 記錄所有結果
    results[time_step] = time_step_results
    
    # 計算並記錄平均準確率
    overall_accuracies = [res['Overall'] for res in time_step_results if 'Overall' in res]
    average_overall_accuracy = np.mean(overall_accuracies) if overall_accuracies else 0.0
    
    # 添加平均值到 CSV
    average_row = {'time_step': time_step, 'run': 'average'}
    average_row['Overall'] = average_overall_accuracy
    
    for action in columns[3:]:
        action_accuracies = [res.get(action, 0.0) for res in time_step_results]
        average_row[action] = np.mean(action_accuracies)
    
    df_avg = pd.DataFrame([average_row], columns=columns)
    df_avg.to_csv(csv_file_path, mode='a', header=False, index=False)
    
    print(f"Average Overall Accuracy for time_step {time_step}: {average_overall_accuracy:.2f}%\n")

# 顯示記錄的結果
for time_step, time_step_results in results.items():
    if isinstance(time_step_results, list):
        for idx, result in enumerate(time_step_results):
            print(f"Results for time_step {time_step}, test {idx + 1}:")
            for action, accuracy in result.items():
                print(f"{action}: {accuracy:.2f}%")
            print()
    else:
        print(f"Average Overall Accuracy for time_step {time_step}: {time_step_results:.2f}%\n")
