import os
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description="Run a series of scripts with a specified time_step.")
parser.add_argument("--time_step", type=int, default=70, help="The size of the time step for the sliding window.")
args = parser.parse_args()

# 定義腳本列表和順序
scripts = [
    "Update_action_npy.py",
    ("Update_train.py", args.time_step),
    "Generate_training_data.py",
    "AugmentData.py",
    ("TrainEnsembleModel.py", args.time_step)
]

# 按順序運行每個腳本
for script in scripts:
    if isinstance(script, tuple):
        script_name, time_step = script
        command = f"python {script_name} --time_step={time_step}"
    else:
        command = f"python {script}"

    print(f"Running {command}...")
    exit_code = os.system(command)
    if exit_code != 0:
        print(f"Error running {script}")
        break
    print(f"{script} completed successfully.\n")
