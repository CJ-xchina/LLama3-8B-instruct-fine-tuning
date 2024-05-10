import os
import shutil
import subprocess

import torch
from iscasmodel.core import ModelUtils

model_utils = ModelUtils()


def list_files_and_directories(start_path, indent=0):
    prefix = ' ' * indent
    if indent == 0:
        print(f"{prefix}根目录: {start_path}")
    else:
        print(f"{prefix}目录: {start_path}")

    for item in os.listdir(start_path):
        path = os.path.join(start_path, item)
        if os.path.isdir(path):
            list_files_and_directories(path, indent + 4)  # 增加缩进显示子目录
        else:
            print(f"{prefix}    文件: {item}")



# 定义命令和参数
command = "python"
script_path = "../script/llama3_api_server.py"

base_model_option = "--base_model"
lora_model_option = "--lora_model"
base_model = os.path.join(model_utils.get_model_output_path(), "final_model")


gpus_option = "--gpus"

if torch.cuda.is_available():
    print("CUDA is available")
    num_devices = torch.cuda.device_count()
    gpus = ','.join(map(str, range(num_devices)))
    print("Number of CUDA devices: ", num_devices)

    for i in range(num_devices):
        print("Device ", i, ": ", torch.cuda.get_device_name(i))
else:
    gpus = ""

full_command = [command, script_path, base_model_option, base_model]
if gpus:
    full_command.extend([gpus_option, gpus])

# 执行命令
try:
    result = subprocess.run(full_command, check=True, text=True)
    print("命令输出:", result.stdout)
except subprocess.CalledProcessError as e:
    print("命令执行出错:", e.stderr)
