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


def copy_directory_contents(src_dir, dst_dir=None):
    """
    Copies all the files from the source directory to a new directory.

    Args:
    src_dir (str): The path to the source directory.
    dst_dir (str): The path to the destination directory. If None, it defaults to src_dir + 'copy'.

    Returns:
    None
    """
    # 如果没有提供目标目录，则创建一个默认的复制目录
    if dst_dir is None:
        dst_dir = src_dir + "copy"

    # 检查源目录是否存在
    if not os.path.exists(src_dir):
        raise ValueError("Source directory does not exist")

    # 创建目标目录
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 遍历源目录及其所有子目录
    for root, dirs, files in os.walk(src_dir):
        # 计算目标目录路径
        destination_path = root.replace(src_dir, dst_dir, 1)

        # 确保目标路径的目录存在
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        # 复制每个文件到新目录
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(destination_path, file)
            shutil.copy(src_file, dst_file)


# 定义命令和参数
command = "python"
script_path = "../script/mistral_api_server.py"

base_model_option = "--base_model"
lora_model_option = "--lora_model"
base_model = "/result/Mistral-7B-instruct-v4/final_model"

copy_directory_contents('/result','/result_c')
list_files_and_directories('/result')

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

list_files_and_directories('/result')