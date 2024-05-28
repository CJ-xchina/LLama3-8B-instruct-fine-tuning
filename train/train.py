import os
import subprocess
import sys

from utils.build_dataset import split_data

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from iscasmodel.core import ModelUtils

if torch.cuda.is_available():
    print("CUDA is available", flush=True)
    num_devices = torch.cuda.device_count()
    print("Number of CUDA devices: ", num_devices, flush=True)

    for i in range(num_devices):
        print("Device ", i, ": ", torch.cuda.get_device_name(i), flush=True)
else:
    print("CUDA is not available", flush=True)
    num_devices = 1

# huggingface_hub.login("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", add_to_git_credential=True)
# 读取数据并划分训练集与验证集
model_utils = ModelUtils()
base_resource_path = os.path.join(model_utils.get_dataset_path(), "k8s")
train_path, valid_file_path = split_data(base_resource_path, 0.05)

# 训练配置
QUANTIZATION = 16  # DEFINE QUANTIZATION HERE. Choose from (16 | 8 | 4)
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
learning_rate = 2e-4
gradient_accumulation_steps = 8
max_seq_length = 512
model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # model's huggingface id
deepspeed_config_file = "ds_zero2_no_offload.json"
additional_path = "models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873"
tokenizer_path = model_path = os.path.join(model_utils.get_model_file_path(), "final_model_hug")  # 模型路径 =
# tokenizer_path = model_path = model_utils.get_model_file_path()
# tokenizer_path = model_path = '/result/Mistral-7B-final-v4'
# output_dir = "/result/LLama3-8B/lora"  # lora模型输出路径
output_dir = model_utils.get_model_output_path()  # 合并后模型输出路径

print(f"model_path is : {model_path}")
print(f"output dir is : {output_dir}")


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


list_files_and_directories(start_path=model_path)

# lora 配置
lr = 1e-4
lora_rank = 64
lora_alpha = 128
lora_trainable = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save = "embed_tokens,lm_head"
lora_dropout = 0.1

# 下载数据集
# command = "python"
# script_path = "../script/download_net_dataset.py"
# full_command = [command, script_path]
# result = subprocess.run(full_command, check=True, text=True)

# train_dataset, validate_dataset = preprocess_data(dataset_path, dataset_name, tokenizer)

ips = model_utils.get_ips()
print(ips)
model_utils.get_name()
master_addr = ips[0]

full_command = [
    f"{sys.executable}",
    "-m",
    "torch.distributed.run",
    f"--nnodes", f"{len(ips)}",
    f"--nproc_per_node", f"{num_devices}",
    f"--node_rank", f"{ips.index(model_utils.get_name())}",
    f"--master_addr", f"{master_addr}",
    "--master_port", "8080",
    "run_clm_sft_with_peft.py",
    f"--deepspeed", f"{deepspeed_config_file}",
    f"--model_name_or_path", f"{model_path}",
    f"--tokenizer_name_or_path", f"{tokenizer_path}",
    f"--dataset_dir", f"{train_path}",
    f"--per_device_train_batch_size", f"{per_device_train_batch_size}",
    f"--per_device_eval_batch_size", f"{per_device_eval_batch_size}",
    "--do_train",
    "--do_eval",
    f"--seed", "42",
    "--fp16",
    "--num_train_epochs", "1",
    "--lr_scheduler_type", "cosine",
    f"--learning_rate", f"{learning_rate}",
    "--warmup_ratio", "0.03",
    "--weight_decay", "0",
    "--logging_strategy", "steps",
    "--logging_steps", "10",
    "--save_strategy", "steps",
    "--save_total_limit", "5",
    "--evaluation_strategy", "steps",
    "--eval_steps", "200",
    "--save_steps", "200",
    f"--gradient_accumulation_steps", f"{gradient_accumulation_steps}",
    f"--preprocessing_num_workers", "8",
    f"--max_seq_length", f"{max_seq_length}",
    f"--output_dir", f"{output_dir}",
    "--overwrite_output_dir",
    "--ddp_timeout", "30000",
    "--logging_first_step", "True",
    f"--lora_rank", f"{lora_rank}",
    f"--lora_alpha", f"{lora_alpha}",
    f"--trainable", f"{lora_trainable}",
    f"--lora_dropout", f"{lora_dropout}",
    f"--modules_to_save", f"{modules_to_save}",
    "--torch_dtype", "float16",
    f"--validation_file", f"{valid_file_path}",
    "--load_in_kbits", f"{QUANTIZATION}",
    "--save_safetensors", "False",
    "--gradient_checkpointing",
    "--ddp_find_unused_parameters", "False"
]

try:
    subprocess.run(full_command, check=True, text=True)
except subprocess.CalledProcessError as e:
    print("Error occurred:", e)
    print("Return code:", e.returncode)
    print("Command run:", ' '.join(e.cmd))
    print("Error output:", e.stderr)
