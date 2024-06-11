import os
import subprocess
import sys

import yaml

from utils.build_dataset import split_data

# 读取配置文件
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# 设置环境变量
os.environ['HF_ENDPOINT'] = config['hf_endpoint']
os.environ['CUDA_LAUNCH_BLOCKING'] = config['cuda_launch_blocking']

import torch
from iscasmodel.core import ModelUtils

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available", flush=True)
    num_devices = torch.cuda.device_count()
    print("Number of CUDA devices: ", num_devices, flush=True)

    for i in range(num_devices):
        print("Device ", i, ": ", torch.cuda.get_device_name(i), flush=True)
else:
    print("CUDA is not available", flush=True)
    num_devices = 1

# 读取数据并划分训练集与验证集
model_utils = ModelUtils()
dataset_additional_path = config['dataset_additional_path']
dataset_split_ratio = config['dataset_split_ratio']
base_resource_path = os.path.join(model_utils.get_dataset_path(), dataset_additional_path)
train_path, valid_file_path = split_data(base_resource_path, dataset_split_ratio)

# 训练配置
QUANTIZATION = config['quantization']
per_device_train_batch_size = config['per_device_train_batch_size']
per_device_eval_batch_size = config['per_device_eval_batch_size']
learning_rate = config['learning_rate']
gradient_accumulation_steps = config['gradient_accumulation_steps']
max_seq_length = config['max_seq_length']
model_id = config['model_id']
deepspeed_config_file = config['deepspeed_config_file']
model_additional_path = config['model_additional_path']
tokenizer_path = model_path = os.path.join(model_utils.get_model_file_path(), model_additional_path)
output_dir = model_utils.get_model_output_path()

print(f"model_path is : {model_path}")
print(f"output dir is : {output_dir}")

# lora 配置
lora_params = config['lora']
lr = lora_params['lr']
lora_rank = lora_params['rank']
lora_alpha = lora_params['alpha']
lora_trainable = lora_params['trainable']
modules_to_save = lora_params['modules_to_save']
lora_dropout = lora_params['dropout']

# 下载数据集
# command = "python"
# script_path = "../script/download_net_dataset.py"
# full_command = [command, script_path]
# result = subprocess.run(full_command, check=True, text=True)

# 获取分布式训练参数
ips = model_utils.get_ips()
print(ips)
model_utils.get_name()
master_addr = ips[0]

# 从配置文件中读取训练参数
training_params = config['training_params']
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
    f"--seed", f"{training_params['seed']}",
    "--fp16" if training_params['fp16'] else "",
    f"--num_train_epochs", f"{training_params['num_train_epochs']}",
    f"--lr_scheduler_type", f"{training_params['lr_scheduler_type']}",
    f"--learning_rate", f"{learning_rate}",
    f"--warmup_ratio", f"{training_params['warmup_ratio']}",
    f"--weight_decay", f"{training_params['weight_decay']}",
    f"--logging_strategy", f"{training_params['logging_strategy']}",
    f"--logging_steps", f"{training_params['logging_steps']}",
    f"--save_strategy", f"{training_params['save_strategy']}",
    f"--save_total_limit", f"{training_params['save_total_limit']}",
    f"--evaluation_strategy", f"{training_params['evaluation_strategy']}",
    f"--eval_steps", f"{training_params['eval_steps']}",
    f"--save_steps", f"{training_params['save_steps']}",
    f"--gradient_accumulation_steps", f"{gradient_accumulation_steps}",
    f"--preprocessing_num_workers", f"{training_params['preprocessing_num_workers']}",
    f"--max_seq_length", f"{max_seq_length}",
    f"--output_dir", f"{output_dir}",
    "--overwrite_output_dir",
    f"--ddp_timeout", f"{training_params['ddp_timeout']}",
    "--logging_first_step" if training_params['logging_first_step'] else "",
    f"--lora_rank", f"{lora_rank}",
    f"--lora_alpha", f"{lora_alpha}",
    f"--trainable", f"{lora_trainable}",
    f"--lora_dropout", f"{lora_dropout}",
    f"--modules_to_save", f"{modules_to_save}",
    f"--torch_dtype", f"{training_params['torch_dtype']}",
    f"--validation_file", f"{valid_file_path}",
    "--load_in_kbits", f"{QUANTIZATION}",
    "--save_safetensors" f"{training_params['save_safetensors']}",
    f"--gradient_checkpointing", f"{training_params['gradient_checkpointing']}",
    "--ddp_find_unused_parameters" f"{training_params['ddp_find_unused_parameters']}"
]

try:
    subprocess.run(full_command, check=True, text=True)
except subprocess.CalledProcessError as e:
    print("Error occurred:", e)
    print("Return code:", e.returncode)
    print("Command run:", ' '.join(e.cmd))
    print("Error output:", e.stderr)
