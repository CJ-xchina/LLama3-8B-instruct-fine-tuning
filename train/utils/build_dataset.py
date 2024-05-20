import json
import os
import random

import datasets
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


def build_dataset(data_path,
                  tokenizer,
                  max_seq_length: int, data_cache_dir=None,
                  preprocessing_num_workers=None,
                  ):
    def tokenize(batch):
        # 处理整个批次的数据
        new_prompts = [
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a serious expert in the field of Kubernetes and use what you have learnt to be able to ask and answer all Kubernetes questions in an expert tone!<|eot_id|>" \
            f"<|start_header_id|>user<|end_header_id|> {input_item} <|eot_id|> " \
            f"<|start_header_id|>assistant<|end_header_id> {output_item}<|eot_id|>"

            for input_item, output_item in zip(batch['input'], batch['output'])
        ]

        return tokenizer(
            new_prompts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True
        )

    all_datasets = []

    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]
    for file in data_path:
        try:
            processed_dataset = datasets.load_from_disk('./')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file)
            tokenized_dataset = raw_dataset.map(
                tokenize,
                batched=True,  # 确保批处理模式开启
                remove_columns=["input", "output"],
                desc="preprocessing on dataset"
            )
            processed_dataset = tokenized_dataset
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    print(f"inputs_ids 长度:{processed_dataset['train'][0]['input_ids']}")
    print(f"inputs_ids 长度:{processed_dataset['train'][0]['attention_mask']}")
    print(f"数据长度:{len(processed_dataset['train'])}")
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets


# def build_dataset(data_path,
#                   tokenizer,
#                   max_seq_length: int,
#                   data_cache_dir=None,
#                   preprocessing_num_workers=None,
#                   ):
#     begin_of_text_id = tokenizer.get_vocab()["<|begin_of_text|>"]
#     start_header_id = tokenizer.get_vocab()["<|start_header_id|>"]
#     end_header_id = tokenizer.get_vocab()["<|end_header_id|>"]
#     eot_id = tokenizer.get_vocab()["<|eot_id|>"]
#     nl_tokens = tokenizer('\n').input_ids
#     _system = tokenizer('system').input_ids
#     _user = tokenizer('user').input_ids
#     _assistant = tokenizer('assistant').input_ids
#     system_message = "You are a serious expert in the field of Kubernetes and use what you have learnt to be able to ask and answer all Kubernetes questions in an expert tone!"
#
#     def tokenize(batch):
#         # 创建一个空列表来存储标记化后的结果
#         tokenized_list = []
#         # 创建一个空列表来存储对应的attention_mask结果
#         attention_mask_list = []
#         for input_item, output_item in zip(batch['input'], batch['output']):
#             input_id = []
#             input_mask = []
#             system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(
#                 system_message).input_ids + [eot_id]
#
#             # 添加系统提示词
#             input_id += system
#             input_mask += [1] * len(input_id)
#
#             # 添加用户提问
#             input_id += [start_header_id] + _user + [end_header_id] + nl_tokens + tokenizer(input_item).input_ids + [
#                 eot_id]
#
#             # 生成截断前的系统回复
#
#             _input_id = input_id + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(
#                 output_item).input_ids + [
#                             eot_id]
#
#             # 需要截断系统回复
#             if len(_input_id) > max_seq_length:
#                 truncate_num = len(_input_id) - max_seq_length
#
#                 # 截断系统的回复
#                 input_id += [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(
#                     output_item).input_ids[:-truncate_num] + [eot_id]
#
#             padding_length = 0
#             origin_len = len(encoded_inputs)
#             # 检查是否需要截断
#             if origin_len > max_seq_length:
#                 encoded_inputs = encoded_inputs[:max_seq_length]
#                 encoded_inputs[-1] = end_of_text_id
#                 attention_mask = [1] * max_seq_length
#
#             # 需要填充
#             else:
#                 # 计算需要填充的个数
#                 padding_length = max_seq_length - len(encoded_inputs)
#                 # 将填充的列表追加到encoded_inputs的末尾
#                 encoded_inputs += [pad_token_id] * padding_length
#
#                 attention_mask = [1] * origin_len + [0] * padding_length
#
#             tokenized_tensor = torch.tensor(encoded_inputs)
#             attention_tensor = torch.tensor(attention_mask, dtype=torch.long)
#             tokenized_list.append(tokenized_tensor)
#             attention_mask_list.append(attention_tensor)
#
#         # 将所有标记化后的序列转换为Dataset
#         tokenized_dataset = Dataset.from_dict({
#             'input_ids': torch.stack(tokenized_list),
#             'attention_mask': torch.stack(attention_mask_list)  # 这里留空，稍后填充
#         })
#
#         # 更新原始批次字典，直接返回tokenized_dataset
#         batch['input_ids'] = tokenized_dataset['input_ids']
#         batch['attention_mask'] = tokenized_dataset['attention_mask']
#         return batch  # 直接返回修改后的批次字典
#
#     all_datasets = []
#
#     if not isinstance(data_path, (list, tuple)):
#         data_path = [data_path]
#     for file in data_path:
#         raw_dataset = load_dataset("json", data_files=file)
#         tokenized_dataset = raw_dataset.map(
#             tokenize,
#             batched=True,
#             remove_columns=["input", "output"],
#             desc="Preprocessing dataset"
#         )
#         all_datasets.append(tokenized_dataset['train'])
#
#     # 合并所有数据集
#     all_datasets = concatenate_datasets(all_datasets)
#
#     return all_datasets


# def build_dataset(data_path,
#                   tokenizer,
#                   max_seq_length: int, data_cache_dir=None,
#                   preprocessing_num_workers=None,
#                   ):
#     def tokenize(batch):
#         new_prompts = []
#         for input_item, output_item in zip(batch['input'], batch['output']):
#             # 这里应该调用 tokenizer.apply_chat_template 并传递 input_item 和 output_item
#             new_prompt = tokenizer.apply_chat_template(
#                 [
#                     {"role": "user", "content": f"Who are you? What are your duties?"},
#                     {"role": "assistant",
#                      "content": "I'm an expert in Kubernetes, so I'll give you a detailed solution to your Kubernetes question or request, and I'll try to be as concise as possible, and I'll search for as much information as I know to complete the answer to your question!"},
#                     {"role": "user", "content": f"{input_item}"},
#                     {"role": "assistant", "content": f"{output_item}"}
#                 ],
#                 truncation=True,
#                 max_length=max_seq_length,
#                 padding=True,
#                 return_tensors="pt",
#             )
#             print(new_prompt)
#             new_prompts.append(new_prompt)
#
#         batch['input_ids'] = new_prompts
#         return batch
#
#     all_datasets = []
#
#     if not isinstance(data_path, (list, tuple)):
#         data_path = [data_path]
#     for file in data_path:
#         try:
#             processed_dataset = datasets.load_from_disk('./')
#         except Exception:
#             raw_dataset = load_dataset("json", data_files=file)
#             tokenized_dataset = raw_dataset.map(
#                 tokenize,
#                 batched=True,  # 确保批处理模式开启
#                 remove_columns=["input", "output"],
#                 desc="preprocessing on dataset"
#             )
#             processed_dataset = tokenized_dataset
#         processed_dataset.set_format('torch')
#         all_datasets.append(processed_dataset['train'])
#     print(f"inputs_ids 举例1：{len(processed_dataset['train'][0]['input_ids'])}")
#     print(f"inputs_ids 举例2：{len(processed_dataset['train'][1]['input_ids'])}")
#     print(f"数据长度:{len(processed_dataset['train'])}")
#     all_datasets = concatenate_datasets(all_datasets)
#     return all_datasets


def split_data(dataset_dir, split_ratio):
    # 删除已存在的 train 和 eval 目录并重新创建
    train_dir = os.path.join(dataset_dir, 'train')
    eval_dir = os.path.join(dataset_dir, 'eval')
    eval_file_path = os.path.join(eval_dir, 'eval.json')

    if os.path.exists(train_dir):
        for file in os.listdir(train_dir):
            os.remove(os.path.join(train_dir, file))
        os.rmdir(train_dir)
    if os.path.exists(eval_dir):
        for file in os.listdir(eval_dir):
            os.remove(os.path.join(eval_dir, file))
        os.rmdir(eval_dir)

    # 构造目录
    os.makedirs(train_dir)
    os.makedirs(eval_dir)

    # 准备写入验证集的文件
    eval_file = open(eval_file_path, 'w', encoding='utf-8')

    # 遍历原数据目录中的所有json文件
    total_train_count = 0
    total_eval_count = 0
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.json') and not filename.startswith('.'):
            file_path = os.path.join(dataset_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = [json.loads(line) for line in file.readlines()]

            # 根据指定的分割比例划分数据
            random.shuffle(data)  # 打乱数据
            split_index = int(len(data) * split_ratio)
            eval_data = data[:split_index]
            train_data = data[split_index:]

            total_train_count += len(train_data)
            total_eval_count += len(eval_data)

            # 将训练数据写回新的训练文件
            new_train_file_path = os.path.join(train_dir, filename)
            with open(new_train_file_path, 'w', encoding='utf-8') as new_train_file:
                for item in train_data:
                    json.dump(item, new_train_file)
                    new_train_file.write('\n')

            # 将验证数据追加到验证文件
            for item in eval_data:
                json.dump(item, eval_file)
                eval_file.write('\n')

    # 关闭验证集文件
    eval_file.close()

    print("Total number of training data:", total_train_count)
    print("Total number of evaluation data:", total_eval_count)

    # 返回训练目录和验证文件的绝对路径
    return os.path.abspath(train_dir), os.path.abspath(eval_file_path)


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "Z:/MY_FIELS/Project/Python/mistral-src/model/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873",
        padding_side="right",
        use_fast=True)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="</s>"))

    print(tokenizer.eos_token_id)
    print(tokenizer.pad_token_id)
    datasets_file = "../../../mistral-v2/resources/data/train/ops_ch_en_001_c.json"

    build_dataset(datasets_file, tokenizer, max_seq_length=50)


if __name__ == "__main__":
    main()
