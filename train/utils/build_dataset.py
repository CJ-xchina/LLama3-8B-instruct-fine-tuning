from dask import datasets
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase

def build_dataset(data_path,
                  tokenizer,
                  max_seq_length: int, data_cache_dir=None,
                  preprocessing_num_workers=None,
                  ):
    def tokenize(batch):
        # 处理整个批次的数据
        new_prompts = [
            f"<s>[INST] Here are the inputs: {input_item} [/INST] \n {output_item} </s>"
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
    print(f"inputs_ids 长度:{len(processed_dataset['train'][0]['input_ids'])}")
    print(f"数据长度:{len(processed_dataset['train'])}")
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets


import os
import json
import random


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
    # 调用函数的示例
    train_dir, eval_file_path = split_data('Z:/MY_FIELS/Project/Python/mistral-src/mistral-v2/resources/data/train', 0.05)
    print("Train directory:", train_dir)
    print("Eval file path:", eval_file_path)


if __name__ == "__main__":
    main()
