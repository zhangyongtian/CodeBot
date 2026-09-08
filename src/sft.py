import os
import sys
from itertools import cycle
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# 将脚本所在目录加入 sys.path，确保 model 和 utils 可以正确导入
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import GPT
from utils import get_device
import tiktoken

# 基于项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# 设置
device = get_device()
data_path = os.path.join(DATA_DIR, 'tiny_codes_sft.json')
pretrain_model_path = os.path.join(MODELS_DIR, 'model_pretrain.pt')
sft_model_save_path = os.path.join(MODELS_DIR, 'model_sft.pt')
loss_fig_path = os.path.join(PROJECT_ROOT, 'loss_imgs', 'loss_sft.png')

# 超参数
context_len = 256
batch_size = 16
learning_rate = 3e-4
max_iters = 500


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, context_len):
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            ids, labels = self._create_sample(
                item['instruction'], item['response'])
            self.samples.append((ids, labels))

    def _create_sample(self, instruction, response):
        # 将提示（Prompt）和响应（Response）格式化
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        response = f"{response}<|endoftext|>"

        # 分词
        prompt_ids = self.tokenizer.encode(
            prompt, allowed_special={"<|endoftext|>"})
        response_ids = self.tokenizer.encode(
            response, allowed_special={"<|endoftext|>"})

        # 创建输入序列和标签（提示部分用 -100 进行掩码）
        ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        # 为语言模型进行移位（将输入和正确答案各偏移一位）
        ids = ids[:-1]
        labels = labels[1:]

        # 根据 context_len（上下文长度）进行填充（padding）或截断（truncation）
        pad_len = self.context_len - len(ids)
        if pad_len > 0:
            ids = ids + [0] * pad_len  # 使用 0 作为填充 ID（padding ID）
            labels = labels + [-100] * pad_len
        elif pad_len < 0:
            ids = ids[:self.context_len]
            labels = labels[:self.context_len]

        return ids, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), \
            torch.tensor(labels, dtype=torch.long)


# 分词器（Tokenizer）和数据集的准备
tokenizer = tiktoken.get_encoding("gpt2")
dataset = SFTDataset(data_path, tokenizer, context_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型与优化器
model = GPT.load_from(pretrain_model_path, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 训练
losses = []
data_iter = cycle(dataloader)
pbar = tqdm(range(max_iters))

for i in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    logits = model(batch_x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch_y.view(-1),
        ignore_index=-100
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

# 保存损失函数图像
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(loss_fig_path)

# 保存模型
model.save(sft_model_save_path)
