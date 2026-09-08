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
        """
        构造单条 SFT 训练样本，核心流程分为 4 步：
        Step 1. Chat Template 包装：将 instruction + response 统一包装成「对话模板」格式
        Step 2. Prompt 掩码：Instruction 部分标记为 -100，模型只学 Response 部分的输出
        Step 3. 序列移位（Teacher Forcing）：ids 与 labels 各偏移一位，实现 i 位置预测 i+1 token
        Step 4. Padding / Truncation：统一到 context_len 长度，便于 batch 拼接训练
        """
        # ====================================================================
        # Step 1. Chat Template（对话模板）——建立训练与推理的"格式契约"
        # ====================================================================
        # 所有训练数据统一包装为以下格式，推理时也必须用相同格式提问，否则模型"识别不出"该输出了。
        #   完整示例：
        #   ### Instruction:
        #   写一个Python冒泡排序
        #
        #   ### Response:
        #   def bubble_sort(arr):
        #       ...
        #   <|endoftext|>
        #
        # 模型从中学到的条件反射：
        #   「当看到前缀 "...### Response:\n" 出现时 → 我接下来要输出"回答内容"，直到 <|endoftext|> 停止」
        #
        # 业界常见模板参考（格式不同但思路完全一致）：
        #   - Llama2:    <s>[INST] {user} [/INST] {assistant} </s>
        #   - ChatML:    <|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
        #   - Alpaca:    ### Instruction:\n...\n\n### Response:\n  ← 当前代码采用的类 Alpaca 格式
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        # 在回答末尾追加 <|endoftext|>，教模型学会"回答结束后自动停止生成"
        response = f"{response}<|endoftext|>"

        # 分别对 Prompt 和 Response 分词，分开处理便于后续做"分段掩码"
        prompt_ids = self.tokenizer.encode(
            prompt, allowed_special={"<|endoftext|>"})
        response_ids = self.tokenizer.encode(
            response, allowed_special={"<|endoftext|>"})

        # ====================================================================
        # Step 2. Prompt 掩码（Partial Supervision）——只监督「回答部分」
        # ====================================================================
        # 核心思想：SFT 要教模型的是「看到指令后如何回答」，不是「如何复述指令」。
        # 因此 Instruction/Prompt 部分对应的 labels 标记为 -100，
        # 配合后续 F.cross_entropy 的 ignore_index=-100 参数，这些位置的损失会被直接忽略，不参与梯度更新。
        #
        # 举例说明（token 用单词代替，便于理解）：
        #   prompt_ids  = [###, Instruction, 写, 冒泡, ###, Response:]  → labels 对应填 -100
        #   response_ids= [def, bubble_sort, ..., <|endoftext|>]        → labels 填真实 token_id（参与损失计算）
        #   ─────────────────────────────────────────────────────────────────────
        #   ids    = [###, Instruction, 写, 冒泡, ###, Response:, def, bubble_sort, ..., <|endoftext|>]
        #   labels = [-100, -100,       -100, -100, -100, -100,      def, bubble_sort, ..., <|endoftext|>]
        #            ↑──────────── 不计算损失 ────────────↑  ↑──────── 监督模型输出 ────────↑
        ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        # ====================================================================
        # Step 3. Teacher Forcing 移位 —— 输入 ids 与标签 labels 各偏移一位
        # ====================================================================
        # 与 pretrain 自回归训练完全一致的逻辑：位置 i 的 logits 预测的是位置 i+1 的 token
        #   原始 ids:    [t0, t1, t2, t3, t4, t5]
        #   ids[:-1] →  [t0, t1, t2, t3, t4]     作为模型输入（batch_x）
        #   labels[1:] → [t1, t2, t3, t4, t5]    作为监督答案（batch_y）
        #   对应关系：模型看到 t0..t4 → 输出 t1..t5 的概率分布，与 labels 逐位算交叉熵
        ids = ids[:-1]
        labels = labels[1:]

        # ====================================================================
        # Step 4. Padding / Truncation —— 统一到固定长度 context_len，便于 DataLoader 组 batch
        # ====================================================================
        # PyTorch DataLoader 要求同一 batch 内的张量形状完全一致，因此必须对样本做长度归一化。
        #   pad_len > 0：样本太短，尾部填充 [0]（tiktoken gpt2 中 ID=0 是 '!'，无关紧要，因为 labels 用 -100 掩码了）
        #   pad_len < 0：样本太长，尾部截断，丢弃超出 context_len 的部分（可能丢失部分回答末尾）
        pad_len = self.context_len - len(ids)
        if pad_len > 0:
            ids = ids + [0] * pad_len
            labels = labels + [-100] * pad_len  # Padding 部分同样标记为 -100，不参与损失
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
    # 计算交叉熵损失
    # 关键参数 ignore_index=-100：
    #   与 _create_sample 中「Prompt 部分 labels = -100」形成对应
    #   效果：凡是 batch_y 中值为 -100 的位置（即 Instruction + Padding 区域），
    #         该位置的 logits 无论输出什么，都不会被计入损失，也不会产生梯度回传。
    #   最终：模型只在「Response 回答部分」的 token 上接受监督，学习"看到指令后该怎么回答"。
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # (B*T, V) 展平为二维
        batch_y.view(-1),                  # (B*T,)   展平为一维
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
