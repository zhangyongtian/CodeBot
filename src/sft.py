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
        # ─────────────────────────────────────────────────────────────────────
        # 【具体 token 举例】（完整走一遍 Step 2 拼接掩码 → Step 3 移位 → Step 4 Padding）
        #   训练样本：instruction="写hi", response="你好"
        #
        # Step 2【拼接 + 掩码（移位前）】总长 = prompt_ids(6) + response_ids(3) = 9
        #   prompt_ids   = [###, Instruction:, 写, hi, ###, Response:]      ← 6 个 token，包装后的指令区
        #   response_ids = [你, 好, <|endoftext|>]                           ← 3 个 token，回答 + EOS 结束符
        #   ids_before_shift    = [###, Instruction:, 写, hi, ###, Response:, 你, 好, <|endoftext|>]  ← 长 9
        #   labels_before_shift = [-100, -100,          -100,-100,-100, 你,       好, <|endoftext|>]
        #                         ↑──── Prompt 区 6 个全掩码 -100（不监督复述指令）───↑ ↑ Response 区 3 个真实值 ↑
        #
        # Step 3【Teacher Forcing 移位】ids 去掉最后一位，labels 去掉第一位，长度都变成 8
        #   核心规则：ids[i] 是"模型看到的输入 token"，labels[i] 是"模型应该预测的下一个 token"
        #   ids    = [###, Instruction:, 写, hi, ###, Response:, 你, 好]   ← 长度 8（就是 ids_before_shift[:-1]，少了最末尾的 <|endoftext|>）
        #   labels = [-100, -100,      -100,-100,-100, 你,       好, <|endoftext|>]  ← 长度 8（就是 labels_before_shift[1:]）
        #                                                                       ↑ 注意这个位置：
        #   逐位对应关系：                                                         ids[7]="好" 对应的监督是 labels[7]="<|endoftext|>"，
        #     i=0: 看到 "###"           → 预测 (-100 跳过)                    即"看到'好'这个字 → 接下来应该输出 EOS 停止"
        #     i=1: 看到 " Instruction:"  → 预测 (-100 跳过)
        #     i=2: 看到 "写"            → 预测 (-100 跳过)
        #     i=3: 看到 "hi"           → 预测 (-100 跳过)
        #     i=4: 看到 "###"           → 预测 (-100 跳过)
        #     i=5: 看到 " Response:"    → 预测 "你"       ← ✅ 关键："Response:" 这个 token 虽然属于 Prompt 但在 ids 里，
        #     i=6: 看到 "你"            → 预测 "好"                          它的作用是"触发模型开始输出回答"；
        #     i=7: 看到 "好"            → 预测 "<|endoftext|>"                但 labels[i=5] 从这里开始才是真实监督。
        #
        # ── 场景 A：假设 context_len=10（需 Padding 2 位）──
        #   pad_len = 10 - 8 = 2
        #   ids    补: [###, Instruction:, 写, hi, ###, Response:, 你, 好, 0,    0   ]
        #   labels 补: [-100, -100,      -100,-100,-100, 你,       好, <|endoftext|>, -100, -100]
        #                                                                      ↑ padding 位全 -100 掩码
        #                                                                   ↑ ids padding 填 0 或任意值都没关系，
        #                                                                     因为对应 labels 已经是 -100
        #
        # ── 场景 B：假设 context_len=7（需 Truncation，尾部砍掉 1 位）──
        #   pad_len = 7 - 8 = -1
        #   ids    截: [###, Instruction:, 写, hi, ###, Response:, 你]   ← 砍掉了最后一位 "好"
        #   labels 截: [-100, -100,      -100,-100,-100, 你,       好]   ← 砍掉了最后一位 "<|endoftext|>"
        #   ⚠️ 实际风险：原本应该学到的"看到'好'→ 输出 EOS 停止"这一条关键监督信号被截断丢失了。
        #              大量样本截断在 Response 末尾的话，模型会学不到"回答完应该停下来输出 <|endoftext|>"
        #              的习惯，推理时可能出现"回答停不下来、一直胡编续写"的问题。
        #              所以 context_len 要覆盖数据集中最长的"指令+回答"总长度。
        # ─────────────────────────────────────────────────────────────────────
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
        # 从 self.samples 中取出第 idx 条样本（Python list[int] 格式，省内存便于多进程加载）
        ids, labels = self.samples[idx]
        # ====================================================================
        # 返回的两个 Tensor：把 Python list 转为 PyTorch 模型要求的 LongTensor
        # 为什么必须用 torch.long (= torch.int64)？
        #   1. GPT 内部 nn.Embedding(V, E) 做查表时，indices 必须是 int64
        #   2. F.cross_entropy 的 target 参数必须是 int64 类别索引
        #   用 int32/float 都会直接报错。
        #
        # 【真实数值样例】以 tiny_codes_sft.json 第 0 条（instruction="Hello",
        #   response="Hello. What can I help you with?"）、context_len=256 为例，
        #   严格按照 Step 2 拼接掩码 → Step 3 Teacher Forcing 移位 后的状态：
        #
        #  Step 2 拼接掩码（移位前，总长 20 = prompt_ids(10) + response_ids(10)）：
        #    ids_before_shift[0:10]  = [###, Inst, :, \n, Hello, \n\n, ###, Resp, :, \n]  ← prompt_ids
        #    ids_before_shift[10:20] = [Hello, ., What, can, I, help, you, with, ?, <eos>] ← response_ids
        #    labels_before_shift[0:10]  = 全 -100（prompt 区掩码）
        #    labels_before_shift[10:20] = [Hello, ., What, can, I, help, you, with, ?, <eos>] ← response 区真实监督
        #
        #  Step 3 移位：ids = ids_before_shift[:-1]  (丢末尾 <eos>，长 19)
        #               labels = labels_before_shift[1:]  (丢首位 -100，长 19)
        #
        #  ① ids_tensor：shape=(256,), dtype=torch.int64
        #     前 19 位有效 token（gpt2 真实 ID，与 ids_before_shift[:-1] 完全一致）：
        #       索引  :   0     1     2    3     4     5    6     7    8    9 |  10    11   12    13   14    15    16    17   18
        #       ID    : [21790, 16288,   25,  198, 15496,  271, 21790, 18261,  25,  198, 15496,  13, 1867,  460,  314, 1037,  345,  351,  30]
        #       解码  : "###"  " Inst" ":" "\n" "Hello" "\n\n" "###" " Resp" ":" "\n" "Hello" "." " Wh" " ca" " I"  " hel" " you" " wi" "?"
        #     后 237 位 = 全 0（padding，填什么都行，因为 labels 对应位置是 -100）
        #
        #  ② labels_tensor：shape=(256,), dtype=torch.int64（与 ids 同长，逐位对应 Teacher Forcing）
        #     前 9 位 = 全 -100（prompt 区，看到 "###"~":" 这些位置都不监督）
        #     第 9~18 位（response 区，真实监督信号，共 10 位 = labels_before_shift[10:20]）：
        #       索引  :   9     10   11    12    13   14     15     16    17    18
        #       ID    : [15496,  13, 1867,  460,  314,  1037,   345,   351,    30, 50256]
        #       解码  : "Hello" "." " Wh" " ca" " I"   " hel" " you" " wi"   "?"  "<|endoftext|>"
        #     逐位含义（ids[i] 输入 → labels[i] 预测目标）：
        #       i=9 :  ids[9]="\n"(Response: 后换行)  →  预测 "Hello"  ← 回答开始！
        #       i=10:  ids[10]="Hello"                →  预测 "."
        #       i=11:  ids[11]="."                    →  预测 " What"
        #       ...     中间依此类推 ...
        #       i=18:  ids[18]="?"                    →  预测 "<|endoftext|>"  ← 回答结束，教模型输出 EOS 停止
        #     后 237 位 = 全 -100（padding 区，同样掩码）
        # ====================================================================
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
    # DataLoader 取 batch（内部：生成 16 个随机 idx → 逐个调用 __getitem__ → torch.stack(dim=0) 拼接）
    # 【关于 ids/labels 的错位关系、-100 掩码语义、逐位对应规则：已在上面 __getitem__ 的注释中完整讲解】
    # 这里只说明多了 batch 维度之后的形状变化（batch_size=16, context_len=256, vocab_size=50257）：
    #   batch_x : (16, 256)  int64  每一行 = 1 条样本的 ids_tensor
    #   batch_y : (16, 256)  int64  每一行 = 对应的 labels_tensor（-100 = 掩码跳过，0~50256 = 真实监督）
    #   logits  : (16, 256, 50257)  float32  →  model(batch_x) 得到每个位置对词表的打分
    #   cross_entropy: logits.view(B*T, V) 与 batch_y.view(B*T) 逐位置计算，-100 的位置被忽略
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
