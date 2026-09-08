import os
import sys
from itertools import cycle
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

device = get_device()
# 使用基于脚本位置的绝对路径，避免工作目录不同导致找不到文件
model_save_path = os.path.join(SCRIPT_DIR, 'models', 'model_pretrain.pt')
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
data_file_path = os.path.join(PROJECT_ROOT, 'data', 'tiny_codes.txt')
loss_fig_path = os.path.join(SCRIPT_DIR, 'loss_pretrain.png')


context_len = 256
vocab_size = 50257
batch_size = 16
learning_rate = 3e-4
max_iters = 20000
embed_dim = 384
n_head = 6
n_layer = 6
ff_dim = 4 * embed_dim
dropout_rate = 0.1


class TokenDataset(Dataset):
    def __init__(self, text, context_len):
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.context_len]
        y = self.tokens[idx+1:idx+self.context_len+1]
        return x, y


with open(data_file_path, "r", encoding="utf-8") as f:
    text = f.read()

dataset = TokenDataset(text, context_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 模型和优化器
model = GPT(
    vocab_size=vocab_size,
    max_context_len=context_len,
    embed_dim=embed_dim,
    n_head=n_head,
    n_layer=n_layer,
    ff_dim=ff_dim,
    dropout_rate=dropout_rate
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f"参数总数: {total_params:,} ({total_params/1e6:.1f}M)")

losses = []
data_iter = cycle(dataloader)  # 转换为无限循环
pbar = tqdm(range(max_iters))

for i in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    logits = model(batch_x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

# 保存结果
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(loss_fig_path)

model.save(model_save_path)
