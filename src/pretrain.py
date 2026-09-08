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
# 使用基于项目根目录的绝对路径，避免工作目录不同导致找不到文件
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
model_save_path = os.path.join(PROJECT_ROOT, 'models', 'model_pretrain.pt')
data_file_path = os.path.join(PROJECT_ROOT, 'data', 'tiny_codes.txt')
loss_fig_path = os.path.join(PROJECT_ROOT, 'loss_imgs', 'loss_pretrain.png')


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
        # 对完整文本做 GPT-2 BPE 分词；allowed_special={"<|endoftext|>"} 表示：
        #   1) 遇到 <|endoftext|> 不报错；2) 将其整体视为 1 个控制 token（id=50256），不参与 BPE 拆分
        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        # 将分词得到的 token id list 转为 long 型一维张量，形状 (N,)；N 为语料总 token 数
        # 必须用 long 类型（整数），因为 nn.Embedding 的输入要求是整数索引
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        # 缓存上下文窗口长度，供 __len__ 和 __getitem__ 切片使用
        self.context_len = context_len

    def __len__(self):
        # 可切出的有效样本数 = 总 token 数 - 窗口长度（每个样本需要 T+1 个连续 token 以构造 x 和 y）
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx):
        # 滑动窗口切片：从 idx 开始取连续的 (context_len+1) 个 token，拆成 x 和 y——两者来自同一窗口，y 相对 x 整体右移 1 位
        # 这样每个位置 k 都形成"用 x[:k+1] 预测下一个词 y[k]"的监督对（自回归语言模型的标准 Teacher Forcing）
        x = self.tokens[idx:idx+self.context_len]         # 输入 x = [t_idx, t_{idx+1}, ..., t_{idx+T-1}]，长度 T
        y = self.tokens[idx+1:idx+self.context_len+1]     # 标签 y = [t_{idx+1}, t_{idx+2}, ..., t_{idx+T}]，长度 T，y[k] 就是 x 第 k 位要预测的"下一个词"
        return x, y


with open(data_file_path, "r", encoding="utf-8") as f:
    text = f.read()

dataset = TokenDataset(text, context_len)
# DataLoader 工作流程示意（L = dataset.__len__() = 总样本数，context_len = 256）：
#   ① 生成全部合法 idx： [0, 1, 2, 3, 4, 5, 6, 7, ..., L-1]    ← 共 L 个，范围由 __len__ 限定
#   ② shuffle=True 打乱：[305, 12, 8888, 0, 777, 42, ...]       ← 还是 L 个，顺序随机，降低样本相关性
#   ③ 按 batch_size=16 连续分组：
#        第 1 个 batch → 取前 16 个 idx  → 各自调用 __getitem__ → 拼接为 batch_x/batch_y，shape (16, 256)
#        第 2 个 batch → 取接下来 16 个 idx → 同样拼接为 (16, 256)
#        ...
#        最后 1 个 batch → 剩余不足 16 条时就按实际条数返回（若 drop_last=True 则会丢弃）
#   每个 epoch 遍历完 L 条样本后会重新执行 ② 和 ③ 再次打乱分组
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

losses = []                                               # 记录每一步(iter)的训练 Loss，最后用于画曲线
# 把 dataloader 包装为无限迭代器：跑完一个 epoch(共 L 条样本)后自动重新 shuffle + 分 batch，无缝从头再来
# 主循环按固定的 max_iters 步数训练，无需手动管理 epoch 边界和迭代器重建
data_iter = cycle(dataloader)
# tqdm 进度条，长度 = max_iters = 20000；每完成一步(=处理一个 batch = 参数更新一次)进度条前进一格
pbar = tqdm(range(max_iters))

# 训练主循环：每一次 for 循环 = 1 iter = 1 个 batch = 参数通过 AdamW 更新 1 次
for i in pbar:
    # ---------- 1. 取一个 batch 并迁移到目标设备(GPU/CPU) ----------
    # 从无限迭代器拿 1 个 batch：batch_x / batch_y 的 shape 都是 (batch_size, context_len) = (16, 256)
    batch_x, batch_y = next(data_iter)
    # 将数据迁移到与 model.parameters() 相同的设备，否则 CPU/GPU 张量混算会直接报错
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    # ---------- 2. 前向传播 + 计算 Loss ----------
    # 前向：输入 (B, T) 的 token id → 模型输出 (B, T, vocab_size) 的每个位置所有词的打分(logits)
    logits = model(batch_x)
    # 交叉熵：把 (B,T) 展平成 (B*T,) 的二维分类问题——所有样本、所有位置都参与"预测下一个词"的监督
    #
    #   【直观例子】假设原始语料 token 是: [我, 爱, 吃, 苹, 果, 。]，context_len=4
    #     batch_x (输入) = [我, 爱, 吃, 苹]   ← 从 [0:4] 切片
    #     batch_y (标签) = [爱, 吃, 苹, 果]   ← 从 [1:5] 切片，整体右移 1 位
    #
    #     每个位置的对应关系（交叉熵就是让 logits 在真实词上得分越高越好）：
    #       x[0]=我 → 看完"我"  → 预测下一个词应该是？→ logits[0] 里 "爱" 得分最高 ←→ y[0]=爱
    #       x[1]=爱 → 看完"我爱" → 预测下一个词应该是？→ logits[1] 里 "吃" 得分最高 ←→ y[1]=吃
    #       x[2]=吃 → 看完"我爱吃"→ 预测下一个词应该是？→ logits[2] 里 "苹" 得分最高 ←→ y[2]=苹
    #       x[3]=苹 → 看完"我爱吃苹"→ 预测下一个词应该是？→ logits[3] 里 "果" 得分最高 ←→ y[3]=果
    #                    ↑                               ↑                         ↑
    #              每个位置的预测分数 logits        交叉熵拉高分            真实下一个词
    #
    #   展平操作（把 B*T 个位置当作一批分类样本一起算 loss）：
    #     logits.view(-1, vocab_size) : (B, T, V) -> (B*T, V)  每一行 = 某样本某位置所有词的得分
    #     batch_y.view(-1)            : (B, T)    -> (B*T,)     对应位置的真实下一个词 id
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

    # ---------- 3. 反向传播 + 优化器更新参数 ----------
    optimizer.zero_grad()                          # 清空上一步累积的梯度(PyTorch 默认累加，不清空会越叠越大)
    loss.backward()                                # 反向：从 Loss 开始沿计算图对所有可训练参数求梯度 dL/dW
    optimizer.step()                               # 更新：AdamW 根据梯度和学习率更新所有参数 W = W - lr*AdamW_update

    # ---------- 4. 记录当前步 Loss，刷新进度条右侧显示 ----------
    losses.append(loss.item())                     # loss.item() 把单元素张量转成 Python 标量，避免显存累积
    pbar.set_postfix({'loss': f'{loss.item():.4f}'})  # 在 tqdm 进度条末尾实时显示当前 Loss(4 位小数)

# 保存结果
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(loss_fig_path)

model.save(model_save_path)
