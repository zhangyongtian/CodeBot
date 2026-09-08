# CodeBot

> **一个从零实现的轻量级 GPT 代码生成模型，包含完整的 Pretrain → SFT → GRPO 训练流水线**

`🐍 Python 3.10+`　`🔥 PyTorch 2.0+`　`📦 uv`　`📄 MIT License`　`🤖 30.1M 参数`

| 徽章 | 说明 | 链接 |
|------|------|------|
| 🐍 Python 3.10+ | 运行时版本要求 | [python.org](https://www.python.org/downloads/) |
| 🔥 PyTorch 2.0+ | 深度学习框架 | [pytorch.org](https://pytorch.org/) |
| 📦 uv | 包管理器（替代 pip/poetry） | [astral-sh/uv](https://github.com/astral-sh/uv) |
| 📄 MIT License | 开源协议 | [LICENSE](LICENSE) |
| 🤖 30.1M 参数 | 模型规模 | [模型架构](#模型架构) |

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [架构概览](#架构概览)
- [目录结构](#目录结构)
- [环境准备](#环境准备)
  - [1. 安装 uv](#1-安装-uv)
  - [2. 创建虚拟环境](#2-创建虚拟环境)
  - [3. 安装依赖](#3-安装依赖)
- [训练流程](#训练流程)
  - [Step 1: 预训练 (Pretrain)](#step-1-预训练-pretrain)
  - [Step 2: 监督微调 (SFT)](#step-2-监督微调-sft)
  - [Step 3: 强化学习微调 (GRPO)](#step-3-强化学习微调-grpo)
- [测试验证](#测试验证)
  - [test_generate.py — 预训练生成测试](#test_generatepy--预训练生成测试)
  - [test_alpaca.py — SFT 数据格式验证](#test_alpacapy--sft-数据格式验证)
- [交互式对话](#交互式对话)
- [模型超参数](#模型超参数)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## 项目简介

CodeBot 是一个**教育友好、生产可用**的 Decoder-Only Transformer 代码生成模型。项目从零实现了 GPT 架构的每一个细节（无 `transformers` 库依赖），并提供了完整的三阶段训练流水线：

| 阶段 | 目标 | 算法 | 数据 | 产物 |
|------|------|------|------|------|
| **Pretrain** | 学习代码语法与模式 | 自回归语言建模 | `tiny_codes.txt` | `model_pretrain.pt` |
| **SFT** | 对齐指令跟随能力 | 监督微调（Alpaca 格式） | `tiny_codes_sft.json` | `model_sft.pt` |
| **GRPO** | 通过奖励优化输出质量 | PPO-style Group Relative | 合成加法任务 | `model_grpo.pt` |

模型规模约 **30.1M 参数**，在普通消费级 CPU/GPU 上即可完成端到端训练。

---

## 核心特性

- ✅ **纯手写 Transformer**：`MultiHeadAttention` / `FFN` / `Block` / `GPT` 全部从零实现，代码带逐行中文注释，适合深度学习入门
- ✅ **Pre-LN 残差结构**：GPT 标准的 Pre-LayerNorm 设计，训练稳定无需复杂 warm-up
- ✅ **权重共享 (Weight Tying)**：Embedding 与 Unembed 层共享参数，省参数 + 提升泛化
- ✅ **因果掩码 (Causal Mask)**：`torch.tril` 实现严格的下三角遮蔽，保证 train-infer 一致性
- ✅ **三阶段完整流水线**：Pretrain → SFT → GRPO，覆盖大模型训练全流程
- ✅ **uv 极速包管理**：使用 [uv](https://github.com/astral-sh/uv) 替代 pip/poetry，依赖安装提速 10-100x
- ✅ **滑动窗口推理**：生成时自动按 `max_context_len` 截断输入，支持超长序列

---

## 架构概览

```
输入 token ids (B, C)
        │
        ▼
  ┌─────────────────┐
  │  Token Embedding│─────── 词嵌入：离散 id → 连续语义空间
  │  + Pos Embedding│─────── 位置嵌入：显式编码语序
  │  + Dropout      │
  └────────┬────────┘
           │ (B, C, E)
           ▼
  ┌─────────────────────────────────────┐
  │  Block × N (Pre-LN 残差堆叠)        │
  │    ┌────────────────────────────┐   │
  │    │  LayerNorm → MHA → +x      │   │  ← 残差①：跨 token 上下文融合
  │    │  LayerNorm → FFN → +x      │   │  ← 残差②：单 token 非线性加工
  │    └────────────────────────────┘   │
  └────────┬────────────────────────────┘
           │ (B, C, E)
           ▼
  ┌─────────────────┐
  │  Final LayerNorm│─────── 输出前数值规整
  └────────┬────────┘
           │ (B, C, E)
           ▼
  ┌─────────────────┐
  │  Unembed Linear │─────── E → vocab_size，权重与 Embedding 共享
  └────────┬────────┘
           │ (B, C, V)
           ▼
  vocab logits → softmax → 自回归采样生成
```

**关键配置：**
- Embedding 维度 `E = 384`
- Attention 头数 `H = 6`，单头维度 `D = 64`（支持 `H×D ≠ E` 的灵活投影）
- FFN 隐层维度 `ff_dim = 4 × E = 1536`
- Transformer 层数 `n_layer = 6`
- 最大上下文长度 `context_len = 256`
- 词表大小 `vocab_size = 50257`（GPT-2 tiktoken）

---

## 目录结构

```
CodeBot/
├── src/                          # 源代码
│   ├── model.py                  # GPT 模型架构（含详细中文注释）
│   ├── utils.py                  # generate() / get_device() 工具函数
│   ├── pretrain.py               # 阶段①：预训练脚本
│   ├── sft.py                    # 阶段②：监督微调脚本
│   ├── grpo.py                   # 阶段③：GRPO 强化学习脚本
│   ├── chat.py                   # 交互式对话 Demo
│   └── loss_imgs/                # 训练曲线图（自动生成）
├── test/                         # 测试脚本
│   ├── test_generate.py          # 预训练模型生成质量测试
│   └── test_alpaca.py            # SFT 数据格式 & Tokenizer 验证
├── data/                         # 训练数据
│   ├── tiny_codes.txt            # 预训练语料（代码文本）
│   └── tiny_codes_sft.json       # SFT 指令-响应对（Alpaca 格式）
├── models/                       # 模型权重（训练后自动生成）
│   ├── model_pretrain.pt         # 预训练权重
│   ├── model_sft.pt              # SFT 权重
│   └── model_grpo.pt             # GRPO 权重
├── pyproject.toml                # uv 包管理配置
├── .gitignore
└── README.md                     # 本文档
```

---

## 环境准备

### 1. 安装 uv

[uv](https://github.com/astral-sh/uv) 是由 Astral 用 Rust 编写的极速 Python 包管理器，替代 pip + venv + poetry。

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装完成后验证：
```bash
uv --version
```

### 2. 创建虚拟环境

在项目根目录下执行：

```bash
# 进入项目目录
cd CodeBot

# 创建虚拟环境（自动读取 pyproject.toml 的 Python 版本要求）
uv venv
```

激活虚拟环境：

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. 安装依赖

```bash
# 安装运行时依赖（torch / tiktoken / matplotlib / tqdm）
uv sync

# 如需安装开发依赖（pytest 等），使用 --dev
uv sync --dev

# 如需 GPU 版本 PyTorch（CUDA 12.1），手动指定：
# uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

验证安装：
```bash
python -c "import torch; import tiktoken; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## 训练流程

> **提示**：所有脚本均已配置绝对路径，从任意工作目录运行均可正确定位 `data/`、`models/` 等资源。

### Step 1: 预训练 (Pretrain)

**目标**：在代码语料上学习基础的语法、结构和模式，掌握"下一个 token 预测"能力。

**数据**：`data/tiny_codes.txt`（纯文本代码语料，使用 `<|endoftext|>` 分隔样本）

**执行命令：**
```bash
python src/pretrain.py
```

**关键超参数**（可在 [src/pretrain.py](src/pretrain.py) 顶部修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `context_len` | 256 | 最大上下文窗口（token 数）|
| `batch_size` | 16 | 每批次样本数 |
| `learning_rate` | 3e-4 | AdamW 学习率 |
| `max_iters` | 20000 | 训练迭代总步数 |
| `embed_dim` | 384 | 嵌入维度 E |
| `n_head` | 6 | 注意力头数 H |
| `n_layer` | 6 | Transformer Block 层数 |
| `ff_dim` | 1536 | FFN 隐层维度（= 4×E）|
| `dropout_rate` | 0.1 | Dropout 概率 |

**产物：**
- 模型权重：`models/model_pretrain.pt`
- 损失曲线：`src/loss_imgs/loss_pretrain.png`

训练启动后会打印参数量，正常输出：
```
参数总数: 30,100,000 (30.1M)
  0%|          | 1/20000 [00:03<18:00:00,  3.24s/it, loss=9.5481]
```

---

### Step 2: 监督微调 (SFT)

**目标**：在预训练基础上，让模型学会遵循 `### Instruction` → `### Response` 的对话格式，具备指令跟随能力。

**数据**：`data/tiny_codes_sft.json`，格式如下：
```json
[
  {"instruction": "Hello", "response": "Hello. What can I help you with?"},
  {"instruction": "写一个 Python 函数计算两数之和", "response": "def add(a, b):\n    return a + b"}
]
```

**前置条件**：已完成 Step 1，存在 `models/model_pretrain.pt`。

**执行命令：**
```bash
python src/sft.py
```

**关键超参数**（可在 [src/sft.py](src/sft.py) 顶部修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `context_len` | 256 | 上下文窗口 |
| `batch_size` | 16 | 批次大小 |
| `learning_rate` | 3e-4 | AdamW 学习率 |
| `max_iters` | 500 | 微调迭代步数 |

**SFT 核心机制**：
- Prompt 部分（`### Instruction: ... ### Response:`）的 token 在计算 Loss 时用 `-100` 掩码（`ignore_index=-100`），**只惩罚 Response 部分的预测错误**
- 自动处理 Padding/Truncation 到 `context_len`
- 加载 Pretrain 权重后继续训练，不会从头初始化

**产物：**
- 模型权重：`models/model_sft.pt`
- 损失曲线：`src/loss_imgs/loss_sft.png`

---

### Step 3: 强化学习微调 (GRPO)

**目标**：使用 GRPO (Group Relative Policy Optimization) 算法，通过奖励函数直接优化模型输出质量。示例任务为**个位数加法**（`a+b=?`，奖励 = 答案正确 1.0 / 错误 0.0）。

**前置条件**：已完成 Step 2，存在 `models/model_sft.pt`。

**执行命令：**
```bash
python src/grpo.py
```

**关键超参数**（可在 [src/grpo.py](src/grpo.py) 顶部修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 7e-6 | GRPO 学习率（比 SFT 小两个数量级）|
| `max_iters` | 4000 | 训练迭代步数 |
| `group_size` | 8 | 每组生成样本数（用于计算组内相对优势）|
| `batch_size` | 4 | 每轮 prompt 数（实际训练 batch = 4 × 8 = 32）|
| `n_update_per_generation` | 2 | 同一批生成数据更新几次 |
| `epsilon` | 0.2 | PPO clip 范围（1±ε）|
| `eval_interval` | 10 | 每多少步做一次全量准确率评估 |

**GRPO 核心机制**：
1. 对每个 prompt，用当前模型生成 `group_size=8` 条独立回答
2. 用奖励函数 `calculate_reward()` 给每条回答打分（0 或 1）
3. 计算组内相对优势：`advantage = reward - mean(reward_group)`
4. PPO-style clipped surrogate loss 优化策略，限制每步更新幅度不超过 ±ε
5. 定期（每 10 步）在全量 81 个加法样本上评估准确率

**产物：**
- 模型权重：`models/model_grpo.pt`
- 准确率曲线：`src/loss_imgs/loss_grpo.png`

训练过程中会实时显示 `loss` 和全量准确率 `acc`。

---

## 测试验证

项目根目录下 `test/` 文件夹包含两个测试脚本，用于验证各阶段产物。

### test_generate.py — 预训练生成测试

**用途**：加载 `models/model_pretrain.pt`，以 `"def"` 为 prompt 生成 5 段独立的代码样本，直观检查预训练质量。

**前置条件**：已完成 Step 1（预训练），且 `models/model_pretrain.pt` 文件存在。

**执行命令：**
```bash
python test/test_generate.py
```

**可调节参数**（在 [test/test_generate.py](test/test_generate.py) 顶部修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | `"def"` | 生成起始提示词，可改为 `"class"`、`"import"` 等观察不同生成效果 |
| `max_new_tokens` | `200` | 单次生成的最大 token 数量 |
| `temperature` | `1.0` | 采样温度：`0` = 贪心解码（确定性最高），值越大随机性/多样性越强 |

**预期输出**：打印 5 段以 `def` 开头的合成代码片段，语法应当逐步合理（训练步数越多越接近真实 Python 代码）。

```
--- 样本 1 ---
def add(a, b):
    return a + b

--- 样本 2 ---
def hello():
    print("Hello, World!")
...
```

> **提示**：如果训练步数较少（< 500 步），生成结果可能是乱码或无意义字符，这是正常现象。建议至少训练 2000 步以上再观察生成效果。

---

### test_alpaca.py — SFT 数据格式验证

**用途**：加载 `data/tiny_codes_sft.json`，验证第一条样本的 Alpaca 格式化结果及 tokenizer 编码正确性，**不需要任何模型权重**。

**执行命令（无需训练即可运行）：**
```bash
python test/test_alpaca.py
```

**预期输出**（示例）：
```
{'instruction': 'Hello', 'response': 'Hello. What can I help you with?'}
### Instruction:
Hello

### Response:
Hello. What can I help you with?<|endoftext|>
[35, 35, 35, 962, 519, 117, 389, 58, 10, 846, 10, 10, 35, 35, 35, 752, 568, 58, 10, 846, 46, 840, 104, 277, 280, 356, 473, 708, 108, 112, 930, 657, 63, 999]
```

---

## 交互式对话

完成 SFT 或 GRPO 后，可以通过 [src/chat.py](src/chat.py) 与模型进行实时多轮对话。

**启动命令：**
```bash
python src/chat.py
```

**切换模型**：编辑 [src/chat.py](src/chat.py) 顶部的 `model_path`：
```python
# 使用 SFT 模型（默认）
model_path = os.path.join(PROJECT_ROOT, 'models', 'model_sft.pt')

# 或使用 GRPO 模型
# model_path = os.path.join(PROJECT_ROOT, 'models', 'model_grpo.pt')
```

**调节生成参数**（同文件顶部）：
```python
max_new_tokens = 200   # 单次最大生成 token 数
temperature = 1.0      # 随机性：0 = 贪心解码，越大越多样
```

**运行示例：**
```
You: 写一个 Python 函数，输入列表返回其中的偶数
Bot:
def get_even_numbers(lst):
    return [x for x in lst if x % 2 == 0]
```

---

## 模型超参数

三阶段使用同一模型骨架，超参数汇总：

| 超参数 | Pretrain | SFT | GRPO |
|--------|----------|-----|------|
| vocab_size | 50257 | 50257 | 50257 |
| max_context_len | 256 | 256 | 256 |
| embed_dim | 384 | 384 | 384 |
| n_head | 6 | 6 | 6 |
| n_layer | 6 | 6 | 6 |
| ff_dim | 1536 | 1536 | 1536 |
| dropout_rate | 0.1 | 0.1 | 0.1 |
| **总参数量** | **~30.1M** | **~30.1M** | **~30.1M** |

---

## 常见问题

### Q1: Windows 下激活虚拟环境报错 `ExecutionPolicy`
PowerShell 执行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
然后重新激活 `.venv\Scripts\Activate.ps1`。

### Q2: 训练很慢，能不能加速？
1. **检查是否在用 GPU**：训练脚本启动会打印 `device`，如果显示 `cpu` 请安装 CUDA 版 PyTorch（见[安装依赖](#3-安装依赖)）。
2. **减小 `batch_size`**：显存不够时调小（如 8、4）。
3. **减小 `max_iters`**：快速验证流水线时 Pretrain 用 1000 步、SFT 用 100 步即可。

### Q3: 报错找不到 `data/tiny_codes.txt` / `models/`
项目脚本已使用绝对路径，**无需 `cd` 到特定目录**，但需确保 `data/` 下文件和 `models/` 目录存在：
```bash
# Windows
mkdir models
mkdir src\loss_imgs

# Linux/macOS
mkdir -p models src/loss_imgs
```

### Q4: 训练到一半中断，能续训吗？
当前脚本未实现 checkpoint 断点续训。如需续训，可在 `pretrain.py` / `sft.py` / `grpo.py` 中定期保存 `optimizer.state_dict()`，训练开始时加载即可。

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源，欢迎自由使用、修改和分发。学习用途无需声明，商业用途请保留版权声明。

---

**⭐ 如果本项目对你有帮助，欢迎点个 Star 支持！**
