# CodeBot - GPT 语言模型

基于 PyTorch 从零实现的 GPT 语言模型项目。

## 目录

- [项目结构](#项目结构)
- [环境准备](#环境准备)
- [使用 uv 创建项目](#使用-uv-创建项目)
- [安装依赖](#安装依赖)
- [文件说明](#文件说明)

## 项目结构

```
CodeBot/
├── src/
│   ├── model.py      # GPT 模型定义
│   └── utils.py      # 工具函数（生成、设备选择）
├── .gitignore
└── README.md
```

## 环境准备

- Python >= 3.9
- [uv](https://github.com/astral-sh/uv) 包管理工具

### 安装 uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 使用 uv 创建项目

```bash
# 1. 创建项目目录并进入
mkdir CodeBot && cd CodeBot

# 2. 使用 uv 初始化项目
uv init --name codebot --python 3.10

# 3. 设置 PyTorch 源（可选，加速国内下载）
# uv pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 安装依赖

### 核心依赖

```bash
# 安装 PyTorch（CUDA 12.1 版本，推荐）
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# 或者安装 CPU 版本
# uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 可选依赖

```bash
# 模型训练工具
uv pip install numpy pandas tqdm

# Jupyter Notebook（交互式开发）
uv pip install jupyter ipython

# 训练日志与可视化
uv pip install tensorboard wandb

# 数据处理
uv pip install datasets transformers tokenizers
```

### 一键安装全部依赖

```bash
# 创建 requirements.txt
cat > requirements.txt << 'EOF'
torch --index-url https://download.pytorch.org/whl/cu121
numpy
pandas
tqdm
jupyter
ipython
tensorboard
EOF

# 使用 uv 批量安装
uv pip install -r requirements.txt
```

## 文件说明

### [src/model.py](file:///c:/Users/zz/Desktop/codebot/CodeBot/src/model.py)

GPT 模型核心实现，包含以下模块：

- `MultiHeadAttention` - 多头自注意力机制
- `LayerNorm` - 层归一化
- `GELU` - GELU 激活函数
- `FFN` - 前馈神经网络
- `Block` - Transformer 编码器块
- `GPT` - GPT 主模型（支持 save/load 权重）

### [src/utils.py](file:///c:/Users/zz/Desktop/codebot/CodeBot/src/utils.py)

工具函数：

- `generate()` - 自回归文本生成（支持 temperature 采样）
- `get_device()` - 自动选择设备（CUDA / MPS / CPU）
