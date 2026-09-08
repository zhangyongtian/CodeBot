import os
import sys
import tiktoken

# 将 src 目录加入 sys.path，确保 model 和 utils 可以正确导入
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import GPT
from utils import get_device, generate


# 基础设置
device = get_device()
model_path = os.path.join(PROJECT_ROOT, 'models', 'model_pretrain.pt')

# 生成设置
prompt = "def"  # 生成文本的起始提示词
max_new_tokens = 200  # 生成 token 数量的上限
temperature = 1.0  # 温度参数（越高则随机性越强）


tokenizer = tiktoken.get_encoding("gpt2")
model = GPT.load_from(model_path, device=device)

# 生成文本
for i in range(5):
    print(f"--- 样本 {i+1} ---")
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    print(generated_text)
    print()
