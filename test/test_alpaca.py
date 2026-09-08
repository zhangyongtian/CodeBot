import os
import sys
import json
import tiktoken

# 计算项目根目录和 data 目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


# 加载 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 加载 JSON 数据
with open(os.path.join(DATA_DIR, 'tiny_codes_sft.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

# 取出第一个样本
item = data[0]
print(item)
# {'instruction': 'Hello', 'response': 'Hello. What can I help you with?'}

# 转换为 Alpaca 格式
text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
print(text)
# ### Instruction:
# Hello
#
# ### Response:
# Hello. What can I help you with?<|endoftext|>

# 转换为 token
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(ids)
# [35, 35, 35, 962, 519, 117, 389, 58, 10, 846, 10, 10, 35, 35, 35, 752, 568, 58, 10, 846, 46, 840, 104, 277, 280, 356, 473, 708, 108, 112, 930, 657, 63, 999]
