import os
import sys
import tiktoken

# 将脚本所在目录加入 sys.path，确保 model 和 utils 可以正确导入
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils import generate, get_device
from model import GPT


# 配置
device = get_device()
# 使用基于项目根目录的绝对路径，统一存放于 models/ 文件夹
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# SFT 模型路径（监督微调）
model_path = os.path.join(PROJECT_ROOT, 'models', 'model_sft.pt')
# GRPO 模型路径（强化学习微调）
# model_path = os.path.join(PROJECT_ROOT, 'models', 'model_grpo.pt')
max_new_tokens = 200
temperature = 1.0


def format_prompt(user_message):
    return f"### Instruction:\n{user_message}\n\n### Response:\n"


# 加载模型和分词器
tokenizer = tiktoken.get_encoding("gpt2")
model = GPT.load_from(model_path, device=device)

while True:
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue

    # 格式化提示词并生成回复
    prompt = format_prompt(user_input)
    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)

    # 仅提取助手的回复部分
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    # 根据回复中是否包含换行符切换输出格式
    if "\n" in response:
        print(f"Bot:\n{response}")
    else:
        print(f"Bot: {response}")
