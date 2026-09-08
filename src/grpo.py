import os
import sys
import tiktoken
import re
from itertools import cycle
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch

# 将脚本所在目录加入 sys.path，确保 model 和 utils 可以正确导入
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils import generate, get_device
from model import GPT


# 数据集
class GRPODataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        for i in range(1, 10):
            for j in range(1, 10):
                prompt = f"### Instruction:\n{i}+{j}=\n\n### Response:\n"
                ground_truth = i + j
                self.data.append((prompt, ground_truth))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_batch(self, prompts, responses, device):
        all_ids = []
        all_masks = []

        for prompt, response in zip(prompts, responses):
            prompt_ids = self.tokenizer.encode(
                prompt, allowed_special={"<|endoftext|>"})
            response_ids = self.tokenizer.encode(
                response, allowed_special={"<|endoftext|>"})

            ids = prompt_ids + response_ids
            mask = [0] * len(prompt_ids) + [1] * len(response_ids)

            all_ids.append(ids)
            all_masks.append(mask)

        # 填充
        max_len = max(len(ids) for ids in all_ids)
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(all_ids, all_masks):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)
            padded_masks.append(mask + [0] * pad_len)

        ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        mask = torch.tensor(padded_masks, dtype=torch.float, device=device)

        return ids, mask


# 奖励函数
def calculate_reward(ground_truth, response):
    try:
        matches = re.findall(r'(-?\d+)', response)
        if matches:
            predicted = int(matches[-1])  # 获取最后一个数值
            return 1.0 if predicted == ground_truth else 0.0
        return 0.0
    except:
        return 0.0


# 组生成
def generate_group(model, tokenizer, prompts, gts, group_size):
    all_prompts = []
    all_responses = []
    all_advantages = []

    for prompt, gt in zip(prompts, gts):
        responses = []
        for _ in range(group_size):
            full_text = generate(model, tokenizer, prompt, temperature=1.0)
            response = full_text[len(prompt):]
            responses.append(response)

        rewards = torch.tensor([calculate_reward(gt, r) for r in responses])
        advantages = rewards - rewards.mean()

        for response, advantage in zip(responses, advantages):
            all_prompts.append(prompt)
            all_responses.append(response)
            all_advantages.append(advantage)

    return all_prompts, all_responses, torch.stack(all_advantages)

# 损失函数


def compute_probs(model, ids):
    logits = model(ids)  # (B, C, V)
    probs = F.softmax(logits[:, :-1, :], dim=-1)  # (B, C-1, V)
    labels = ids[:, 1:]  # (B, C-1)

    token_probs = torch.gather(
        probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, C-1)

    return token_probs


def grpo_loss(model, old_model, ids, mask, advantages, epsilon=0.2):
    # 当前模型中各个词元的概率
    probs = compute_probs(model, ids)
    # 旧模型中各个词元的概率
    with torch.no_grad():
        old_probs = compute_probs(old_model, ids)

    # 每个词元的概率比（加上一个极小值以防止除以 0）
    ratio = probs / (old_probs + 1e-8)
    advantages = advantages.unsqueeze(-1)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    mask = mask[:, 1:]  # 掩码也相应移位
    token_objective = torch.min(unclipped, clipped) * mask

    # 按样本数（batch_size × group_size）进行归一化
    n_samples = ids.size(0)  # batch_size × group_size
    return -token_objective.sum() / n_samples


# 配置
device = get_device()
# 使用基于项目根目录的绝对路径，统一存放于 models/ 文件夹
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sft_model_path = os.path.join(PROJECT_ROOT, 'models', 'model_sft.pt')
grpo_model_save_path = os.path.join(PROJECT_ROOT, 'models', 'model_grpo.pt')

# 超参数
learning_rate = 7e-6
max_iters = 4000
n_update_per_generation = 2  # 对同一批生成数据的更新次数
eval_interval = 10
epsilon = 0.2  # 裁剪范围
group_size = 8  # 组大小
batch_size = 4

# 初始化
tokenizer = tiktoken.get_encoding("gpt2")
model = GPT.load_from(sft_model_path, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

old_model = GPT.load_from(sft_model_path, device=device)  # 旧模型
old_model.eval()

dataset = GRPODataset(tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
data_iter = cycle(dataloader)

# 训练循环
accuracies = []
current_accuracy = 0.0
pbar = tqdm(range(max_iters))

for i in pbar:
    # 获取批次数据
    prompts, gts = next(data_iter)

    # 更新旧模型（old_model）
    old_model.load_state_dict(model.state_dict())

    # 使用旧模型生成多个样本，并计算奖励和优势值
    all_prompts, all_responses, all_advantages = generate_group(
        old_model, tokenizer, prompts, gts, group_size
    )

    # 创建批次数据
    ids, mask = dataset.get_batch(all_prompts, all_responses, device)
    all_advantages = all_advantages.to(device)

    # 对生成的数据进行多次更新
    for _ in range(n_update_per_generation):
        optimizer.zero_grad()
        loss = grpo_loss(model, old_model, ids, mask, all_advantages, epsilon)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

    # 定期评估
    if i % eval_interval == 0:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for prompt, gt in dataset.data:
                response = generate(model, tokenizer, prompt, temperature=0)
                reward = calculate_reward(gt, response)
                correct += reward > 0
                total += 1
        model.train()
        current_accuracy = correct / total * 100
        accuracies.append(current_accuracy)

    pbar.set_postfix({'loss': f'{loss.item():.4f}',
                     'acc': f'{current_accuracy:.1f}%'})

# 保存训练完成的模型
model.save(grpo_model_save_path)

plt.figure()
steps = list(range(0, len(accuracies) * eval_interval, eval_interval))
plt.plot(steps, accuracies)
plt.xlabel('Iteration')
plt.ylabel('Accuracy (%)')
plt.title('GRPO Training')
plt.grid(True)
plt.tight_layout()
loss_fig_path = os.path.join(SCRIPT_DIR, "loss_imgs", "loss_grpo.png")
plt.savefig(loss_fig_path)
