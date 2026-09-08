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
                # ──────────────────────────────────────────────────────────────
                # 构造「问题-答案」对的一个样本。
                # 举例：当 i=3, j=5 时：
                #   prompt = "### Instruction:\n3+5=\n\n### Response:\n"
                #              即 Alpaca 风格的输入提示，要求模型填空 "3+5=?"
                #   ground_truth = 8  (int 类型，正确的数学答案)
                #   之后 append 到 self.data 的元组：
                #     ("### Instruction:\n3+5=\n\n### Response:\n", 8)
                # ──────────────────────────────────────────────────────────────
                prompt = f"### Instruction:\n{i}+{j}=\n\n### Response:\n"
                ground_truth = i + j
                self.data.append((prompt, ground_truth))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_batch(self, prompts, responses, device):
        all_ids = []   # 存放每个样本的完整 token ID 列表（prompt+response，未填充）
        all_masks = [] # 存放每个样本对应的损失掩码（未填充）

        for prompt, response in zip(prompts, responses):
            # ── 具体数据形态示例（GPT-2 BPE 词表规则确定，无需运行即可推断） ──
            # 取 prompt="### Instruction:\n3+5=\n\n### Response:\n"、response="8"：
            #   prompt_ids (list[int], 长度≈14)：
            #     [2922,      # "###"
            #      13093,     # " Instruction"  （带前置空格）
            #      25,        # ":"
            #      198,       # "\n"
            #      18,        # "3"
            #      10,        # "+"
            #      30,        # "5"
            #      28,        # "="
            #      198,       # "\n"
            #      198,       # "\n"
            #      2922,      # "###"
            #      18261,     # " Response"    （带前置空格）
            #      25,        # ":"
            #      198]       # "\n"
            #   规律：单个数字(0-9)、#、换行、+、=、空格在 GPT-2 BPE 中都是单 token；
            #         常见英文整词（Instruction / Response）也作为单 token 存储。
            #
            #   response_ids (list[int], 长度≈1~2)：
            #     一位数答案如 "8"  → [26]             （单 token）
            #     两位数答案如 "17" → [16, 28]         （"1" + "7" 两个 token 拼接）
            # ─────────────────────────────────────────────────────────────
            prompt_ids = self.tokenizer.encode(
                prompt, allowed_special={"<|endoftext|>"})
            # response 是模型生成的原始文本（如 "8"、"17" 等），同样分词为 token IDs
            response_ids = self.tokenizer.encode(
                response, allowed_special={"<|endoftext|>"})

            # ─────────────────────────────────────────────────────────────
            # 步骤 2：拼接 tokens，并构造「损失掩码 mask」
            #  ids  = [prompt_tokens  ... | response_tokens  ...]
            #  mask = [0, 0, ..., 0       | 1, 1, ..., 1     ]
            #        └── prompt 部分不计算损失 ┘└── response 部分才计算损失 ┘
            #
            # 为什么 prompt 部分要掩码掉？
            # GRPO 只奖励/惩罚模型「生成的回答」，而 prompt 是给定的输入上下文，
            # 它的出现概率不应参与策略梯度的更新（否则会把模型本身不需要预测的内容也纳入损失）。
            #
            # ── 具体数据形态示例（沿用上面 "3+5=?" "8" 的例子） ──
            #   ids（15 个整数）：
            #     [2922,13093,25,198,18,10,30,28,198,198,2922,18261,25,198, 26]
            #      └──────────────────── 14 个 prompt token ────────────────┘ └─ 1 个 response token("8")
            #   mask（15 个 0/1，一一对应 ids 的每个位置）：
            #     [0,   0,    0, 0,  0, 0, 0, 0, 0,  0,  0,   0,    0, 0,   1]
            # ─────────────────────────────────────────────────────────────
            ids = prompt_ids + response_ids
            mask = [0] * len(prompt_ids) + [1] * len(response_ids)

            all_ids.append(ids)
            all_masks.append(mask)
            
        # ─────────────────────────────────────────────────────────────────
        # 步骤 3：Padding 填充对齐 —— 让批次中每个样本的长度一致（取最大值）
        # DataLoader 要求 batch 内每个 Tensor 形状相同，因此短序列末尾补 0。
        # 这里的 0 只是纯填充值（并不代表真实词表中的 token），
        # 因为对应的 mask 会同时补 0，因此这些填充位不会参与损失计算，对训练无影响。
        #
        # ── 具体数据案例：假设 all_ids / all_masks 里有 2 个样本 ──
        # 样本 A（短，回答 "8"，单 token）：
        #   ids_A  = [2922,13093,25,198,18,10,30,28,198,198,2922,18261,25,198, 26]  (len=15)
        #   mask_A = [0,   0,    0, 0,  0, 0, 0, 0, 0,  0,  0,   0,    0, 0,   1]
        #
        # 样本 B（长，回答 "17"，双 token "1"+"7"）：
        #   ids_B  = [2922,13093,25,198,19,10,26,28,198,198,2922,18261,25,198, 16, 28]  (len=16)
        #   mask_B = [0,   0,    0, 0,  0, 0, 0, 0, 0,  0,  0,   0,    0, 0,   1,  1]
        #
        # → max_len = max(15, 16) = 16
        #
        # 样本 A 需要补 pad_len=16-15=1 个 0：
        #   padded_ids_A   = [2922,13093,25,198,18,10,30,28,198,198,2922,18261,25,198,26, 0]
        #   padded_masks_A = [0,   0,    0, 0,  0, 0, 0, 0, 0,  0,  0,   0,    0, 0, 1, 0]
        #                                                                └─ 补了 1 个 0 ┘
        #
        # 样本 B 已经是最长，补 pad_len=0 个 0：
        #   padded_ids_B   = [2922,13093,25,198,19,10,26,28,198,198,2922,18261,25,198,16,28]
        #   padded_masks_B = [0,   0,    0, 0,  0, 0, 0, 0, 0,  0,  0,   0,    0, 0, 1,  1]
        #
        # 最终 padded_ids / padded_masks 中的每个子 list 长度都 = 16，
        # 才能拼成 (2, 16) 的矩形 Tensor。
        # ─────────────────────────────────────────────────────────────────
        max_len = max(len(ids) for ids in all_ids)   # 批次中最长序列的 token 数
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(all_ids, all_masks):
            pad_len = max_len - len(ids)             # 当前样本需要补的 0 的个数
            padded_ids.append(ids + [0] * pad_len)   # 序列尾部填充 0
            padded_masks.append(mask + [0] * pad_len)# 填充位置的 mask 也设为 0（不参与损失）


        ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        mask = torch.tensor(padded_masks, dtype=torch.float, device=device)

        return ids, mask


# 奖励函数：判断模型生成的回答是否与真实答案一致，返回 0/1 二元奖励
#   ground_truth: int，正确的数学答案（如 8）
#   response    : str，模型生成的原始文本（如 "8"、"答案是17"、"abc" 等）
#   思路：用正则提取 response 中所有整数，取**最后一个**作预测值，与 ground_truth 对比
#   例：ground_truth=8, response="答案是 8" → matches=["8"]     → predicted=8 → 奖励 1.0
#   例：ground_truth=17,response="15 不对，应该是 17"→ matches=["15","17"]→取最后 17→奖励 1.0
#   例：ground_truth=3, response="xyz"       → matches=[]      → 无数字    → 奖励 0.0
def calculate_reward(ground_truth, response):
    try:
        matches = re.findall(r'(-?\d+)', response)
        if matches:
            predicted = int(matches[-1])  # 获取最后一个数值
            return 1.0 if predicted == ground_truth else 0.0
        return 0.0
    except:
        return 0.0


# 组生成（Group Sampling）：对每个 prompt 采样 group_size 个回答，再按组计算优势值
#   输入：
#       prompts    : 一批问题（长度 = batch_size，如 4 道加法题）
#       gts        : 与 prompts 一一对应的真实答案 int 列表，仅用于事后算奖励，不进模型
#       group_size : 每个 prompt 独立采样的回答数（如 8），用于 GRPO「组内归一化」
#   输出：
#       all_prompts   : 长度 = batch_size × group_size（如 4×8=32），prompt 被重复 group_size 次
#       all_responses : 长度同上，每个 prompt 对应的 group_size 个采样回答
#       all_advantages: (batch_size × group_size,) 的 Tensor，组内优势 = reward - 同组 reward 均值
#
#   例（取 batch_size=1, group_size=8, prompt="3+5=?", gt=8）：
#     采样 8 次 → responses = ["8", "17", "8", "6", "8", "5", "13", "8"]  （temperature=1.0 所以各不相同）
#     算奖励    → rewards   = [ 1,   0,   1,   0,   1,   0,   0,   1 ]    （答对给 1，答错给 0）
#     均值 mean = (1+0+1+0+1+0+0+1)/8 = 0.5
#     优势值    → advantages = rewards - 0.5 = [+0.5,-0.5,+0.5,-0.5,+0.5,-0.5,-0.5,+0.5]
#       含义：组内「比平均好」的回答（答对）获得正优势，损失下降时概率↑；
#             组内「比平均差」的回答（答错）获得负优势，损失下降时概率↓。
def generate_group(model, tokenizer, prompts, gts, group_size):
    all_prompts = []
    all_responses = []
    all_advantages = []

    for prompt, gt in zip(prompts, gts):
        # ── 对当前 prompt 独立采样 group_size 次（temperature=1.0 → 高随机性，保证答案多样性） ──
        responses = []
        for _ in range(group_size):
            full_text = generate(model, tokenizer, prompt, temperature=1.0)
            # generate() 返回的是「prompt 原文 + 模型新生成的全部回答」拼接好的完整字符串；
            # 这里用 full_text[len(prompt):] 按字符长度把开头的 prompt 整体切掉，
            # 只保留模型在生成阶段真正新输出的回答部分（例如 "8"、"17" 等）。
            # 例：prompt = "### Instruction:\n3+5=\n\n### Response:\n"
            #     full_text = "### Instruction:\n3+5=\n\n### Response:\n8"
            #     response  = "8"
            response = full_text[len(prompt):]
            responses.append(response)

        # ── 用 ground_truth 给这组回答打分（仅做评分员，不进模型 forward） ──
        #   rewards shape: (group_size,)，每个元素是 0.0 或 1.0
        rewards = torch.tensor([calculate_reward(gt, r) for r in responses])
        # ── GRPO 组内归一化优势：reward 减去本组 reward 的均值 ──
        #   优点：不需要额外训练 Critic/Value 网络，直接用组均值做 baseline，
        #        保证一组内优势值的均值为 0，正负样本自然平衡。
        advantages = rewards - rewards.mean()

        # ── 把「prompt、对应的回答、优势值」一一配对，平铺展开到总列表中 ──
        #   若 batch_size=4, group_size=8 → 此处循环 4×8=32 次，all_prompts 等列表长度变 32
        for response, advantage in zip(responses, advantages):
            all_prompts.append(prompt)
            all_responses.append(response)
            all_advantages.append(advantage)

    # all_advantages 目前是 list[Scalar Tensor]，用 stack 拼成 (B×group_size,) 的一维 Tensor
    return all_prompts, all_responses, torch.stack(all_advantages)

# 损失函数


def compute_probs(model, ids):
    """
    目标：在 ids 序列的每个位置 i，取出「模型预测下一个真实 token ids[i+1] 的概率」。
          返回 (B, C-1) 给 PPO/GRPO 算 ratio = π_new / π_old。

    用具体数字 ids=[18,10,30,26,0]（即 "3","+","5","8",pad，来自 get_batch）理解：
        token_probs 返回 C-1=4 个概率：
          [ p("+"|"3"),   p("5"|"3+"),   p("8"|"3+5"),   p(pad|"3+5=8") ]
            0.95           0.92           0.78 ★          0.01
          mask=[ 0,           0,             1,               0     ]
        （只有 ★ 那一位是 response 段，后续乘 mask=1 才真正参与损失；
         prompt/pad 位的概率虽然也算出来了，但乘 mask=0 被丢掉，不影响结果。）

    内部 4 步：
        ① model(ids)              → (B, C,    V)   一次前向，得到所有位置 logits
        ② softmax(logits[:,:-1,:])→ (B, C-1,  V)   丢末尾位（它后面没 token 可预测），变概率
        ③ labels = ids[:,1:]      → (B, C-1)       右移一位 = 每个位置要预测的真实 token ID
        ④ gather(probs, labels)   → (B, C-1)       对每个位置 i，只取「真实 token ids[i+1] 对应那格概率」
    """
    logits = model(ids)
    probs = F.softmax(logits[:, :-1, :], dim=-1)
    labels = ids[:, 1:]
    token_probs = torch.gather(
        probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
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
loss_fig_path = os.path.join(PROJECT_ROOT, 'loss_imgs', "loss_grpo.png")
plt.savefig(loss_fig_path)
