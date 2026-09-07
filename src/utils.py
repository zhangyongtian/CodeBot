import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()  # 评估模式

    # ❶ 将提示词转换为 token
    device = next(model.parameters()).device  # 获取模型参数所在的设备
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    # 保存已生成 token 的变量
    generated_ids = ids.clone()

    # ❷ token 生成循环
    for _ in range(max_new_tokens):
        # 超过上下文长度时，仅使用末尾部分
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]

        # 预测下一个 token
        logits = model(ids)[:, -1, :tokenizer.n_vocab]
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        # 生成结束 token 时停止
        if next_id.item() == tokenizer.eot_token:
            break

        # 追加新生成的 token
        ids = torch.cat((ids, next_id), dim=1)
        generated_ids = torch.cat((generated_ids, next_id), dim=1)

    # ❸ 解码后返回
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
