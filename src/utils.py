import torch
import torch.nn.functional as F


# 禁用梯度计算：推理/生成阶段不需要反向传播，不计算梯度 → 节省显存 + 加速
# 作用等价于 with torch.no_grad(): 上下文管理器，装饰器形式更简洁
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()  # 评估模式：关闭 Dropout / BatchNorm 训练时特有行为（注意和 no_grad 不同，两者配合使用）

    # ❶ 将提示词转换为 token
    device = next(model.parameters()).device  # 获取模型参数所在的设备
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    # 保存完整生成序列（与 ids 分工不同，必须独立克隆）
    #   ids:           作为模型当前步的输入窗口，超过 max_context_len 会被截断（会丢前面）
    #   generated_ids: 始终完整保留 prompt + 所有已生成 token，用于最后解码输出
    #   必须 .clone()：若直接赋值 = ids，则两者指向同一块内存，ids 被截断时 generated_ids 也会丢失内容
    generated_ids = ids.clone()

    # ❷ token 生成循环
    for _ in range(max_new_tokens):
        # 滑动窗口截断：当序列超过模型最大上下文长度时，仅保留末尾部分继续推理
        # ids.size(1)           当前已生成序列长度（token 数）
        # model.max_context_len  Transformer 模型单次前向传播支持的最大窗口大小
        # [:, -L:]              只取最后 L 个 token 作为当前步输入（丢弃前面超出的部分）
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]

        # 预测下一个 token：前向传播 → 取最后位置预测 → 截断到有效词表
        # model(ids):     (1, N, vocab_size)  每个位置预测下一个词的分数
        # [:, -1, :]:     (1, vocab_size)     仅取序列末尾位置（预测"下一个词"用）
        # [:n_vocab]:     (1, n_vocab)        截断到 tokenizer 的实际词表大小
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
