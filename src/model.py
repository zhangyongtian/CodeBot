import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, dropout_rate=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        # W_q: Query 投影矩阵 —— 将输入的 E 维嵌入同时投影出 H 个头各自的 Query 向量，
        #      输出扁平拼接为 H*D 维，后续由 .view(B,C,H,D) 拆分为 H 个独立的头。
        self.W_q = nn.Linear(E, H*D, bias=False)
        # W_k: Key 投影矩阵 —— 与 W_q 对称，为每个头生成独立的 Key 向量。
        self.W_k = nn.Linear(E, H*D, bias=False)
        # W_v: Value 投影矩阵 —— 与 W_q 对称，为每个头生成独立的 Value 向量。
        self.W_v = nn.Linear(E, H*D, bias=False)
        # W_o: 输出投影矩阵 —— 将多头注意力计算完成后拼接的 H*D 维结果
        #      线性投影回原始嵌入维度 E，完成多头信息的语义融合。
        self.W_o = nn.Linear(H*D, E, bias=False)

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 解包输入张量 x 的三维 shape：(Batch, Context, Embedding)
        #   B = batch_size：批次大小，一次前向传播中包含多少条序列
        #   C = context_len：序列长度，每条序列包含多少个 token
        #   E = embed_dim：每个 token 的嵌入向量维度
        B, C, E = x.shape
        H, D = self.n_head, self.head_dim

        # Q (Query 查询向量)：每个 token "我要找什么信息" 的表示
        #   投影后 shape=(B, C, H*D)：B×C 个 token 各自拥有 H 个头拼接的 Query
        Q = self.W_q(x)  # (B, C, H*D)
        # K (Key 键向量)：每个 token "我持有什么信息" 的表示，用于与 Query 计算相似度
        K = self.W_k(x)  # (B, C, H*D)
        # V (Value 值向量)：每个 token "我的实际内容信息"，相似度加权求和的对象
        V = self.W_v(x)  # (B, C, H*D)

        # 两步重排 Q/K/V：
        #   1) .view(B, C, H, D) —— 将 H*D 维的扁平向量拆分为 H 个头，每头 D 维
        #   2) .transpose(1, 2)  —— 把头维度 H 挪到 C 之前，使 H 成为 batch 维，
        #                           后续 matmul 可一次并行计算 H 个头的注意力分数
        Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

        # 计算 Query 与 Key 的点积相似度矩阵（= 自注意力分数张量）
        #   Q shape = (B, H, C, D)
        #   K.transpose(-2,-1) shape = (B, H, D, C)  ← 转置最后两维便于矩阵乘
        #   scores shape = (B, H, C_q, C_k) = (B, H, C, C)
        #     · 第 1 维 H：第几个注意力头（每个头独立学习一种相关性模式）
        #     · 倒数第 2 个 C = C_q：Query 端的 token 索引（谁在"看" / 发起查询）
        #     · 倒数第 1 个 C = C_k：Key   端的 token 索引（被"看" / 被查询）
        #   即 scores[b][h][i][j] 表示：第 b 条句子、第 h 个头中，
        #   第 i 个 token (Query) 与第 j 个 token (Key) 的未缩放点积相关性分数。
        #
        #   举例（单头 h、序列 = "我 喜欢 吃 苹果"，C=4）：
        #     scores[b][h] 是一张 4×4 的相关性热力图（行 i = 观察者，列 j = 被观察者）：
        #       i \ j  | token_0("我") | token_1("喜欢") | token_2("吃") | token_3("苹果")
        #       -------|---------------|-----------------|---------------|-----------------
        #       "我"   |   我→我       |    我→喜欢      |    我→吃      |    我→苹果
        #       "喜欢" |   喜欢→我     |    喜欢→喜欢    |    喜欢→吃    |    喜欢→苹果
        #       "吃"   |   吃→我       |    吃→喜欢      |    吃→吃      |    吃→苹果
        #       "苹果" |   苹果→我     |    苹果→喜欢    |    苹果→吃    |    苹果→苹果
        #   H 个头 → H 张独立的相关性图，比如头0学动宾关系、头1学指代关系等。
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, C, C)
        # 缩放点积：除以 √D，防止点积值过大导致 softmax 梯度消失
        scores = scores / (D ** 0.5)

        # 因果掩码（Causal Mask / Look-ahead Mask）：构造 C×C 的下三角矩阵
        #   作用：自回归语言模型(GPT)必须"只看过去和现在，不能偷看未来"。
        #   tril = triangle lower，规则：mask[i][j] = 1 当且仅当 j ≤ i
        #     · 下三角(含对角线)=1：允许 token_i 关注 token_0 ~ token_i（过去+当前）
        #     · 上三角=0        ：禁止 token_i 关注 token_{i+1} ~ token_{C-1}（未来）
        #
        #   举例（序列 = "我 喜欢 吃 苹果"，C=4）：
        #       i \ j  | 0("我") | 1("喜欢") | 2("吃") | 3("苹果")
        #       -------|---------|-----------|---------|-----------
        #       0 "我" |    1    |     0     |    0    |     0
        #       1 "喜欢" |    1    |     1     |    0    |     0
        #       2 "吃" |    1    |     1     |    1    |     0
        #       3 "苹果" |    1    |     1     |    1    |     1
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        # 按 mask 遮蔽未来位置：将 mask=0 的相关性分数替换为 -∞
        #   -∞ 经过 softmax 后严格等于 0，从而这些位置的注意力权重 = 0，
        #   相当于模型在预测 token_i 时完全"看不到"未来的 token_{i+1...}，
        #   保证训练时的并行计算与推理时的自回归生成行为一致。
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 自注意力第②步：softmax 把分数归一化为概率权重（每行和 = 1）
        #   dim=-1：对 Key 端（被观察者）维度做 softmax，
        #           即每个 token_i 的"注意力预算"（总和=1）分配给 j=0..C-1。
        weights = F.softmax(scores, dim=-1)  # (B, H, C, C)
        # 对注意力权重施加 Dropout：随机置零部分权重，防止模型过度依赖个别 token。
        weights = self.attention_dropout(weights)
        # 自注意力第③步：用注意力权重对 Value 做加权求和（上下文融合）
        #   weights shape = (B, H, C_q, C_k)
        #   V       shape = (B, H, C_k, D)
        #   hidden  shape = (B, H, C_q, D)  ← 每个 Query 位置得到融合了全句信息的新表示
        #
        #   数学含义（固定 b、h、i 时）：
        #     hidden[b][h][i] = Σ(j=0..C-1)  weights[b][h][i][j] * V[b][h][j]
        #   即 token_i 的输出表示 = 按"token_i 对 token_j 的关注度"作为权重，
        #   将所有 token_j 的 Value 向量加权求和，得到上下文感知的表示。
        #   因果掩码保证 j > i 的 weights = 0，因此未来 token 的 Value 不会被混入。
        #
        #   Q/K/V 角色分工：
        #     Q = "我问什么"（只参与打分，此处已退休）
        #     K = "我有什么索引"（只参与打分，此处已退休）
        #     V = "我真实的内容"（最终承载信息，被加权求和后输出）
        hidden = torch.matmul(weights, V)  # (B, H, C, D)

        # 把头维 H 从第 1 位挪回第 3 位，便于与序列维 C 对齐后做拼接；
        # .contiguous() 解决 transpose 带来的内存不连续问题，否则后续 .view 会报错
        hidden = hidden.transpose(1, 2).contiguous()  # (B, C, H, D)
        # 多头物理拼接：将 H 个 D 维的头按顺序首尾相接，拼成一条 H*D 维的长向量
        hidden = hidden.view(B, C, H * D)  # (B, C, H*D)
        # 输出投影 W_o：将拼接后的多头信息通过可学习线性层语义融合，
        # 并投影回原始嵌入维度 E，以支持残差连接 x = x + attn(x)
        output = self.W_o(hidden)  # (B, C, E)
        # 输出端 Dropout：防止模型过拟合，随机丢弃一部分神经元激活值
        output = self.output_dropout(output)

        return output


class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),  # GELU()
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim, dropout_rate=0.1):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = nn.LayerNorm(embed_dim)  # LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)  # LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ff_dim, dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, max_context_len, embed_dim, n_head, n_layer, ff_dim, dropout_rate):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_context_len = max_context_len
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_context_len, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim, n_head, ff_dim, dropout_rate)
            for _ in range(n_layer)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.unembed = nn.Linear(embed_dim, vocab_size)

        self.embed.weight = self.unembed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        B, C = ids.shape
        device = ids.device

        pos = torch.arange(0, C, dtype=torch.long, device=device)
        emb = self.embed(ids)
        pos_emb = self.pos_embed(pos)
        x = self.dropout(emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        logits = self.unembed(x)
        return logits

    def save(self, file_path):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_context_len': self.max_context_len,
            'embed_dim': self.embed_dim,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        }
        torch.save(checkpoint, file_path)

    @classmethod
    def load_from(cls, file_path, device='cpu'):
        checkpoint = torch.load(file_path, map_location=device)

        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_context_len=checkpoint['max_context_len'],
            embed_dim=checkpoint['embed_dim'],
            n_head=checkpoint['n_head'],
            n_layer=checkpoint['n_layer'],
            ff_dim=checkpoint['ff_dim'],
            dropout_rate=checkpoint['dropout_rate']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model