import torch
import torch.nn as nn
import torch.nn.functional as F
#
# ==============================================================================
# 模型架构总览：类的包含关系 & 数据流转（从输入 token ids → 输出 vocab logits）
# ==============================================================================
#
# 【包含层级（从外到内，上到下）】
#
#   GPT (最外层大模型)
#     ├── nn.Embedding(vocab_size, embed_dim)        # 词嵌入：token id → E 维向量
#     │                                                  （离散符号→连续空间，语义相近的词向量也近）
#     ├── nn.Embedding(max_context_len, embed_dim)   # 位置嵌入：位置索引 → E 维向量
#     │                                                  （注意力本身无顺序，需单独编码每个位置的位置）
#     ├── nn.Dropout(dropout_rate)                   # 嵌入后 Dropout（随机丢弃神经元，防止过拟合）
#     ├── ModuleList([Block] × n_layer)              # N 层 Transformer Block（核心堆叠，逐层抽象语义）
#     │     └── Block (每层结构相同，Pre-LN 残差)
#     │           ├── nn.LayerNorm(E)          ──┐
#     │           │                                  （对每个 token 的 E 维做均值0方差1，稳定数值分布）
#     │           ├── MultiHeadAttention          │  = x + attn(norm1(x))  残差①
#     │           │     ├── W_q / W_k / W_v       │    └─ Q/K/V 三种独立投影 → .view 拆分多头并行计算
#     │           │     ├── attention_dropout     │       → 因果掩码(防偷看未来) + softmax(归一化为权重) + 权重·V
#     │           │     └── W_o + output_dropout  │       → .transpose+.view 拼接多头 + W_o 融合回 E 维
#     │           ├── nn.LayerNorm(E)          ──┘
#     │           └── FFN
#     │                 ├── Linear(E→ff_dim) + GELU    = x + ffn(norm2(x))  残差②
#     │                 │                                （升维 + 非线性激活，单 token 内部特征加工）
#     │                 └── Linear(ff_dim→E) + Dropout  （降维回 E，便于残差相加 + Dropout 正则）
#     ├── nn.LayerNorm(embed_dim)                    # 所有 Block 之后的 Final Norm（输出前最后一次数值规整）
#     └── nn.Linear(embed_dim, vocab_size) [unembed]  # 嵌入 → vocab 维度的 logits
#                                                        （与 embed 共享权重，映射回词表概率分数）
#
# 【数据流转（shape 变化，B=batch, C=seq_len, E=embed_dim, V=vocab_size）】
#
#   ids (B, C)   token id 序列（离散整数）
#     │
#     ├── embed(ids)         ──→ (B, C, E)  词嵌入（每个 id 查表得到 E 维向量，语义空间连续化）
#     ├── pos_embed(0..C-1) ──→ (   C, E)  位置嵌入（每个位置序号查表得到 E 维向量，告诉模型"这是第几个词"）
#     │  两者相加 + dropout      (B, C, E)  ← 送入 Block 堆（残差路径保证原始信息无损传递）
#     ↓
#   Block × n_layer            (B, C, E)  ← 每层内部：
#     │                              ① norm1(稳定分布) → MHA(跨 token 融合上下文) → +x(残差，梯度回传顺畅)
#     │                              ② norm2(稳定分布) → FFN(单 token 非线性加工) → +x(残差，梯度回传顺畅)
#     ↓
#   Final LayerNorm            (B, C, E)   稳定数值（深层堆叠后分布可能偏移，最后一次归一化）
#     ↓
#   unembed Linear             (B, C, V)   映射到词表大小的分数（logits）
#                                      ↓ softmax 即可得到下一个 token 的概率分布（逐词自回归采样生成）
#
# 【各模块设计动机与好处（Why this design?）】
#
# · 词嵌入 + 位置嵌入：
#   ├── 词嵌入：把离散的 token id 映射到连续向量空间 → 好处：让语义相似的词向量也相似（"猫"接近"狗"），
#   │             同时把离散符号转为神经网络能处理的实数运算。
#   └── 位置嵌入：注意力本身不携带顺序信息（"A打B"和"B打A"在纯注意力下等价），
#                 必须单独给每个位置一个可学习的向量相加 → 好处：显式编码位置顺序，模型才能区分语序。
#
# · Dropout（嵌入后 / 注意力权重 / 输出）：
#   训练时随机丢弃部分神经元 → 好处：防止模型过度依赖某些特定的激活或注意力连接，缓解过拟合，
#   提升泛化能力。嵌入后 + 权重 + 输出三处分别针对信息流的不同阶段施加正则。
#
# · 堆叠多层 Block（× n_layer）：
#   类似人"反复阅读、多层抽象"的思维 → 好处：第1层学简单的相邻词依赖（如动宾搭配），
#   第k层学更高级的跨句语义（如指代消解、长距离依赖），层数越深表示越抽象。
#
# · Block 内部两次残差连接（x + attn(...) / x + ffn(...)）：
#   直接把子层输入和输出逐元素相加 → 好处：① 解决深层网络的"梯度消失"问题
#   （梯度可以沿残差支路无衰减地传回浅层）；② 不改变维度，下一层直接接着用；
#   ③ 相当于"在原始表示上做增量修改"，保留已有信息的同时学习新信息。
#
# · Pre-LN 结构（先 Norm 再进子层，而非子层后再 Norm）：
#   norm1(x) → attn → +x （GPT 风格），而不是 attn(x) → norm → +x（原始 Transformer 风格）
#   → 好处：深层模型训练更稳定，不需要复杂的学习率 warm-up 技巧也能收敛；
#   残差支路始终保持"未归一化"的梯度流通性。
#
# · LayerNorm（每层 2 个 + 最终 1 个）：
#   对每个 token 的 E 维做均值=0、方差=1，再用 gamma/beta 缩放偏移 → 好处：
#   ① 稳定每层输入的数值分布，避免深层网络激活越来越大/小；
#   ② 对 batch 内序列长度不敏感（和 BatchNorm 不同），NLP 场景更合适；
#   ③ Final Norm 在输出前做最后一次规整，保证送入 unembed 的数值分布稳定。
#
# · MultiHeadAttention（多头自注意力）：
#   把 E 维拆成 H 个独立的 D 维小空间分别算注意力，最后拼接 → 好处：
#   ① 同时学习多种相关性模式（头0学语法、头1学指代、头2学位置等），表达能力更强；
#   ② 单头注意力一次只能聚焦一种"相关"，多头相当于多视角综合判断；
#   ③ 通过 view+transpose 实现并行，无需 for 循环，GPU 一次 matmul 算完所有头。
#
# · Q/K/V 三种不同投影（而不是共用同一个）：
#   同一 token x 分别乘三个不同的权重矩阵 → 好处：三种角色各司其职，
#   Q 表达"我要找什么"，K 表达"我有什么可被找的特征"，V 表达"我真正的内容"。
#   若共用一个矩阵，查询与索引无法解耦，注意力表达力大幅下降。
#
# · 缩放点积（除以 √D）：
#   scores = Q·K^T / √D → 好处：D 较大时点积绝对值会很大（比如 D=64 时方差≈64），
#   直接送 softmax 会进入梯度趋近 0 的饱和区（权重接近 one-hot）。除以 √D 把方差拉回 1，
#   保证 softmax 梯度健康、注意力分布平滑。
#
# · 因果掩码（下三角 tril + 填 -∞）：
#   禁止 token_i 看到 j > i 的未来 token → 好处：训练时可以把整段序列一次性喂入并行计算
#   （训练效率高），同时保证训练时的预测行为与推理时严格自回归（逐词生成）的行为完全一致，
#   不会出现"考试偷看答案"导致的 train-infer mismatch。
#
# · W_o 输出投影（H*D → E）：
#   多头拼接后再乘一个可学习矩阵 → 好处：单纯 concat 只是把多头"摆在一起"，
#   W_o 通过线性组合把各头学到的不同模式**融合**成一个综合表示，同时对齐回 E 维以便残差相加。
#
# · FFN 前馈网络（E → ff_dim → E，中间 GELU）：
#   两层 MLP 夹一个激活，通常 ff_dim = 4×E → 好处：注意力负责"在不同 token 之间搬信息"
#   （跨 token 交互），FFN 负责"对每个 token 单独做非线性变换"（token 内部特征加工），
#   两者互补。先升维再降维 + GELU 非线性，提供充足的表达能力（类似大脑锥体细胞的树突加工）。
#
# · GELU 激活（替代 ReLU）：
#   0.5x·(1 + tanh(√(2/π)(x + 0.044715x³))) → 好处：相比 ReLU 的硬截断，
#   GELU 是"带随机性的软门控"（近似 Dropout 的预期效果），
#   平滑处处可导，训练更稳定，是 GPT/BERT 等现代 Transformer 的标配激活。
#
# · unembed 权重与 embed 权重共享（embed.weight = unembed.weight）：
#   词嵌入矩阵和最终线性层共用同一份权重 → 好处：① 参数减少一半（参数量 = V×E 而不是 2VE），
#   ② 语义一致："输出哪个词"的分类方向与"输入该词时的嵌入方向"对齐，通常带来更好的泛化。
#
# · _init_weights 正态初始化（std=0.02）：
#   Linear/Embedding 用 N(0, 0.02) 初始化 → 好处：和 Transformer 论文/GPT-2 官方实现保持一致，
#   保证前几层激活的方差不会过大（导致训练不稳定）也不会过小（梯度消失），
#   是收敛速度和最终性能的经验最优值。
# ==============================================================================



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
    """Layer Normalization：对每个样本、每个 token 的嵌入维度独立做归一化。
    公式：y = gamma * (x - mean) / sqrt(var + eps) + beta
    作用：稳定深层网络的数值分布，避免激活值过大/过小；gamma/beta 可学习，恢复表达能力。
    """
    def __init__(self, embed_dim):
        super().__init__()
        # gamma (缩放参数)：shape=(E,)，初始化为 1，训练中学习每个维度的"理想标准差倍数"
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        # beta  (偏移参数)：shape=(E,)，初始化为 0，训练中学习每个维度的"理想均值偏移"
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        # eps 数值稳定性常数：防止 var≈0 时出现除零错误
        self.eps = 1e-5

    def forward(self, x):
        # 输入 x shape 通常为 (B, C, E)
        # dim=-1：对最后一维（嵌入维度 E）求均值，即每个 token 独立算自己的均值；
        # keepdim=True 保留维度为 (B, C, 1)，方便后续广播减法
        mean = x.mean(dim=-1, keepdim=True)
        # 计算每个 token 的 E 维方差；unbiased=False 使用有偏估计（除以 N），
        # 与 PyTorch 官方 nn.LayerNorm 行为保持一致
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 标准化：先减均值使均值=0，再除以 √(方差+eps) 使方差≈1；广播后 shape 仍为 (B, C, E)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 可学习仿射变换：逐维度缩放（gamma）+ 偏移（beta），恢复网络需要的分布幅度；
        # 初始 gamma=1、beta=0 等价于恒等映射，训练中逐步学到最佳归一化强度
        return self.gamma * norm_x + self.beta


class GELU(nn.Module):
    """Gaussian Error Linear Unit 激活函数。
    近似公式：GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x³)))
    含义：x 乘以"x 大于高斯噪声的概率"（≈ 0.5*(1+erf(x/√2)) 的 tanh 近似），
    是一种平滑、处处可导的"软门控"激活，相比 ReLU 硬截断更稳定，是 GPT/BERT 等 Transformer 的标配。
    """
    def forward(self, x):
        # cdf 近似项：√(2/π) ≈ 0.7979，将输入经三次项微调后送入 tanh，
        # 近似标准正态分布的累积分布函数 Φ(x) = P(N(0,1) ≤ x)
        cdf_term = torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        # GELU(x) = x * Φ(x)  的 tanh 近似：
        #   x > 0 时 Φ(x)≈1 → 输出≈x（近似通过）；
        #   x < 0 时 Φ(x)≈0 → 输出≈0（近似被门控掉但保留微小斜率，无梯度死亡）；
        #   整体处处可导且平滑，训练比 ReLU 更稳定
        return 0.5 * x * (1 + torch.tanh(cdf_term))


class FFN(nn.Module):
    """Feed-Forward Network 前馈网络（Position-wise FFN）。
    结构：Linear(E → ff_dim) → GELU → Linear(ff_dim → E) → Dropout
    作用：与 MultiHeadAttention 互补 —— MHA 在"不同 token 之间搬信息"（跨 token 交互），
    而 FFN 对"每个 token 单独"做非线性特征加工（token 内特征变换），互不干扰。
    通常 hidden_dim = 4 × embed_dim（Transformer 论文经验值，升维再降维提供充足表达能力。
    """
    def __init__(self, embed_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.layers = nn.Sequential(
            # 第1层：升维线性变换 E → hidden_dim，将 token 的表示投射到更高维的特征空间，
            # 以便后续非线性可以学习更丰富的特征组合
            nn.Linear(embed_dim, hidden_dim),
            # GELU 非线性激活：提供非线性表达能力（类似神经元的门控开关），
            # 相比 ReLU 更平滑，训练更稳定
            nn.GELU(),
            # 第2层：降维线性变换 hidden_dim → E，将高维特征压缩回原始嵌入维度，
            # 以便 Block 尾部的残差连接 x + ffn(x) 维度匹配
            nn.Linear(hidden_dim, embed_dim),
            # Dropout：随机丢弃部分输出神经元，正则化，防止过拟合
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # 直接走 Sequential 流水线；输入输出 shape 相同 = (B, C, E)，与 MHA 一致
        # 注意：Linear 作用于最后一维，所以所有 token 独立通过同一组权重，
        # 即"逐位置 (position-wise)"——位置间互不交流
        return self.layers(x)


class Block(nn.Module):
    """Transformer Block（Pre-LN 残差结构，GPT 标准做法）。
    子层顺序：Norm → 子层(MHA/FFN) → 残差相加，而不是 Post-LN 的 子层→Norm→残差。
    Pre-LN 的好处：深层模型训练更稳定，无需复杂 warm-up 即可收敛。
    """
    def __init__(self, embed_dim, n_head, ff_dim, dropout_rate=0.1):
        super().__init__()
        # 每个注意力头的维度 = 嵌入维度 ÷ 头数（保证 H*D = E，拼接后维度对齐残差）
        head_dim = embed_dim // n_head
        # norm1：进入 MultiHeadAttention 之前的 LayerNorm（Pre-LN 第一处）
        #   输入 shape = (B, C, E) → 输出 shape = (B, C, E)，数值分布被归一化到均值0方差1
        self.norm1 = nn.LayerNorm(embed_dim)  # LayerNorm(embed_dim)
        # attn：多头自注意力子层
        #   输入 shape = (B, C, E) → 输出 shape = (B, C, E)
        #   返回值：每个 token 融合了句子中所有可见 token（≤自己位置）信息的上下文感知表示
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, dropout_rate)
        # norm2：进入 FFN 之前的 LayerNorm（Pre-LN 第二处）
        #   输入 shape = (B, C, E) → 输出 shape = (B, C, E)
        self.norm2 = nn.LayerNorm(embed_dim)  # LayerNorm(embed_dim)
        # ffn：前馈网络子层（Position-wise FFN）
        #   输入 shape = (B, C, E) → 输出 shape = (B, C, E)
        #   返回值：每个 token 独立经过升维→GELU→降维后的非线性变换表示，
        #   与注意力的"跨 token 搬信息"形成互补
        self.ffn = FFN(embed_dim, ff_dim, dropout_rate)

    def forward(self, x):
        # 残差支路①：先 Norm 再 MHA，再与原始 x 相加
        #   self.norm1(x)      → (B, C, E)  归一化后的 x
        #   self.attn(...)     → (B, C, E)  注意力输出：融合了上下文信息
        #   x + attn(...)      → (B, C, E)  残差相加：原始 x + 上下文增量
        #     好处：梯度直接走残差支路传回浅层，解决深层网络梯度消失
        x = x + self.attn(self.norm1(x))
        # 残差支路②：先 Norm 再 FFN，再与上一步结果相加
        #   self.norm2(x)      → (B, C, E)  对上一步输出做第二次归一化
        #   self.ffn(...)      → (B, C, E)  FFN 输出：单 token 非线性加工
        #   x + ffn(...)       → (B, C, E)  残差相加：保持维度不变，信息流畅通
        x = x + self.ffn(self.norm2(x))
        # 返回整层 Block 处理完的表示，维度和输入完全一致 (B, C, E)，
        # 这样下一层 Block 可以直接复用，无需额外维度对齐
        return x


class GPT(nn.Module):
    """GPT 自回归语言模型主类（Decoder-Only Transformer）。
    核心结构：词嵌入 + 位置嵌入 → n_layer 个 Transformer Block（Pre-LN 残差）
           → Final LayerNorm → unembed（权重共享）→ vocab 维度 logits。
    """
    def __init__(self, vocab_size, max_context_len, embed_dim, n_head, n_layer, ff_dim, dropout_rate):
        super().__init__()
        # --- 保存超参数（用于 save/load 时恢复模型结构）---
        self.vocab_size = vocab_size      # 词表大小 V：token id 的取值范围是 [0, V-1]
        self.max_context_len = max_context_len  # 最大上下文长度：训练/推理时允许的最长序列
        self.embed_dim = embed_dim        # 嵌入维度 E：每个 token 的向量表示维度
        self.n_head = n_head              # 注意力头数 H：MultiHeadAttention 并行头数
        self.n_layer = n_layer            # Transformer Block 堆叠层数：决定模型深度
        self.ff_dim = ff_dim              # FFN 隐层维度：通常取 4×E，升维增强表达能力
        self.dropout_rate = dropout_rate  # Dropout 概率：正则化强度

        # --- 嵌入层 ---
        # 词嵌入矩阵：把离散的 token id (整数) 映射为连续的 E 维语义向量
        #   输入 shape = (B, C) 整数 → 输出 shape = (B, C, E) 浮点数
        #   本质：查表操作，weight shape = (vocab_size, E)
        #   语义相近的词，其向量在空间中也接近（训练中学到）
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 位置嵌入矩阵：把"第几个位置的 token"映射为 E 维位置向量
        #   输入 shape = (C,) 整数 [0,1,..C-1] → 输出 shape = (C, E)
        #   本质：可学习的位置编码，显式告诉模型 token 的顺序
        #   作用：注意力本身无序（对排列等变），必须靠位置编码提供语序信息
        self.pos_embed = nn.Embedding(max_context_len, embed_dim)
        # 嵌入融合后的 Dropout：嵌入层 + 位置嵌入相加后随机部分输出，防止过拟合
        self.dropout = nn.Dropout(dropout_rate)

        # --- Transformer Block 堆叠 ---
        # n_layer 层的 Block 列表，逐层处理；每层输入输出 shape 都是 (B, C, E)
        # 用 ModuleList 而非普通 list：这样参数会被注册到模型中，model.parameters() 才能拿到
        self.blocks = nn.ModuleList([
            Block(embed_dim, n_head, ff_dim, dropout_rate)
            for _ in range(n_layer)
        ])

        # --- 输出端 ---
        # Final LayerNorm：所有 Block 走完之后的最后一次归一化
        #   Pre-LN 结构残差分支始终没被 Norm，输出端数值分布可能漂移，
        #   用这一层把 (B, C, E) 再拉回均值 0 方差 1，稳定送入 unembed
        self.norm = nn.LayerNorm(embed_dim)
        # unembed 投影层：把 E 维语义向量 → vocab_size 维 logits（每个词的分数）
        #   weight shape = (vocab_size, embed_dim)（nn.Linear 的 weight 是 out×in）
        #   作用：相当于"反向查表"，给每个词表位置打分，后续 softmax 得到词概率
        self.unembed = nn.Linear(embed_dim, vocab_size)

        # 权重共享（Weight Tying）：让词嵌入与 unembed 分类矩阵共用同一份权重
        #   - 省一半参数（两个 (V,E) 大矩阵合并为一个）
        #   - 语义对齐："词 i 被编码成什么向量"和"什么向量被解码为词 i"完全一致，泛化更好
        #   - 正则化：约束自由度，抑制过拟合；GPT-2/GPT-3/LLaMA 均采用
        self.embed.weight = self.unembed.weight
        # 递归初始化所有子模块参数：model.apply(fn) 会深度优先遍历所有子 module
        #   覆盖：所有 nn.Linear(W_q/W_k/W_v/W_o/FFN/unembed) + nn.Embedding(embed/pos_embed)
        #   保证整个模型使用 GPT-2 官方推荐的 std=0.02 正态初始化开局
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """GPT-2 风格的参数初始化：Linear/Embedding 权重用 N(0, 0.02²)，Linear bias 置 0。
        被 self.apply(...) 递归调用到每个子模块上。
        """
        if isinstance(module, nn.Linear):
            # 线性层权重：均值为 0、标准差 0.02 的正态分布初始化
            # 选择 0.02 的原因：在 E=768 等常用尺度下，初始激活方差≈1，配合 LayerNorm 稳定
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # 线性层偏置：全 0 初始化（不提供初始偏移，让残差一开始就是 identity）
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 嵌入层（词嵌入 + 位置嵌入）：与 Linear 相同的 N(0, 0.02²) 初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        """前向传播：输入 token id (B, C)，输出下一个词的 logits (B, C, V)。
        注意：logits[b][c] 表示"在第 b 条句子的前 c+1 个 token 上下文下，预测第 c+1 个 token 的分数"，
        训练时与 ids[b][c+1] 做交叉熵 loss（自回归目标）。
        """
        # 从输入 id 张量中解包 batch 大小 B 和当前序列长度 C
        B, C = ids.shape
        device = ids.device

        # 生成位置索引 [0, 1, 2, ..., C-1]，作为位置嵌入的输入
        pos = torch.arange(0, C, dtype=torch.long, device=device)
        # 词嵌入：查 (B, C) 次表，得到每个 token 的语义向量 (B, C, E)
        emb = self.embed(ids)
        # 位置嵌入：查 C 次表，得到每个位置的向量 (C, E)，广播到 (B, C, E) 与 emb 相加
        pos_emb = self.pos_embed(pos)
        # 融合：语义 + 位置 逐元素相加，再经 Dropout 正则后得到模型输入 x (B, C, E)
        x = self.dropout(emb + pos_emb)

        # 逐层通过 n_layer 个 Transformer Block；每层保持 shape = (B, C, E) 不变
        # 每一层：MHA 跨 token 融上下文 → FFN 单 token 做非线性加工，信息逐级抽象
        for block in self.blocks:
            x = block(x)
        # Final LayerNorm：n_layer 层残差累加后数值可能飘，统一拉回标准分布 (B, C, E)
        x = self.norm(x)

        # ──────────────────────────────────────────────────────────────────
        # unembed 层（反向词嵌入）：E 维语义向量 → V 维词表打分（logits）
        # ──────────────────────────────────────────────────────────────────
        # 数学操作：x (B, C, E) 与 unembed.weight (V, E) 做矩阵乘法 → logits (B, C, V)
        #   · 本质：与 embed 层"id → 向量"互为逆操作（embed 是查表，unembed 是反查表）
        #   · 权重共享（__init__ 第 425 行 embed.weight = unembed.weight）：
        #     同一个 (V, E) 矩阵同时用于"编码 token 为向量"和"解码向量为词分数"，
        #     语义对齐（词 i 的嵌入方向 = 分类出词 i 的方向），且参数省一半。
        #
        # 输出 logits 三维形状 (B, C, V) 的详细含义：
        #   第 0 维 B = batch_size：第几条句子（共 16 条并行）
        #   第 1 维 C = context_len：句子中第几个 token 位置（共 256 个位置）
        #   第 2 维 V = vocab_size = 50257：对应 tiktoken 词表中的每一个 token
        #     ↳ 索引 v (0 ~ 50256) 与 token id 严格一一对应
        #       v=0    → "!"
        #       v=1    → "."
        #       v=1212 → "def"
        #       ...
        #       v=50256 → "<|endoftext|>"（结束符）
        #
        # 具体地：logits[b][c][v] = 浮点数分数（未归一化，不是概率）
        #   含义：第 b 条句子、第 c 个位置，在"看到了前 c+1 个 token (ids[b][0..c])"
        #         的上下文之后，给"下一个词是第 v 号 token"打多少分。
        #   分数越高 → 模型认为该词是下一个词的"可能性越大"。
        #
        # 为什么每个位置都要产出一整份 V 维打分？——对应训练的 256 个并行监督信号：
        #   logits[b][0]  (V 维) ← 用 ids[b][0] 预测 → 目标 = ids[b][1]
        #   logits[b][1]  (V 维) ← 用 ids[b][0..1] 预测 → 目标 = ids[b][2]
        #   logits[b][2]  (V 维) ← 用 ids[b][0..2] 预测 → 目标 = ids[b][3]
        #   ...
        #   logits[b][255] (V 维) ← 用 ids[b][0..255] 预测 → 目标 = ids[b][256]
        #   （因果掩码保证第 c 位绝对看不到 ids[b][c+1..] 的未来信息）
        #
        # 后续使用分两条路径（都沿 V 维度操作）：
        #   【训练】pretrain.py 第 114 行：沿 V 维算 cross_entropy(softmax(logits), 正确id) → Loss
        #     → logits.view(-1, V) 把 B*C 个位置拼成 (B*C, V)，当做 50257 类分类任务算交叉熵
        #     → Loss 越低 = 正确词对应的 v 维度上分数越高 / 其他词分数越低
        #   【推理/生成】test_generate.py：只取最后一个位置 logits[:, -1, :] → (B, V)
        #     → F.softmax(..., dim=-1) 沿 V 维归一化为概率 P(v)，总和=1
        #     → torch.multinomial() 或 argmax() 沿 V 维采样出下一个 token id
        logits = self.unembed(x)
        return logits

    def save(self, file_path):
        """
        将模型（超参数 + 训练好的权重）打包保存到指定路径。
        采用「state_dict + 超参数」分离式打包，是 PyTorch 官方推荐的持久化方式：
          - 好处①：解耦结构与权重，即使模型类代码改动（如加层/改类名）也能单独加载权重
          - 好处②：纯张量序列化，跨设备/跨代码版本兼容性强
        """
        checkpoint = {
            # ─── 权重参数（训练成果，约占 99.9% 文件体积）───
            # self.state_dict() 返回 OrderedDict，包含模型所有可学习张量：
            #   token_embed.weight          词嵌入矩阵        [V, E]
            #   blocks.i.attn.q_proj.weight 第 i 层 Q 投影矩阵 [E, E]
            #   blocks.i.ffn.fc1.weight     第 i 层 FFN 第一层 [ff_dim, E]
            #   ... 以及所有 bias、LayerNorm 参数等
            # 这些是梯度下降训练出来的真正「知识」，推理时缺一不可。
            'model_state_dict': self.state_dict(),

            # ─── 超参数（结构描述，几个整数，用于「重建骨架」）───
            # 这些参数决定了模型张量的形状，加载时必须先知道它们才能实例化空模型。
            # 若只保存 state_dict 而不保存这些，load 时无法知道每层该造多大。
            'vocab_size': self.vocab_size,           # 词表大小 V，决定 token_embed / unembed 的维度
            'max_context_len': self.max_context_len, # 最大上下文长度 C，决定位置嵌入矩阵行数
            'embed_dim': self.embed_dim,             # 残差流维度 E，决定绝大多数矩阵的列/行
            'n_head': self.n_head,                   # 注意力头数 H，决定每个头的维度 D=E/H
            'n_layer': self.n_layer,                 # Transformer 块的数量，决定网络深度
            'ff_dim': self.ff_dim,                   # FFN 中间层宽度，决定 ffc1/ffc2 的形状
            'dropout_rate': self.dropout_rate,       # dropout 概率（推理时不生效，但保持配置一致）
        }
        # 序列化：把整个 checkpoint dict 写入磁盘文件
        torch.save(checkpoint, file_path)

    @classmethod
    def load_from(cls, file_path, device='cpu'):
        """
        从 checkpoint 文件还原模型（两步走：先搭骨架 → 再灌权重）。
        @classmethod 的好处是不用先有模型实例即可调用：Transformer.load_from('xxx.pt')
        """
        # 第一步：反序列化读取整个 checkpoint dict
        # map_location=device 确保张量直接落在目标设备上（避免先加载到 GPU 再转到 CPU 的报错）
        checkpoint = torch.load(file_path, map_location=device)

        # ─── 第二步：用超参数「重建空骨架」（此时所有权重都是随机初始化的垃圾值）───
        # 为什么不直接反序列化出一个 model 对象？
        #   1. PyTorch 直接 pickle 整个 model 会绑定类路径和类结构，脆弱易崩
        #   2. 显式用 cls(...) 构造，保证当前代码的模型类定义和权重形状 100% 一致
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_context_len=checkpoint['max_context_len'],
            embed_dim=checkpoint['embed_dim'],
            n_head=checkpoint['n_head'],
            n_layer=checkpoint['n_layer'],
            ff_dim=checkpoint['ff_dim'],
            dropout_rate=checkpoint['dropout_rate']
        )

        # ─── 第三步：把训练好的真实权重「灌进空骨架」，覆盖随机值 ───
        # load_state_dict 会按 key 严格匹配赋值：
        #   checkpoint['model_state_dict']['blocks.0.attn.q_proj.weight']
        #   → 赋值给 model.blocks[0].attn.q_proj.weight
        # 若形状不匹配或有缺失/多余 key，会抛异常提示，避免静默错误。
        model.load_state_dict(checkpoint['model_state_dict'])

        # 第四步：把整个模型移动到目标设备（CPU / CUDA / MPS）
        # 虽然 state_dict 已经 map_location 到 device，但 model 对象本身的 .to() 是必须的
        # 它会递归调用所有子模块和参数的 .to(device)，保证后续 forward 在正确设备上跑。
        model.to(device)

        # 返回：结构正确 + 权重正确 + 设备正确 的可用模型实例
        return model