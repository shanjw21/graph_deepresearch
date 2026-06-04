# GPT-2 Transformer 解码器架构详解

---

## 一、Embedding 层：`wte` 和 `wpe`

```
GPT2Model(
    (wte): Embedding(50257, 768)    ← Word Token Embedding
    (wpe): Embedding(1024, 768)     ← Word Position Embedding
    (drop): Dropout(p=0.1, inplace=False)
    ...
)
```

| 名称 | 全称 | 形状 | 作用 |
|------|------|------|------|
| `wte` | Word Token Embedding | `(50257, 768)` | 把 token ID 转换成**语义向量** |
| `wpe` | Word Position Embedding | `(1024, 768)` | 把位置编号转换成**位置向量** |

---

### `wte` — 词嵌入（Token Embedding）

**作用**：把离散的 token ID 映射为连续的向量表示。

```
输入: input_ids = [13752, 4508, 287, 362, 257]
                  ↑
                 每个数字是词表中的 ID

        ┌─────────────────────────────────────────┐
        │         wte 查找表 (50257 × 768)        │
        │                                          │
        │  ID=0     → [ 0.02, -0.15, ...,  0.08] │
        │  ID=1     → [ 0.11,  0.03, ..., -0.22] │
        │  ...                                     │
        │  ID=13752 → [-0.05,  0.18, ...,  0.01] │ ← "Free"
        │  ID=4508  → [ 0.07, -0.12, ...,  0.15] │ ← "entry"
        │  ID=287   → [ 0.03,  0.09, ..., -0.06] │ ← "in"
        │  ID=362   → [-0.11,  0.04, ...,  0.12] │ ← "2"
        │  ID=257   → [ 0.14, -0.07, ...,  0.03] │ ← "a"
        │  ...                                     │
        │  ID=50256 → [ 0.01, -0.03, ...,  0.09] │ ← EOS/PAD
        └─────────────────────────────────────────┘

输出: token_embeddings ∈ ℝ^(seq_len × 768)
```

**关键点**：
- 词表大小 50257：GPT-2 能识别的所有 token 数量
- 向量维度 768：每个 token 用 768 个浮点数表示语义
- 这是一个**可学习的查找表**，训练过程中会不断更新

---

### `wpe` — 位置嵌入（Position Embedding）

**作用**：告诉模型每个 token 在序列中的位置。

**为什么需要位置信息？**

Transformer 的注意力机制是"无序"的——它同时看到所有 token，不知道谁在前谁在后。

```
"猫 吃 鱼"  和  "鱼 吃 猫"

如果没有位置信息，模型看到的都是 {猫, 吃, 鱼} 这三个 token
无法区分主语和宾语！

加上位置信息后:
位置 0 → "猫" 是第一个词 → 主语
位置 1 → "吃" 是第二个词 → 动词
位置 2 → "鱼" 是第三个词 → 宾语
```

```
位置编码查找表 (1024 × 768):

位置 0 → [ 0.01, -0.05, ...,  0.12]  ← 序列开头
位置 1 → [-0.03,  0.08, ..., -0.06]  ← 第二个位置
位置 2 → [ 0.06,  0.02, ...,  0.09]  ← 第三个位置
...
位置 1023 → [ 0.04, -0.11, ...,  0.07]  ← 最大位置

输出: position_embeddings ∈ ℝ^(seq_len × 768)
```

**关键点**：
- 最大长度 1024：GPT-2 最多处理 1024 个 token 的序列
- 位置编码是**可学习的**（不是固定的正弦函数）
- 不同位置的编码向量不同，模型通过学习来理解"顺序"

---

### `wte` + `wpe` 的组合

```python
# GPT-2 源码中的实际操作
hidden_states = wte(input_ids) + wpe(position_ids)
```

```
input_ids = [13752, 4508, 287, 362, 257]   (5个token)
position_ids = [0, 1, 2, 3, 4]             (5个位置)

wte 输出:                              wpe 输出:
┌─────────────────────────┐           ┌─────────────────────────┐
│ [-0.05, 0.18, ..., 0.01] │ 位置0    │ [ 0.01, -0.05, ..., 0.12] │
│ [ 0.07, -0.12, ..., 0.15] │ 位置1    │ [-0.03, 0.08, ..., -0.06] │
│ [ 0.03, 0.09, ..., -0.06] │ 位置2    │ [ 0.06, 0.02, ..., 0.09] │
│ [-0.11, 0.04, ..., 0.12] │ 位置3    │ [-0.02, 0.11, ..., 0.03] │
│ [ 0.14, -0.07, ..., 0.03] │ 位置4    │ [ 0.09, -0.01, ..., -0.08] │
└─────────────────────────┘           └─────────────────────────┘
                 ↓ 相加
        ┌─────────────────────────┐
        │ [-0.04, 0.13, ..., 0.13] │  ← "Free" + 位置0
        │ [ 0.04, -0.04, ..., 0.09] │  ← "entry" + 位置1
        │ [ 0.09, 0.11, ..., 0.03] │  ← "in" + 位置2
        │ [-0.13, 0.15, ..., 0.15] │  ← "2" + 位置3
        │ [ 0.23, -0.08, ..., -0.05] │  ← "a" + 位置4
        └─────────────────────────┘
                 ↓
        hidden_states ∈ ℝ^(5 × 768)
```

**相加后的向量同时包含两种信息**：
- **语义信息**：这个 token 是什么词（来自 wte）
- **位置信息**：这个词在序列的哪个位置（来自 wpe）

---

## 二、GPT-2 Transformer 解码器完整架构

```
输入: input_ids ∈ ℝ^(batch × seq_len)
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Embedding 层                                 │
│                                                                  │
│   hidden = wte(input_ids) + wpe(position_ids) + dropout         │
│                                                                  │
│   wte: Embedding(50257, 768)  ← 词嵌入                          │
│   wpe: Embedding(1024, 768)   ← 位置嵌入                        │
│   drop: Dropout(0.1)          ← 防止过拟合                      │
│                                                                  │
│   输出: hidden ∈ ℝ^(batch × seq_len × 768)                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Transformer Block × 12 层 (GPT-2 Small)            │
│              Transformer Block × 36 层 (GPT-2 Large)            │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              单个 Transformer Block 结构                    │  │
│  │                                                            │  │
│  │   输入: hidden ∈ ℝ^(batch × seq_len × 768)                │  │
│  │            │                                               │  │
│  │            ├──→ LayerNorm ──→ Masked Multi-Head Attention ─┤  │
│  │            │                       │                       │  │
│  │            │                       ▼                       │  │
│  │            │              attn_output ∈ ℝ^(batch×seq×768) │  │
│  │            │                       │                       │  │
│  │            └───────────────────────┼── residual connection │  │
│  │                                    │                       │  │
│  │                                    ▼                       │  │
│  │                           hidden + attn_output             │  │
│  │                                    │                       │  │
│  │            ┌───────────────────────┤                       │  │
│  │            │                       │                       │  │
│  │            ▼                       ▼                       │  │
│  │      LayerNorm ──→ Feed Forward Network (MLP) ──→         │  │
│  │                          │                                 │  │
│  │                          ▼                                 │  │
│  │                   ffn_output ∈ ℝ^(batch×seq×768)          │  │
│  │                          │                                 │  │
│  │                          └── residual connection ──→       │  │
│  │                                         │                  │  │
│  │                                         ▼                  │  │
│  │                              hidden + ffn_output           │  │
│  │                                         │                  │  │
│  │                                         ▼                  │  │
│  │                                  输出到下一层                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     输出层                                       │
│                                                                  │
│   logits = hidden × wte.weight.T  ← 权重共享！                  │
│                                                                  │
│   输出: logits ∈ ℝ^(batch × seq_len × 50257)                    │
│   含义: 每个位置对词表中每个 token 的预测分数                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、Transformer Block 内部详解

### ① Layer Normalization

```python
# 对每个 token 的 768 维向量做归一化
LayerNorm(768)

输入: x ∈ ℝ^(768)
输出: y = (x - mean) / sqrt(var + eps) * gamma + beta

作用: 稳定训练，加速收敛
```

### ② Masked Multi-Head Attention（核心）

```
输入: hidden ∈ ℝ^(batch × seq_len × 768)

Step 1: 计算 Q, K, V
        Q = hidden × W_Q    (768 → 768)
        K = hidden × W_K    (768 → 768)
        V = hidden × W_V    (768 → 768)

Step 2: 拆分成多个头 (12 个头)
        Q, K, V 各拆成 12 份，每份 64 维
        Q_i ∈ ℝ^(batch × seq_len × 64),  i=1..12

Step 3: 计算注意力分数
        score = Q_i × K_i^T / √64
        score ∈ ℝ^(batch × seq_len × seq_len)

Step 4: 应用 Mask（因果掩码）
        ┌─────────────────────────┐
        │ 1  0  0  0  0  0  0  0 │ ← 位置0只能看到自己
        │ 1  1  0  0  0  0  0  0 │ ← 位置1能看到0和1
        │ 1  1  1  0  0  0  0  0 │ ← 位置2能看到0,1,2
        │ 1  1  1  1  0  0  0  0 │ ← 位置3能看到0,1,2,3
        │ 1  1  1  1  1  0  0  0 │
        │ 1  1  1  1  1  1  0  0 │
        │ 1  1  1  1  1  1  1  0 │
        │ 1  1  1  1  1  1  1  1 │ ← 位置7能看到所有
        └─────────────────────────┘
        0 的位置被设为 -∞，softmax 后变成 0

        为什么用 Mask？
        → GPT 是自回归模型，生成时只能看到前面的词
        → 训练时也要保持这个约束，不能"偷看"未来

Step 5: Softmax + 加权求和
        attn_weights = softmax(score)
        attn_output = attn_weights × V

Step 6: 拼接所有头 + 线性投影
        output = Concat(head_1, ..., head_12) × W_O
        output ∈ ℝ^(batch × seq_len × 768)
```

### ③ Feed-Forward Network (MLP)

```
输入: x ∈ ℝ^(batch × seq_len × 768)

Step 1: 扩张 (768 → 3072)
        h = x × W_1 + b_1
        h ∈ ℝ^(batch × seq_len × 3072)

Step 2: 激活函数 (GELU)
        h = GELU(h)
        GELU(x) = x × Φ(x)  ← 高斯误差线性单元
        比 ReLU 更平滑，效果更好

Step 3: 压缩 (3072 → 768)
        output = h × W_2 + b_2
        output ∈ ℝ^(batch × seq_len × 768)

作用: 对每个 token 独立做非线性变换
      相当于给模型"思考"的能力
```

### ④ 残差连接 (Residual Connection)

```
输出 = 输入 + 子层输出

Attention:
    hidden = hidden + attn_output

MLP:
    hidden = hidden + ffn_output

作用:
1. 缓解梯度消失（梯度可以直接跳过子层传播）
2. 保留原始信息（子层学习"增量"而非"全新表示"）
3. 使得深层网络可训练
```

---

## 四、完整数据流示例

```
输入: "The cat sat on the mat"
      ↓ tokenizer
input_ids: [464, 3797, 3332, 319, 262, 2603]
      ↓ wte + wpe
hidden: [6 × 768]  ← 每个 token 变成 768 维向量，包含语义+位置信息
      ↓
┌─────────────────────────────────────────────┐
│ Layer 1:                                    │
│   LayerNorm → Masked Attention → + residual │
│   LayerNorm → MLP → + residual              │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ Layer 2:                                    │
│   LayerNorm → Masked Attention → + residual │
│   LayerNorm → MLP → + residual              │
└──────────────────┬──────────────────────────┘
                   ↓
                 ... (共 12 层或 36 层)
                   ↓
┌─────────────────────────────────────────────┐
│ Layer 12:                                   │
│   LayerNorm → Masked Attention → + residual │
│   LayerNorm → MLP → + residual              │
└──────────────────┬──────────────────────────┘
                   ↓
              LayerNorm (最终)
                   ↓
logits: [6 × 50257]  ← 每个位置对 50257 个 token 的预测分数
                   ↓
      argmax / softmax → 预测下一个 token
```

---

## 五、关键设计总结

| 组件 | 作用 | 形状 |
|------|------|------|
| `wte` | 词嵌入：token ID → 语义向量 | `(50257, 768)` |
| `wpe` | 位置嵌入：位置编号 → 位置向量 | `(1024, 768)` |
| `LayerNorm` | 归一化，稳定训练 | `(768,)` |
| `Masked Attention` | 关注上下文，但不能看未来 | 多头 × 64 维 |
| `MLP` | 非线性变换，增加表达能力 | `768 → 3072 → 768` |
| `Residual` | 缓解梯度消失，保留原始信息 | 直接相加 |
| `Dropout` | 正则化，防止过拟合 | `p=0.1` |

---

## 六、`c_attn`：QKV 合并投影层详解

### 为什么 Q、K、V 可以合并？

在标准 Transformer 中，Q、K、V 是**三个独立的线性层**：

```
标准 Transformer:
    Q = x × W_Q    (768 → 768)
    K = x × W_K    (768 → 768)
    V = x × W_V    (768 → 768)

    三次独立计算，三个独立权重矩阵
```

但 GPT-2 把它们**合并成一次计算**：

```
GPT-2:
    [Q, K, V] = x × W_c_attn    (768 → 2304)
                              ↑
                        768 × 3 = 2304

    一次计算，一个权重矩阵，然后拆分
```

**数学上完全等价，但计算效率更高**（一次矩阵乘法 vs 三次）。

---

### `c_attn` 的结构

```
c_attn: Linear(768, 2304)

权重矩阵 W_c_attn ∈ ℝ^(768 × 2304)
偏置 b_c_attn ∈ ℝ^(2304)

内部逻辑上分成三块:
┌─────────────────────────────────────────────────────┐
│                  W_c_attn (768 × 2304)              │
│                                                      │
│  ┌───────────────┬───────────────┬───────────────┐  │
│  │   W_Q 块      │   W_K 块      │   W_V 块      │  │
│  │  (768 × 768)  │  (768 × 768)  │  (768 × 768)  │  │
│  │               │               │               │  │
│  │  用于生成 Q   │  用于生成 K   │  用于生成 V   │  │
│  └───────────────┴───────────────┴───────────────┘  │
│         ↓                ↓                ↓          │
│      列 0~767        列 768~1535     列 1536~2303    │
└─────────────────────────────────────────────────────┘
```

---

### 前向传播过程

```
输入: hidden ∈ ℝ^(batch × seq_len × 768)

Step 1: 线性投影（一次矩阵乘法）
        qkv = hidden × W_c_attn + b_c_attn
        qkv ∈ ℝ^(batch × seq_len × 2304)

Step 2: 拆分成 Q, K, V
        Q, K, V = chunk(qkv, 3, dim=-1)

        Q ∈ ℝ^(batch × seq_len × 768)
        K ∈ ℝ^(batch × seq_len × 768)
        V ∈ ℝ^(batch × seq_len × 768)

Step 3: 拆分成多头（12 个头）
        Q = reshape(Q, batch, seq_len, 12, 64)
        K = reshape(K, batch, seq_len, 12, 64)
        V = reshape(V, batch, seq_len, 12, 64)

        每个头 64 维: 768 / 12 = 64

Step 4: 计算注意力
        score = Q × K^T / √64
        attn = softmax(mask(score)) × V

Step 5: 拼接所有头
        output = reshape(attn, batch, seq_len, 768)
```

---

### 具体数值示例

```
输入: "The cat sat"
      ↓ tokenizer
input_ids: [464, 3797, 3332]    (3 个 token)

经过 Embedding 后:
hidden ∈ ℝ^(1 × 3 × 768)
        ↓
┌─────────────────────────────────────────────────────────┐
│                    c_attn 线性层                         │
│                                                          │
│  hidden × W_c_attn = qkv                                │
│                                                          │
│  (1 × 3 × 768) × (768 × 2304) = (1 × 3 × 2304)        │
│                                                          │
│  每个 token 的 768 维向量 → 2304 维向量                   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
                  qkv ∈ ℝ^(1 × 3 × 2304)
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
     Q ∈ ℝ^(1×3×768)  K ∈ ℝ^(1×3×768)  V ∈ ℝ^(1×3×768)
            │              │              │
            ▼              ▼              ▼
     拆成 12 个头     拆成 12 个头     拆成 12 个头
     每头 (1×3×64)   每头 (1×3×64)   每头 (1×3×64)
            │              │              │
            └──────────────┼──────────────┘
                           │
                           ▼
                   Multi-Head Attention
```

---

### PyTorch 代码实现

```python
import torch
import torch.nn as nn

class GPT2Attention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads  # 64

        # QKV 合并投影层
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)  # 768 → 2304

        # 输出投影层
        self.c_proj = nn.Linear(hidden_size, hidden_size)  # 768 → 768

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Step 1: QKV 合并投影
        qkv = self.c_attn(x)  # (batch, seq, 2304)

        # Step 2: 拆分成 Q, K, V
        Q, K, V = qkv.chunk(3, dim=-1)  # 各 (batch, seq, 768)

        # Step 3: 拆分成多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_size)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_size)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_size)

        # 转置为 (batch, num_heads, seq, head_size)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Step 4: 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5)
        # scores: (batch, num_heads, seq, seq)

        # 应用因果掩码（下三角矩阵）
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (batch, num_heads, seq, head_size)

        # Step 5: 拼接所有头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        # attn_output: (batch, seq, 768)

        # Step 6: 输出投影
        output = self.c_proj(attn_output)
        # output: (batch, seq, 768)

        return output
```

---

### 为什么 GPT-2 用合并方式？

| 对比 | 分离投影（标准） | 合并投影（GPT-2） |
|------|----------------|------------------|
| **矩阵乘法次数** | 3 次 | 1 次 |
| **权重矩阵数量** | 3 个 | 1 个 |
| **计算效率** | 较慢 | 较快 |
| **数学等价性** | - | 完全等价 |
| **实现简洁性** | 代码多 | 代码少 |

```
标准方式:
    Q = x × W_Q    ← 第 1 次矩阵乘法
    K = x × W_K    ← 第 2 次矩阵乘法
    V = x × W_V    ← 第 3 次矩阵乘法

GPT-2 方式:
    qkv = x × W_c_attn  ← 只有 1 次矩阵乘法
    Q, K, V = split(qkv)

GPU 上，一次大矩阵乘法比三次小矩阵乘法更快（更好的并行性）
```

---

### Q、K、V 的语义解释

```
Q = Query（查询）：我在找什么？
    → 当前 token 想要关注哪些信息

K = Key（键）：我能提供什么？
    → 每个 token 能提供的"索引"信息

V = Value（值）：我的实际内容是什么？
    → 每个 token 的实际语义内容

注意力计算:
    score = Q × K^T    → 当前 token 与每个 token 的"相关性"
    output = softmax(score) × V    → 根据相关性加权提取信息

例子: "The cat sat on the mat"
    当处理 "sat" 时:
    Q("sat") × K("cat") = 高分  → "sat" 和 "cat" 高度相关
    Q("sat") × K("the") = 低分  → "sat" 和 "the" 相关性低

    所以 "sat" 会更多地关注 "cat" 的 Value 信息
```

---

### 总结

```
c_attn 的本质:
    一个 Linear(768, 2304) 层
    = 把 Q、K、V 三个投影合并成一次计算
    = 输出后拆分成三份，分别作为 Q、K、V
    = 数学上等价于三个独立的 Linear(768, 768)
    = 但计算效率更高（1 次大矩阵乘法 vs 3 次小矩阵乘法）

"QKV 合并投影层" 这个名字的含义:
    Q = Query（查询）：我在找什么？
    K = Key（键）：每个 token 能提供什么？
    V = Value（值）：每个 token 的实际内容是什么？
    c_attn 把这三者一次性算出来，所以叫 "QKV 合并投影"
```
