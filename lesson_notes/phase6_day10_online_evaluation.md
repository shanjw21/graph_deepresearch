# 阶段六 · Day 10：智能体在线评估 — Langfuse 追踪与评分

## 今日目标

1. 构建一个邮件处理 Agent（LangGraph），理解可观测性的基础
2. 接入 Langfuse 追踪，看清 Agent 每一步的执行过程
3. 掌握三种评分方案：即时评分 / 深层评分 / 异步评分
4. 理解在线评估四大维度：成本、延迟、用户反馈、自动评分

---

## 知识点一：为什么需要智能体评估？

```
传统软件：输入 → 确定性函数 → 输出（可预测、可测试）
智能体系统：输入 → LLM 推理 + 工具调用 + 多步决策 → 输出（不确定、难测试）
```

**智能体评估的两个层次：**

| 层次 | 名称 | 时机 | 方法 |
|------|------|------|------|
| 在线评估 | Online Eval | 生产环境实时监控 | 成本、延迟、用户反馈、自动评分 |
| 离线评估 | Offline Eval | 开发阶段系统性检查 | 数据集评测、A/B 测试、LLM-as-a-Judge |

**今天学在线评估，明天学离线评估 + LLM-as-a-Judge。**

---

## 知识点二：构建邮件处理 Agent

我们用一个简单的邮件 Agent 来学习评估。它有 4 个节点：

```
START → read_email → classify_email →  SPAM → handle_spam → END
                                   →  HAM → drafting_response → notify_mr_wayne → END
```

**核心结构：**

```python
class EmailState(TypedDict):
    email: Dict[str, Any]       # 原始邮件
    is_spam: Optional[bool]     # 垃圾邮件判定
    draft_response: Optional[str]  # 回复草稿
    messages: List[str]           # 对话历史
```

**关键设计点：**
- `classify_email`：用 LLM 判断 SPAM/HAM，返回条件路由
- `route_email`：根据 `is_spam` 决定走哪个分支
- `drafting_response`：用 LLM 生成礼貌回复
- `notify_mr_wayne`：格式化通知消息

这个 Agent 足够简单来学习评估概念，又足够复杂来展示追踪的价值。

---

## 知识点三：Langfuse 追踪接入

在 Day 4 我们已经学过基础的 Langfuse CallbackHandler。今天深入理解追踪结构：

```
Trace（追踪）— 一次完整的 Agent 调用
├── Span（跨度）— 一个逻辑步骤（如 classify_email）
│   ├── Generation（生成）— 一次 LLM 调用
│   └── Generation — 另一次 LLM 调用
└── Span — 另一个步骤
    └── Generation ...
```

**两种接入方式对比：**

| 方式 | 代码 | 适用场景 |
|------|------|---------|
| CallbackHandler | `config={"callbacks": [handler]}` | LangChain/LangGraph 自动追踪 |
| @observe 装饰器 | `@observe()` | 自定义函数追踪 |

**推荐组合：** 用 `@observe()` 包裹入口函数 + CallbackHandler 追踪 LangGraph 内部。

```python
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

@observe()
def process_email(email_data):
    handler = CallbackHandler()
    result = compiled_graph.invoke(
        input={"email": email_data, ...},
        config={"callbacks": [handler]}
    )
    return result
```

**@observe() 做了什么？**
- 自动创建 Trace
- 自动记录函数的 input 和 output（无需手动设置）
- 捕获异常和执行时间
- 与 CallbackHandler 的 observation 自动关联

---

## 知识点四：三种评分方案

### 方案一：span.score_trace() — 即时评分（推荐）

```python
with langfuse.start_as_current_observation(as_type="span", name="workflow") as span:
    result = process_email(email_data)

     # 调用你的评分函数
    score = evaluate_response_quality(result["draft_response"])

    # 通过 span 对象直接对 trace 评分
    span.score_trace(
        name="user-feedback",
        value=score,
        data_type="NUMERIC"
    )
```

真实场景中的评分流程：
1.计算评分（根据业务逻辑计算评分）
```python
# 方式A：用 LLM 评估输出质量
def evaluate_response_quality(result):
    """用 LLM 判断回复质量，返回 0-1 分数"""
    prompt = f"评估以下回复的质量，返回0-1的分数：{result}"
    response = llm.invoke(prompt)
    return float(response.content)

# 方式B：用规则评估
def evaluate_response_quality(result):
    """基于规则的评分"""
    score = 0.0
    if len(result) > 50: score += 0.3
    if "感谢" in result: score += 0.3
    # ...
    return score

# 方式C：用户反馈
def get_user_feedback():
    """等待用户点击满意/不满意"""
    # 异步等待用户反馈...
    return user_rating

```
2.发送评分到langfuse

```python
with langfuse.start_as_current_observation(as_type="span") as span:
    result = process_email(email)

    # 调用你的评分函数
    score = evaluate_response_quality(result["draft_response"])

    # 发送到 Langfuse
    span.score_trace(name="quality", value=score, data_type="NUMERIC")

```

**适用：** 评分逻辑就在执行代码旁边，最直观。

### 方案二：score_current_trace() — 上下文评分

```python
with langfuse.start_as_current_observation(as_type="span", name="workflow"):
    result = process_email(email_data)
    # 不需要 span 对象，自动找当前 trace
    langfuse.score_current_trace(
        name="quality-score",
        value=0.9,
        data_type="NUMERIC"
    )
```

**适用：** 不想用 span 变量，直接通过 langfuse 客户端在当前上下文中评分。

### 方案三：create_score(trace_id=...) — 异步评分

```python
# 执行时保存 trace_id
with langfuse.start_as_current_observation(as_type="span", name="workflow") as obs:
    result = process_email(email_data)
    saved_trace_id = obs.trace_id

# ... 稍后或别处，完全脱离执行上下文 ...
langfuse.create_score(
    trace_id=saved_trace_id,
    name="user-feedback",
    value=1,
    data_type="NUMERIC",
    comment="用户点击了有帮助"
)
```

**适用：** 用户延迟反馈、批量评分、跨服务评分。

**三种方案对比：**

| | 方案一 | 方案二 | 方案三 |
|---|---|---|---|
| 上下文依赖 | 需要 with | 需要 with | 不需要 |
| trace_id | 自动 | 自动 | 手动提供 |
| 时间耦合 | 同步 | 同步 | 完全解耦 |
| 典型场景 | 即时评分 | 上下文评分 | 异步/批量评分 |

---

## 知识点五：在线评估四大维度

### 1. 成本（Costs）

Langfuse 自动记录每次 LLM 调用的 token 用量：

```
Token Usage:
- classify_email: ~150 tokens (input) + ~50 tokens (output)
- drafting_response: ~200 tokens (input) + ~100 tokens (output)
- 总计: ~500 tokens
```

**面试要点：** 能说出"通过 Langfuse 的 token 追踪，我发现 classify_email 占总成本的 40%，所以优化了这个节点的提示词长度。"

### 2. 延迟（Latency）

Langfuse 自动记录每个节点的执行时间：

```
Latency:
- read_email: <1ms
- classify_email: 1.2s (LLM 调用)
- drafting_response: 1.5s (LLM 调用)
- 总计: ~2.7s
```

**面试要点：** "我发现 drafting_response 是延迟瓶颈，占整个流程的 55%。"

### 3. 用户反馈（User Feedback）

通过评分方案收集用户的 👍/👎：

```python
# 前端用户点击"有帮助"时调用
langfuse.create_score(
    trace_id=saved_trace_id,
    name="user-feedback",
    value=1,  # 1=有帮助, 0=无帮助
    data_type="NUMERIC"
)
```

### 4. 自动评分（LLM-as-a-Judge，明天深入）

用另一个 LLM 自动评估输出质量：
- 正确性（correctness）
- 有用性（helpfulness）
- 毒性（toxicity）
- 幻觉（hallucination）

---

## 动手任务

### 任务：构建邮件 Agent + Langfuse 追踪 + 三种评分

**步骤：**

1. 定义 EmailState 数据模型
2. 实现 5 个节点函数（read_email, classify_email, handle_spam, drafting_response, notify_mr_wayne）
3. 构建 StateGraph 并编译
4. 用 @observe + CallbackHandler 接入 Langfuse 追踪
5. 分别用三种方案为合法邮件和垃圾邮件的执行结果评分
6. 去 Langfuse 控制台查看追踪和评分结果

**骨架代码见 `phase6_day10_skeleton.py`**

**验收标准：**
1. 合法邮件走 HAM 分支，生成回复草稿
2. 垃圾邮件走 SPAM 分支，直接标记
3. Langfuse 控制台能看到完整的 Trace 结构（每个节点一个 Span）
4. 三种评分方案都能在控制台看到分数

---

## 今日面试题

**Q1: "在线评估和离线评估的区别是什么？"**
> 在线评估是在生产环境实时监控：成本、延迟、用户反馈。离线评估是在开发阶段系统性测试：用数据集跑批量评测、A/B 对比。在线评估告诉你"现在有什么问题"，离线评估告诉你"改了之后有没有变好"。

**Q2: "Langfuse 的三种评分方案分别适合什么场景？"**
> span.score_trace() 适合执行时即时评分，通过 span 对象直接操作；score_current_trace() 适合在当前上下文中不想用 span 变量的场景；create_score(trace_id=...) 适合用户延迟反馈或批量异步评分，因为它完全解耦了评分和执行的时间。

**Q3: "为什么 @observe() 和 CallbackHandler 要一起用？"**
> @observe() 包裹入口函数，自动创建 Trace 并记录函数级的 input/output（v4 无需手动设置）；CallbackHandler 传入 LangGraph 的 config，自动追踪图内每个节点的 LLM 调用。两者配合才能看到"入口 → 节点 → LLM 调用"的完整追踪链路。

**Q4: "如何用在线评估发现 Agent 的瓶颈？"**
> 通过 Langfuse 的延迟追踪，看哪个节点耗时最长（通常是 LLM 调用节点）；通过成本追踪，看哪个节点的 token 消耗最大（通常是提示词最长的节点）；通过用户反馈追踪，看哪类请求的用户满意度最低。三者结合定位瓶颈。

---

## 参考源码位置

| 内容 | 文件 |
|------|------|
| 邮件 Agent 构建 | `03_trace_and_evaluation_langgraph_agents.ipynb` Cell 13-17 |
| Langfuse 追踪接入 | 同上 Cell 19 |
| 三种评分方案 | 同上 Cell 26-28 |
| 在线评估概念 | 同上 Cell 22-31 |
| 离线评估 + 数据集 | 同上 Cell 32-48 |
| LLM-as-a-Judge | 同上 Cell 49-50 |
