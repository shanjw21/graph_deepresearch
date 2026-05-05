# 阶段六 · Day 11：离线评估 + LLM-as-a-Judge

## 今日目标

1. 理解离线评估与在线评估的区别，建立完整评估思维
2. 用 Langfuse 数据集管理测试用例，建立评估基准
3. 用 v4 Experiment SDK 跑批量实验，对比不同配置
4. 实现 LLM-as-a-Judge 自动评分函数

---

## 知识点一：离线评估 vs 在线评估

```
在线评估（Day 10）：生产环境实时监控 — "现在有什么问题？"
离线评估（Day 11）：开发阶段系统性测试 — "改了之后有没有变好？"
```

**离线评估的核心流程：**

```
准备数据集 → 定义任务函数 → 跑实验 → 自动评分 → 分析结果 → 迭代优化
```

**典型应用场景：**
- 上线前回归测试：确保新版 Agent 不退化
- A/B 对比：对比不同模型 / 提示词 / 参数的效果
- 质量门禁：低于阈值的版本不允许上线

---

## 知识点二：Langfuse 数据集管理

数据集是离线评估的基石。每个数据项包含 **input**（输入）和 **expected_output**（期望输出）。

```python
from langfuse import get_client

langfuse = get_client()

# 创建数据集
langfuse.create_dataset(
    name="email-agent-test",
    description="邮件处理 Agent 测试数据集"
)

# 添加测试项
langfuse.create_dataset_item(
    dataset_name="email-agent-test",
    input={"email": {"sender": "京东客服", "subject": "发票", "body": "..."}},
    expected_output={"is_spam": False}
)

# 获取数据集
dataset = langfuse.get_dataset("email-agent-test")
print(f"数据集包含 {len(dataset.items)} 个测试项")
```

**数据项结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `input` | dict | Agent 的输入（邮件数据） |
| `expected_output` | dict | 期望的输出（is_spam 判定等） |
| `id` | str | 数据项唯一 ID |

---

## 知识点三：v4 Experiment SDK — 批量跑实验

v4 用 `dataset.run_experiment()` 替代了 v3 的 `item.run()` 循环，更简洁。

```python
dataset = langfuse.get_dataset("email-agent-test")

# 定义任务函数：接收 item，返回结果
def my_task(*, item, **kwargs):
    """每个数据项执行这个函数"""
    email_data = item.input["email"]
    result = process_email(email_data)
    return result

# 跑实验
dataset.run_experiment(
    name="run-gpt4o",
    task=my_task,
    metadata={"model": "gpt-4o", "temperature": 0}
)
```

**run_experiment 做了什么？**
- 遍历数据集的每个 item
- 对每个 item 调用 task 函数
- 自动创建 Trace 记录 input/output
- 关联 dataset item 用于后续对比
- 去 Langfuse 控制台 → Datasets 页面查看实验结果

**对比不同配置：**

```python
# 实验 1：原始模型
dataset.run_experiment(name="exp-baseline", task=task_baseline)

# 实验 2：优化后的提示词
dataset.run_experiment(name="exp-improved-prompt", task=task_improved)
```

---

## 知识点四：LLM-as-a-Judge — 自动评分

用另一个 LLM 来评估 Agent 的输出质量，替代人工标注。

**核心模式：**

```
Agent 输出 + 期望输出 → 评分提示词 → Judge LLM → 评分结果
```

**代码实现：**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

judge_llm = ChatOpenAI(model=model_name, temperature=0, base_url=base_url, api_key=api_key)

judge_prompt = """
你是一名评估专家。请评估以下 Agent 输出的质量。

输入：{input}
期望输出：{expected}
实际输出：{actual}

评估维度：正确性（0或1）
只输出分数，不要其他内容。
"""

def judge_output(input_data, expected, actual):
    """用 LLM 判断实际输出是否正确"""
    prompt = judge_prompt.format(
        input=input_data,
        expected=expected,
        actual=actual
    )
    response = judge_llm.invoke([HumanMessage(content=prompt)])
    score = float(response.content.strip())
    return score
```

**常见评估维度：**

| 维度 | 评估内容 | 分数类型 |
|------|---------|---------|
| 正确性 | 输出是否符合期望 | 0/1 |
| 完整性 | 输出是否包含关键信息 | 0-1 连续 |
| 有用性 | 输出对用户是否有帮助 | 0/1 |
| 毒性 | 输出是否包含有害内容 | 0/1 |

**面试要点：** "我在离线评估中实现了 LLM-as-a-Judge 自动评分函数，对邮件分类正确性和回复质量两个维度打分，在 Langfuse 控制台对比不同实验的得分分布。"

---

## 知识点五：完整离线评估工作流

```
1. 准备阶段：构造测试数据集 → 上传到 Langfuse
2. 实验阶段：定义 task 函数 → run_experiment 跑实验
3. 评分阶段：LLM-as-a-Judge 自动评分 + 规则评分
4. 分析阶段：Langfuse 控制台对比实验结果
```

**与 Day 10 在线评估的关系：**

| | 在线评估（Day 10） | 离线评估（Day 11） |
|---|---|---|
| 时机 | 生产环境实时 | 开发阶段批量 |
| 数据来源 | 真实用户请求 | 预构造测试集 |
| 评分方式 | 用户反馈 / 即时评分 | LLM-as-a-Judge / 规则评分 |
| 目的 | 监控当前状态 | 验证改进效果 |

---

## 动手任务

### 任务：邮件 Agent 离线评估 + LLM-as-a-Judge

**步骤：**

1. 复用 Day 10 的邮件 Agent
2. 构造测试数据集（至少 6 条：3 条合法 + 3 条垃圾）
3. 上传数据集到 Langfuse
4. 定义 task 函数，用 `run_experiment` 跑实验
5. 实现 LLM-as-a-Judge 评分函数
6. 去 Langfuse 控制台查看实验结果

**骨架代码见 `phase6_day11_skeleton.py`**

**验收标准：**
1. Langfuse 控制台能看到数据集和测试项
2. 实验跑完所有数据项，每项都有 Trace
3. LLM-as-a-Judge 评分函数能正确返回 0 或 1
4. 控制台能看到每项的评分

---

## 今日面试题

**Q1: "离线评估和在线评估的区别是什么？各自用什么方法？"**
> 离线评估是在开发阶段用预构造的数据集系统性测试，方法包括数据集评测、LLM-as-a-Judge 自动评分、规则评分。在线评估是生产环境实时监控，方法包括成本追踪、延迟追踪、用户反馈。离线评估告诉你"改了之后有没有变好"，在线评估告诉你"现在有什么问题"。

**Q2: "什么是 LLM-as-a-Judge？它的优缺点是什么？"**
> 用另一个 LLM 来评估 Agent 的输出质量。优点是可扩展、成本低、一致性好；缺点是 Judge LLM 本身可能出错（评判偏差），需要定期用人工标注校准。实际中通常结合规则评分使用：规则能判断的用规则（如格式检查），规则无法判断的用 LLM（如语义正确性）。

**Q3: "如何用 Langfuse 数据集做 A/B 对比实验？"**
> 第一步创建包含 input + expected_output 的数据集；第二步用 run_experiment 跑 baseline 实验；第三步修改配置（模型、提示词等）再跑一次实验；第四步在 Langfuse 控制台对比两次实验的评分分布，判断改进是否有效。

**Q4: "离线评估的评分函数应该怎么设计？"**
> 分两层：第一层是规则评分，检查格式、长度、关键字等确定性特征；第二层是 LLM-as-a-Judge，评估语义正确性、有用性等模糊特征。两层结合比单一方法更可靠。评分提示词要明确评分标准和输出格式，减少 Judge LLM 的随意性。

---

## 参考源码位置

| 内容 | 文件 |
|------|------|
| 数据集创建与管理 | `03_trace_and_evaluation_langgraph_agents.ipynb` Cell 32-40 |
| 实验运行与对比 | 同上 Cell 41-48 |
| LLM-as-a-Judge 配置 | 同上 Cell 49-50 |
| 在线评估（Day 10） | 同上 Cell 22-31 |
