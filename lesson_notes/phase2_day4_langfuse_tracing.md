# 阶段二 · Day 4：Langfuse 追踪——给工作流装上"仪表盘"

## 今日目标

1. 注册 Langfuse 账号，获取 API Key
2. 理解 Trace / Span / Score 三个核心概念
3. 给 Day 3 的分析师生成工作流接上 Langfuse 追踪
4. 在 Langfuse 控制台看到完整的执行记录

---

## Step 1：注册 Langfuse（10 分钟）

Langfuse 是一个开源的 LLM 可观测性平台，有免费 Cloud 版本。

### 注册步骤

1. 打开 https://cloud.langfuse.com
2. 点击 **Sign up**，用 GitHub 或 Google 账号登录
3. 登录后，点击左侧 **Settings** → **API Keys**
4. 你会看到两组 Key：
   - `PK-LF-...` — Public Key（公钥）
   - `SK-LF-...` — Secret Key（私钥）
5. 记下这两个值和你的 Host 地址（Cloud 版是 `https://cloud.langfuse.com`）

### 配置 .env 文件

在你的 `.env` 文件中添加：

```
# 之前已有的
BASE_URL=你的OpenAI代理地址
API_KEY=你的OpenAI密钥
MODEL=你的模型名

# 新增 Langfuse 配置
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 验证连接

```python
from langfuse import get_client

langfuse = get_client()
if langfuse.auth_check():
    print("连接成功！")
else:
    print("连接失败，检查 Key 和 Host")
```

---

## 知识点一：Trace / Span / Score 三层结构

```
Trace（追踪）—— 一次完整的用户请求
  │
  ├── Span（跨度）—— 一个具体的步骤
  │     ├── 节点执行
  │     ├── LLM 调用
  │     └── 搜索请求
  │
  └── Score（评分）—— 对这次请求的质量评价
        ├── 用户反馈（点赞/点踩）
        └── 自动评分（LLM-as-a-Judge）
```

**类比：**
- **Trace** = 一次外卖订单（从下单到送达）
- **Span** = 订单中的每个环节（接单 → 做菜 → 配送）
- **Score** = 用户评价（五星好评）

**面试话术：** "Trace 是一次完整的请求链路，Span 是链路中的每个步骤，Score 是对结果的评分。Langfuse 用这三层结构实现全链路可观测性。"

---

## 知识点二：两种接入方式

### 方式一：CallbackHandler（自动追踪 LangChain 调用）

```python
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

# 传给 LangGraph 的 config，自动追踪每个 LLM 调用
result = graph.invoke(input, config={"callbacks": [langfuse_handler]})
```

**效果：** 自动记录每次 LLM 调用的输入、输出、token 消耗、延迟。你不需要写任何额外代码。

### 方式二：@observe 装饰器（手动追踪自定义函数）

```python
from langfuse import observe

@observe()
def my_function(input_text):
    # 这个函数的调用会被记录为一个 Trace
    return "result"
```

**效果：** 被装饰的函数自动变成一个 Trace，函数内的每次 LLM 调用自动变成 Span。

### 项目中的用法：两种结合

```python
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

langfuse = get_client()

@observe()
def run_research(topic):
    handler = CallbackHandler()
    result = graph.invoke(
        {"topic": topic, ...},
        config={"callbacks": [handler]}  # 自动追踪 LLM 调用
    )
    return result
```

---

## 动手任务

### 任务：给 Day 3 的工作流加上 Langfuse 追踪

**骨架代码：**

```python
"""
阶段二 · Day 4 任务：给分析师工作流加上 Langfuse 追踪

你的任务：
1. 配置 .env 中的 Langfuse Key
2. 验证 Langfuse 连接
3. 补全 Langfuse 追踪代码（标记 TODO 的部分）
4. 运行后到 Langfuse 控制台查看追踪记录
"""

import os
from typing import List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

# ============================================
# Step 1: 加载环境变量
# ============================================
load_dotenv()

base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model = os.getenv("MODEL")

# ============================================
# Step 2: 验证 Langfuse 连接
# ============================================
# TODO: 导入 get_client，创建 langfuse 客户端，验证连接
# 提示：
#   from langfuse import get_client
#   langfuse = get_client()
#   print("连接成功" if langfuse.auth_check() else "连接失败")



# ============================================
# Step 3: 数据模型 + 状态模型（和 Day 3 一样）
# ============================================

class Analyst(BaseModel):
    affiliation: str = Field(description="分析师的主要隶属机构或组织")
    name: str = Field(description="分析师姓名")
    role: str = Field(description="分析师在研究主题中的具体角色定位")
    description: str = Field(description="分析师的关注焦点、关切点和动机的详细描述")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="包含所有分析师角色和隶属机构的综合列表")


class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]


analyst_instructions = """你需要创建一组 AI 分析师人设。请严格遵循以下指引：

1. 先审阅研究主题：
{topic}

2. 查看（可选的）编辑反馈，它将指导分析师的人设创建：

{human_analyst_feedback}

3. 基于上述文档与/或反馈，识别最值得关注的主题。

4. 选出前 {max_analysts} 个主题。

5. 为每个主题分配一位分析师。"""

llm = ChatOpenAI(model=model, temperature=0, base_url=base_url, api_key=api_key)


def create_analysts(state: GenerateAnalystsState):
    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_feedback = state["human_analyst_feedback"]
    structured_llm = llm.with_structured_output(Perspectives, method="function_calling")
    system_message = analyst_instructions.format(
        topic=topic,
        max_analysts=max_analysts,
        human_analyst_feedback=human_feedback
    )
    result = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="生成分析师集合。")
    ])
    return {"analysts": result.analysts}


def human_feedback_node(state: GenerateAnalystsState):
    pass


builder = StateGraph(GenerateAnalystsState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback_node)
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_edge("human_feedback", END)
graph = builder.compile()


# ============================================
# Step 4: 运行 + Langfuse 追踪
# ============================================

if __name__ == "__main__":
    topic = "人工智能在医疗领域的应用"

    # TODO: 实现 Langfuse 追踪
    # 提示：
    #   1. 导入 CallbackHandler: from langfuse.langchain import CallbackHandler
    #   2. 创建 handler = CallbackHandler()
    #   3. 把 handler 传入 config={"callbacks": [handler]}
    #   4. 调用 graph.invoke(input, config)
    #   5. 调用 langfuse.flush() 确保数据发送
    #
    # 预期：graph.invoke 接受两个参数
    #   graph.invoke(输入dict, config={"callbacks": [handler]})
    #
    # 注意：LangGraph 的 invoke 签名是
    #   graph.invoke(input, config=None)
    #   config 参数是第二个参数

    result = # TODO: 你来实现

    print("生成的分析师：")
    for i, analyst in enumerate(result["analysts"], 1):
        print(f"\n--- 分析师 {i} ---")
        print(analyst.persona)

    # TODO: 刷新 langfuse 缓冲区，确保数据发送到服务器
    # 提示：langfuse.flush()

    print("\n请到 Langfuse 控制台查看追踪记录！")
```

**验收标准：**
1. 运行后，终端正常输出 3 个分析师
2. 打开 Langfuse 控制台（https://cloud.langfuse.com），左侧点击 **Traces**
3. 能看到一条新的追踪记录，点进去能看到：
   - LLM 调用的输入（提示词）和输出（分析师列表）
   - Token 消耗数量
   - 执行耗时

---

## 今日面试题

**Q1: "Langfuse 的 Trace / Span / Score 分别是什么？"**
> Trace 是一次完整的请求链路（比如一次完整的分析师生成流程）。Span 是链路中的每个步骤（比如一次 LLM 调用、一次搜索）。Score 是对结果的评分（比如用户反馈或自动评估）。

**Q2: "CallbackHandler 的原理是什么？"**
> 它利用 LangChain 的回调机制。LLM 每次调用前后，LangChain 会触发 on_llm_start / on_llm_end 回调，CallbackHandler 在回调中自动记录输入、输出、token 和延迟到 Langfuse，不需要修改业务代码。

**Q3: "为什么要用 langfuse.flush()？"**
> Langfuse SDK 默认是异步批量发送数据的（先存到内存缓冲区）。flush() 强制把缓冲区的数据立即发送到服务器。在脚本末尾调用确保数据不丢失。

---

## 参考源码位置

| 内容 | 文件 | 位置 |
|------|------|------|
| Langfuse 连接验证 | `03_trace_and_evaluation_langgraph_agents.ipynb` | cell-11 |
| CallbackHandler 用法 | `03_trace_and_evaluation_langgraph_agents.ipynb` | cell-19 |
| @observe 装饰器用法 | `03_trace_and_evaluation_langgraph_agents.ipynb` | cell-19 |
| score_trace 评分 | `03_trace_and_evaluation_langgraph_agents.ipynb` | cell-26 |
