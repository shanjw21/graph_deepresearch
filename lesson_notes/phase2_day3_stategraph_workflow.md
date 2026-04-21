# 阶段二 · Day 3：StateGraph 单线工作流 + 节点函数

## 今日目标

1. 理解 LangGraph StateGraph 的三个核心操作：add_node / add_edge / compile
2. 实现第一个能跑的 LangGraph 工作流：`START → create_analysts → human_feedback → END`
3. 理解节点函数的"读状态 → 调 LLM → 返回更新"模式
4. 完成阶段一遗留的状态读写图

---

## 知识点一：StateGraph 是什么？

StateGraph 就是一张**有向图**：
- **节点（node）** = 函数，接收状态，返回状态更新
- **边（edge）** = 数据流向，定义节点之间的执行顺序
- **START / END** = LangGraph 内置的特殊节点，标记入口和出口

```python
from langgraph.graph import START, END, StateGraph

builder = StateGraph(MyState)       # 1. 创建图
builder.add_node("节点名", 函数)     # 2. 添加节点
builder.add_edge(START, "节点名")    # 3. 连接边
builder.add_edge("节点名", END)
graph = builder.compile()            # 4. 编译
result = graph.invoke({"key": "值"}) # 5. 执行
```

**面试话术：** "StateGraph 把工作流建模为有向图，节点是纯函数，边是数据流。编译后通过 invoke 触发执行。"

---

## 知识点二：节点函数的三步模式

每个节点函数都遵循同一个模式：

```python
def 节点名(state: 状态类型):
    # 第一步：从 state 中读取需要的字段
    topic = state['topic']

    # 第二步：执行业务逻辑（通常是调用 LLM）
    result = llm.invoke(...)

    # 第三步：返回 dict，只包含需要更新的字段
    return {"字段名": result}
```

**关键规则：**
- 返回的 dict 只需要包含**变化的字段**，不需要返回整个状态
- LangGraph 自动把返回值 merge 到全局状态
- 普通字段直接覆盖，累加字段（operator.add）自动追加

---

## 知识点三：项目中的节点函数解析

### create_analysts（源码第 563-598 行）

```python
def create_analysts(state: GenerateAnalystsState):
    # 读
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')

    # 调 LLM（用结构化输出）
    structured_llm = llm.with_structured_output(Perspectives)
    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts
    )
    analysts = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="生成分析师集合。")
    ])

    # 返回更新
    return {"analysts": analysts.analysts}
```

**读写分析：**
```
读：topic, max_analysts, human_analyst_feedback
写：analysts
```

### human_feedback（源码第 601-612 行）

```python
def human_feedback(state: GenerateAnalystsState):
    pass  # 空函数，只作为 interrupt_before 的停靠点
```

**读写分析：** 什么都不读，什么都不写。存在的唯一目的是让 `interrupt_before` 有地方暂停。

### should_continue（源码第 615-633 行）——条件路由

```python
def should_continue(state: GenerateAnalystsState):
    if state.get('human_analyst_feedback'):
        return "create_analysts"  # 有反馈 → 回去重新生成
    return END                      # 无反馈 → 结束
```

**注意：** 条件路由函数不是节点，它返回的是**下一个节点的名字**，不是状态更新。

---

## 知识点四：add_edge vs add_conditional_edges

```python
# 固定边：A 执行完一定走 B
builder.add_edge("A", "B")

# 条件边：A 执行完，根据条件走 B 或 C
builder.add_conditional_edges(
    "A",                    # 从哪个节点出发
    should_continue,        # 路由函数，返回节点名字符串
    ["create_analysts", END] # 可能的目标节点列表
)
```

---

## 动手任务

### 任务：搭建 "生成分析师" 的线性工作流

**目标：** 实现 `START → create_analysts → human_feedback → END`，能跑通并输出分析师列表。

**骨架代码：**

```python
"""
阶段二 · Day 3 任务：生成分析师工作流

你的任务：
1. 补全 create_analysts 节点函数（标记 TODO 的部分）
2. 补全图的构建（标记 TODO 的部分）
3. 运行并验证输出

参考源码：research_assistant.py 第 563-598 行
"""

import os
from typing import List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

# ============================================
# Step 1: 数据模型（阶段一 Day 1 学过的）
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


# ============================================
# Step 2: 状态模型（阶段一 Day 2 学过的）
# ============================================

class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]


# ============================================
# Step 3: 提示词模板
# ============================================

analyst_instructions = """你需要创建一组 AI 分析师人设。请严格遵循以下指引：

1. 先审阅研究主题：
{topic}

2. 查看（可选的）编辑反馈，它将指导分析师的人设创建：

{human_analyst_feedback}

3. 基于上述文档与/或反馈，识别最值得关注的主题。

4. 选出前 {max_analysts} 个主题。

5. 为每个主题分配一位分析师。"""


# ============================================
# Step 4: 初始化 LLM
# ============================================

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ============================================
# Step 5: 节点函数 —— 你来实现
# ============================================

def create_analysts(state: GenerateAnalystsState):
    """
    TODO: 实现这个函数

    提示：
    1. 从 state 中读取 topic, max_analysts, human_analyst_feedback
    2. 用 llm.with_structured_output(Perspectives) 创建结构化 LLM
    3. 用 analyst_instructions.format(...) 格式化提示词
    4. 调用 structured_llm.invoke([SystemMessage(...), HumanMessage(...)])
    5. 返回 {"analysts": 结果.analysts}
    """
    # TODO: 在这里写你的代码
    pass


def human_feedback(state: GenerateAnalystsState):
    """空节点，作为人类审核的停靠点"""
    pass


# ============================================
# Step 6: 构建图 —— 你来实现
# ============================================

# TODO: 创建 StateGraph，使用 GenerateAnalystsState
# builder = StateGraph(...)

# TODO: 添加节点 create_analysts 和 human_feedback
# builder.add_node(...)

# TODO: 添加边
# START → create_analysts → human_feedback → END

# TODO: 编译图
# graph = builder.compile()


# ============================================
# Step 7: 运行测试
# ============================================

if __name__ == "__main__":
    result = graph.invoke({
        "topic": "人工智能在医疗领域的应用",
        "max_analysts": 3,
        "human_analyst_feedback": "",
        "analysts": []
    })

    print("生成的分析师：")
    for i, analyst in enumerate(result["analysts"], 1):
        print(f"\n--- 分析师 {i} ---")
        print(analyst.persona)
```

**验收标准：**
1. 运行后输出 3 个分析师，每个有 name / role / affiliation / description
2. 能用 `result["analysts"][0].name` 直接访问属性

**完成后思考：**
- create_analysts 函数读了哪些字段？写了哪个字段？
- 如果 `human_analyst_feedback` 不为空，会发生什么？（提示：目前不会怎样，因为还没加条件路由，阶段三再加）

---

## 今日面试题

**Q1: "StateGraph 中节点函数的返回值格式是什么？"**
> 返回一个 dict，key 是要更新的状态字段名，value 是新值。只需要包含变化的字段，不需要返回整个状态。LangGraph 自动 merge 到全局状态。

**Q2: "human_feedback 节点为什么是空的？"**
> 它存在的唯一目的是作为 interrupt_before 的停靠点。配合 checkpointer，工作流执行到这里会暂停，人类可以审核当前状态并注入反馈，然后从暂停点继续执行。

**Q3: "add_edge 和 add_conditional_edges 的区别？"**
> add_edge 是静态连接，A 完了必定走 B。add_conditional_edges 是动态路由，A 完了根据路由函数的返回值决定走哪条路。路由函数返回的是下一个节点的名字符串。

---

## 阶段一遗留：状态读写图

现在你有了节点函数的代码，可以完成阶段一 Day 2 的实验 3 了：

### ResearchGraphState 读写图

```
字段                    写入节点                  读取节点
────────────────────────────────────────────────────────────────
topic                   用户输入(初始化)           create_analysts, initiate_all_interviews,
                                                  write_report, write_introduction, write_conclusion

max_analysts            用户输入(初始化)           create_analysts

human_analyst_feedback  人类注入(update_state)     create_analysts, initiate_all_interviews

analysts                create_analysts            initiate_all_interviews

sections                conduct_interview          write_report, write_introduction, write_conclusion
                        (operator.add 累加)

introduction            write_introduction         finalize_report

content                 write_report               finalize_report

conclusion              write_conclusion           finalize_report

final_report            finalize_report            用户获取结果
```

**你的任务：** 画出 InterviewState 的字段读写图。提示：参考源码第 640-856 行的节点函数。

---

## 参考源码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| create_analysts 节点 | `research_assistant.py` | 563-598 |
| human_feedback 节点 | `research_assistant.py` | 601-612 |
| should_continue 路由 | `research_assistant.py` | 615-633 |
| analyst_instructions 提示词 | `research_assistant.py` | 329-342 |
| 主图构建 | `research_assistant.py` | 1125-1173 |
| 图编译 | `research_assistant.py` | 1180-1184 |
