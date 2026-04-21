# 阶段一 · Day 2：TypedDict 状态模型 + operator.add 累加机制

## 今日目标

1. 理解 TypedDict 为什么是 LangGraph 状态管理的首选
2. 掌握 `operator.add` 累加机制——Map-Reduce 的核心基础
3. 理解 MessagesState 的继承关系
4. 亲手重建 3 个状态模型（GenerateAnalystsState / InterviewState / ResearchGraphState）

---

## 知识点一：为什么用 TypedDict 而不是 Pydantic？

昨天学的 Analyst / Perspectives 用 **Pydantic**（BaseModel），今天学的状态模型用 **TypedDict**。为什么不一样？

```
Pydantic (BaseModel)          TypedDict
─────────────────             ──────────
用途：LLM 的输入/输出 schema    用途：工作流节点之间传递的状态
特点：严格的类型校验             特点：轻量，只是类型提示
验证：会报错如果类型不对          验证：运行时不校验，只是声明

Analyst ← Pydantic             ResearchGraphState ← TypedDict
   ↑ LLM 需要严格的 schema         ↑ 节点之间传递，灵活性更重要
   ↑ Field(description=...)         ↑ 用 Annotated[list, operator.add] 做累加
     告诉 LLM 填什么                   告诉 LangGraph 怎么合并
```

**一句话总结：** Pydantic 管 LLM 输入输出，TypedDict 管工作流内部状态。

---

## 知识点二：operator.add——状态累加的秘密武器

这是今天最核心的知识点，也是后续 Map-Reduce 的基础。

### 普通字段 vs 累加字段

```python
from typing import Annotated, List
import operator
from typing_extensions import TypedDict

class ResearchGraphState(TypedDict):
    topic: str                                    # 普通字段：直接覆盖
    sections: Annotated[list, operator.add]        # 累加字段：用 + 合并
```

**区别在哪？** 当多个节点返回同一个字段的值时：

```
普通字段 topic:
  节点A返回 {"topic": "AI医疗"}
  节点B返回 {"topic": "AI教育"}
  最终: topic = "AI教育"         ← 后者覆盖前者

累加字段 sections:
  节点A返回 {"sections": ["第一节内容"]}
  节点B返回 {"sections": ["第二节内容"]}
  节点C返回 {"sections": ["第三节内容"]}
  最终: sections = ["第一节内容", "第二节内容", "第三节内容"]   ← 用 + 合并
```

### 用代码验证

```python
import operator

# 模拟 operator.add 的行为
old_sections = ["第一节"]
new_sections = ["第二节"]
result = old_sections + new_sections  # operator.add 就是调用 +
print(result)  # ["第一节", "第二节"]
```

### 为什么这是 Map-Reduce 的基础？

在 Deep Research 项目中，3 个分析师**并行**做访谈，每个分析师产出一个报告小节：

```
分析师1 → write_section → {"sections": ["AI诊断的进展..."]}
分析师2 → write_section → {"sections": ["AI伦理的挑战..."]}
分析师3 → write_section → {"sections": ["AI成本的分析..."]}
                                      ↓
                    sections 字段自动累加为:
                    ["AI诊断的进展...", "AI伦理的挑战...", "AI成本的分析..."]
                                      ↓
                         write_report 节点拿到完整列表，生成最终报告
```

**如果没有 operator.add，** 你需要手动写代码合并 3 个并行节点的输出。有了它，LangGraph 自动帮你做。

**面试考点：**
- Q: "operator.add 在状态中的作用？"
- A: "当多个并行节点返回同名 list 字段时，LangGraph 自动用 + 合并而不是覆盖。这是 Map-Reduce 模式实现并行结果聚合的核心机制。"

---

## 知识点三：MessagesState——对话历史的自动追加

```python
from langgraph.graph import MessagesState

class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list
```

**MessagesState 是 LangGraph 提供的特殊状态类，它的 messages 字段默认使用 operator.add。**

等价于你自己写：

```python
class InterviewState(TypedDict):
    messages: Annotated[list, operator.add]   # MessagesState 自带这个
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list
```

**为什么 messages 要累加？**

访谈过程中，每个节点都会往对话历史里追加消息：

```
generate_question → {"messages": [AIMessage("AI在医疗影像方面...")]}
                                                        ↓ messages 自动追加
generate_answer   → {"messages": [AIMessage("根据文献...")]}
                                                        ↓ messages 自动追加
generate_question → {"messages": [AIMessage("那中国三甲医院...")]}
                                                        ↓ messages 自动追加
generate_answer   → {"messages": [AIMessage("北京协和医院...")]}
```

所有节点都返回**新消息**，LangGraph 自动把它们追加到 messages 列表。节点函数永远能通过 `state["messages"]` 看到完整的对话历史。

---

## 知识点四：三个状态模型的完整解析

### 模型 1：GenerateAnalystsState（最简单）

```python
class GenerateAnalystsState(TypedDict):
    topic: str                   # 输入：研究主题
    max_analysts: int            # 输入：分析师数量上限
    human_analyst_feedback: str  # 人机协同：人类反馈
    analysts: List[Analyst]      # 输出：生成的分析师列表
```

**数据流：**
```
用户输入 topic + max_analysts
    ↓
create_analysts 节点读取 topic，生成 analysts
    ↓
human_feedback 节点，人类可修改 human_analyst_feedback
    ↓
如果有反馈 → 回到 create_analysts，重新生成 analysts
```

**注意：** 这个状态里没有 `operator.add`，因为 analysts 每次都是整体重新生成，不需要累加。

### 模型 2：InterviewState（中等复杂）

```python
class InterviewState(MessagesState):          # 继承 → 自带 messages: Annotated[list, operator.add]
    max_num_turns: int                         # 控制参数：访谈最多几轮
    context: Annotated[list, operator.add]     # 累加：每轮搜索的文档都追加
    analyst: Analyst                           # 输入：当前分析师
    interview: str                             # 输出：完整访谈记录文本
    sections: list                             # 输出：报告小节
```

**两个累加字段的作用：**

```
messages（继承自 MessagesState，自动累加）
  └─ 对话历史：每轮问答的消息自动追加

context（手动声明 Annotated[list, operator.add]）
  └─ 搜索结果：每轮搜索的文档自动追加
  └─ search_web 返回 {"context": ["文档A"]}
     search_baike 返回 {"context": ["文档B"]}
     → context 变成 ["文档A", "文档B"]
  └─ 下一轮搜索又追加 → context 变成 ["文档A", "文档B", "文档C", "文档D"]
  └─ write_section 时可以引用所有轮次搜集的全部素材
```

### 模型 3：ResearchGraphState（最复杂）

```python
class ResearchGraphState(TypedDict):
    topic: str                                   # 输入
    max_analysts: int                            # 输入
    human_analyst_feedback: str                  # 人机协同
    analysts: List[Analyst]                      # 分析师列表
    sections: Annotated[list, operator.add]      # ★ 累加：所有访谈的报告小节
    introduction: str                            # 报告引言
    content: str                                 # 报告主体
    conclusion: str                              # 报告结论
    final_report: str                            # 最终报告
```

**sections 的完整生命周期：**

```
阶段1: initiate_all_interviews
  为每个 analyst 创建 Send("conduct_interview", {...})
  → 3 个访谈子图并行启动

阶段2: 每个访谈子图内部
  ... 访谈循环 ...
  → write_section 返回 {"sections": ["小节1内容"]}
  → write_section 返回 {"sections": ["小节2内容"]}
  → write_section 返回 {"sections": ["小节3内容"]}

阶段3: sections 自动累加为 ["小节1内容", "小节2内容", "小节3内容"]

阶段4: write_report / write_introduction / write_conclusion
  都读取 state["sections"]，拿到完整的小节列表
```

---

## 动手实验

### 实验 1：验证 operator.add 的行为（10 分钟）

```python
import operator
from typing import Annotated, List
from typing_extensions import TypedDict

# 模拟 LangGraph 的状态更新逻辑
def update_state(current_state: dict, node_output: dict, state_schema: type):
    """简化版的状态更新：模拟 LangGraph 内部行为"""
    result = dict(current_state)
    for key, new_value in node_output.items():
        # 检查这个字段是否有 operator.add 注解
        if key == "sections":
            result[key] = result.get(key, []) + new_value  # 累加
        else:
            result[key] = new_value  # 覆盖
    return result

# 模拟并行节点返回
state = {"topic": "AI医疗", "sections": []}

# 节点A（分析师1的访谈结果）
state = update_state(state, {"sections": ["AI诊断进展"]}, None)
print(f"节点A后: {state}")

# 节点B（分析师2的访谈结果）
state = update_state(state, {"sections": ["AI伦理挑战"]}, None)
print(f"节点B后: {state}")

# 节点C（分析师3的访谈结果）
state = update_state(state, {"sections": ["AI成本分析"]}, None)
print(f"节点C后: {state}")

print(f"\n最终 sections: {state['sections']}")
# 预期: ['AI诊断进展', 'AI伦理挑战', 'AI成本分析']
```

---

### 实验 2：完整状态模型重建（20 分钟）

**你的任务：** 不看源码，根据下面的描述，自己写 3 个状态模型。

**GenerateAnalystsState：**
- `topic: str` — 研究主题
- `max_analysts: int` — 分析师数量上限
- `human_analyst_feedback: str` — 人类反馈
- `analysts: List[Analyst]` — 分析师列表
- 全部是普通字段，没有累加

**InterviewState：**
- 继承 `MessagesState`（从 `langgraph.graph` 导入）
- `max_num_turns: int` — 访谈轮次上限
- `context: Annotated[list, operator.add]` — 搜索结果，累加
- `analyst: Analyst` — 当前分析师
- `interview: str` — 访谈记录
- `sections: list` — 报告小节
- 有两个累加字段（messages + context）

**ResearchGraphState：**
- `topic: str` — 研究主题
- `max_analysts: int` — 分析师数量
- `human_analyst_feedback: str` — 人类反馈
- `analysts: List[Analyst]` — 分析师列表
- `sections: Annotated[list, operator.add]` — 报告小节，累加
- `introduction: str` — 引言
- `content: str` — 主体
- `conclusion: str` — 结论
- `final_report: str` — 最终报告
- 只有一个累加字段

**写完后和源码对比：**
- `research_assistant.py` 第 258-268 行（GenerateAnalystsState）
- `research_assistant.py` 第 281-294 行（InterviewState）
- `research_assistant.py` 第 297-316 行（ResearchGraphState）

---

### 实验 3：画状态读写图（15 分钟）

**任务：** 画出每个字段在哪些节点被"读"和"写"。

示例：

```
ResearchGraphState 字段读写图:

字段              写入节点              读取节点
─────────────────────────────────────────────────────────
topic             用户输入(初始化)      create_analysts, initiate_all_interviews,
                                       write_report, write_introduction, write_conclusion

max_analysts      用户输入(初始化)      create_analysts

analysts          create_analysts       initiate_all_interviews, conduct_interview

sections          conduct_interview     write_report, write_introduction, write_conclusion
                  (通过 operator.add)   finalize_report (间接，通过 content)

introduction      write_introduction    finalize_report

content           write_report          finalize_report

conclusion        write_conclusion      finalize_report

final_report      finalize_report       用户获取结果
```

**你的任务：** 对 InterviewState 做同样的分析。哪些字段在哪个节点被写入？哪些被读取？

---

## 今日面试题

### 基础题

**Q1: "TypedDict 和 Pydantic BaseModel 在项目中分别用在哪里？为什么？"**
> Pydantic 用于 LLM 的输入输出（Analyst/Perspectives/SearchQuery），因为它需要严格的 schema 约束 LLM 的输出格式。TypedDict 用于工作流内部的状态传递（ResearchGraphState/InterviewState），因为它更轻量，而且支持 Annotated 累加机制，Pydantic 做不到。

**Q2: "operator.add 在状态中起什么作用？"**
> 当多个并行节点返回同一个 list 字段时，LangGraph 自动用 `+` 合并而不是覆盖。比如 3 个分析师并行访谈，每个返回 `{"sections": ["小节内容"]}`，最终 sections 自动合并为包含 3 个小节的列表。这是 Map-Reduce 模式的基础。

**Q3: "MessagesState 和普通 TypedDict 有什么区别？"**
> MessagesState 是 LangGraph 提供的内置状态类，它的 messages 字段默认用 operator.add 声明，意味着每次节点返回新消息时自动追加而不是覆盖。省去了手动声明 `messages: Annotated[list, operator.add]`。

### 进阶题

**Q4: "ResearchGraphState 的 sections 和 InterviewState 的 sections 是同一个东西吗？"**
> 不是。InterviewState 的 sections 是单个访谈产出的一个小节（列表长度为1），返回到主图后通过 operator.add 累加到 ResearchGraphState 的 sections 中。子图的输出映射到主图的累加字段，这就是 Map-Reduce 的 Reduce 步骤的数据来源。

**Q5: "为什么 GenerateAnalystsState 里的 analysts 不用 operator.add？"**
> 因为分析师生成不是并行操作——每次都是整体重新生成（由人类反馈触发），后一次完全覆盖前一次，不需要累加。而访谈是并行的（多个分析师同时进行），所以 sections 需要累加。

---

## 今日总结

```
你这两天学到的阶段一全部知识：

Pydantic 模型（管 LLM 输入输出）
  ├── Analyst       4字段 + @property persona
  ├── Perspectives  容器类，包装 List[Analyst]
  └── SearchQuery   对话→搜索词

TypedDict 状态模型（管工作流内部状态）
  ├── GenerateAnalystsState  全普通字段，无累加
  ├── InterviewState         继承 MessagesState，2个累加字段
  └── ResearchGraphState     1个累加字段 sections

核心机制：
  operator.add → 并行节点返回的 list 自动合并 → Map-Reduce 的基础

明天进入阶段二：StateGraph 单线工作流
  把这些模型"串"起来，让数据真正"流"起来
```

---

## 参考源码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| GenerateAnalystsState | `research_assistant.py` | 258-268 |
| SearchQuery | `research_assistant.py` | 271-278 |
| InterviewState | `research_assistant.py` | 281-294 |
| ResearchGraphState | `research_assistant.py` | 297-316 |
| operator.add 实际使用 | `research_assistant.py` | 291, 312 |
| MessagesState 导入 | `research_assistant.py` | 208 |
