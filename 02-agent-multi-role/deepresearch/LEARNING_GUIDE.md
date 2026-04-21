# Deep Research Assistant — 面试项目学习指南

> **核心理念：你不是在"读"这个项目，而是在"重建"它。**
> 每个阶段结束后，你应该能用白板画出该阶段的架构图，并解释清楚每一个设计决策。

---

## 项目一句话总结

**基于 LangGraph 的多智能体 Map-Reduce 系统：多个 AI 分析师并行"访谈"专家，最后汇总成一份结构化研究报告。**

---

## 架构全景图

```
用户输入 (topic, max_analysts)
         │
         ▼
  ┌──────────────┐
  │ create_       │  ← LLM 动态生成 3~5 个不同视角的分析师
  │ analysts      │    （医疗数据分析师 / AI 伦理顾问 / ...）
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ human_       │  ← interrupt_before 暂停，等待人类确认
  │ feedback     │    可调整分析师人选，或直接放行
  └──────┬───────┘
         │
    ┌────┴────┬────────┐              ← Map 阶段
    ▼         ▼        ▼             （Send 并行）
 ┌─────┐  ┌─────┐  ┌─────┐
  访谈1    访谈2    访谈N           每个访谈是一个子图
 │ask_q │  │      │
 │→search│  │      │
 │→ans  │  │      │              循环 max_num_turns 轮
 │→route│  │      │
 │→save │  │      │
 │→write│  │      │
 │section│ │      │
 └──┬───┘  └──┬───┘  └──┬───┘
    │         │         │
    ▼         ▼         ▼        ← Reduce 阶段
  sections 列表自动聚合
         │
    ┌────┼────────┐              三路并行生成
    ▼    ▼        ▼
  write  write    write
  report intro  conclusion
    │    │        │
    └────┼────────┘
         ▼
  finalize_report               组装最终 Markdown 报告
         │
         ▼
    final_report
```

---

## 阶段一：基础砖块（建议 2~3 天）

> **目标：** 理解项目的所有数据模型和 LLM 调用方式，能用 Pydantic + LangChain 写出最小的结构化输出 Demo。

### 1.1 你需要掌握的概念

| 概念 | 项目中对应 | 面试怎么说 |
|------|-----------|-----------|
| Pydantic 模型 | `Analyst`, `Perspectives`, `SearchQuery` | "用 Pydantic 定义 LLM 的输出 schema，确保类型安全" |
| 结构化输出 | `llm.with_structured_output(Perspectives)` | "通过 function calling 约束 LLM 返回格式，避免解析失败" |
| TypedDict 状态 | `ResearchGraphState`, `InterviewState` | "用 TypedDict 定义图的节点间传递状态，比 dataclass 更轻量" |
| operator.add 累加 | `Annotated[list, operator.add]` | "并行节点返回的结果自动累加，这是 LangGraph Map-Reduce 的核心机制" |
| MessagesState | `InterviewState(MessagesState)` | "LangGraph 内置的消息列表状态，自动追加消息" |

### 1.2 动手任务：重建数据模型

**任务：** 不看源码，仅根据下面的描述，自己写出 `Analyst`、`Perspectives`、`SearchQuery`、`ResearchGraphState`、`InterviewState` 这 5 个类。

**提示词：**
- `Analyst` 有 4 个字段：affiliation, name, role, description，还有一个 `@property persona` 把它们拼成人设字符串
- `ResearchGraphState` 有 9 个字段，其中 `sections` 用 `operator.add` 累加
- `InterviewState` 继承 `MessagesState`，额外有 `max_num_turns`, `context`(累加), `analyst`, `interview`, `sections`

### 1.3 动手任务：最小结构化输出 Demo

```python
# 目标：用 20 行代码实现"给一个主题，返回 3 个分析师"
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI

class Analyst(BaseModel):
    name: str = Field(description="分析师姓名")
    role: str = Field(description="角色定位")

class Perspectives(BaseModel):
    analysts: List[Analyst]

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(Perspectives)

result = structured_llm.invoke("为'AI在医疗中的应用'生成3个分析师")
print(result.analysts)
```

**你的任务：** 运行这段代码，然后修改它——让 `Analyst` 增加 `affiliation` 和 `description` 字段，观察 LLM 的输出变化。

### 1.4 面试高频问题

1. **"为什么用 Pydantic 而不是直接让 LLM 返回 JSON？"**
   → 结构化输出通过 function calling 实现，Pydantic 自动校验类型，比手动解析 JSON 字符串更可靠
2. **"operator.add 在状态中起什么作用？"**
   → 当多个并行节点返回同名 key 的 list 时，LangGraph 自动用 `+` 合并，这是 Map-Reduce 聚合的基础
3. **"temperature=0 的意义？"**
   → 保证报告生成的确定性和可复现性，同一输入得到相同输出

---

## 阶段二：单线工作流（建议 2 天）

> **目标：** 理解 LangGraph 的 StateGraph 机制，能用节点 + 边搭出一条线性工作流。

### 2.1 你需要掌握的概念

| 概念 | 项目中对应 | 面试怎么说 |
|------|-----------|-----------|
| StateGraph | `builder = StateGraph(ResearchGraphState)` | "用有向图建模工作流，节点是函数，边是数据流" |
| add_node | `builder.add_node("create_analysts", create_analysts)` | "每个节点是一个纯函数：接收状态，返回状态更新" |
| add_edge | `builder.add_edge(START, "create_analysts")` | "定义数据流向，START/END 是 LangGraph 的内置端点" |
| 节点函数签名 | `def create_analysts(state) -> dict` | "节点函数接收完整状态，返回 partial dict 来更新状态" |
| compile + invoke | `graph = builder.compile(); graph.invoke(input)` | "编译为可执行图，invoke 是同步调用" |

### 2.2 动手任务：搭出"生成分析师"的线性流

**目标：** 实现 `START → create_analysts → human_feedback → END` 这条简化版流程。

**关键代码骨架：**

```python
from langgraph.graph import START, END, StateGraph

builder = StateGraph(GenerateAnalystsState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)  # pass 空函数
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_edge("human_feedback", END)

graph = builder.compile()
result = graph.invoke({"topic": "AI医疗", "max_analysts": 3, "human_analyst_feedback": ""})
```

**你的任务：**
1. 运行上面的代码，确认能生成分析师
2. 尝试给 `human_analyst_feedback` 传入一条反馈（如"请增加一个关注法律合规的分析师"），观察结果变化

### 2.3 面试高频问题

1. **"节点函数的返回值是什么格式？"**
   → 返回 `dict`，key 对应状态字段名，value 是要更新的值。LangGraph 自动 merge 到全局状态
2. **"为什么不直接调用 LLM，而要用 StateGraph 包一层？"**
   → StateGraph 提供了条件路由、并行执行、状态持久化、人机协同等能力，裸调 LLM 做不到

---

## 阶段三：条件路由与人机协同（建议 1~2 天）

> **目标：** 理解条件路由（conditional_edges）和 interrupt_before 机制，实现"人类审核 → 反馈 → 重新生成"的循环。

### 3.1 核心机制解析

```
                    ┌─── 有反馈 ──→ 回到 create_analysts
                    │
create_analysts → human_feedback ─┤
                    │
                    └─── 无反馈 ──→ 继续后续流程
```

**关键代码（源码第 615-633 行）：**
```python
def should_continue(state):
    if state.get('human_analyst_feedback'):
        return "create_analysts"   # 有反馈，重新生成
    return END                      # 无反馈，结束

builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])
```

**interrupt_before 机制（源码第 1181-1184 行）：**
```python
graph = builder.compile(
    interrupt_before=['human_feedback'],  # 执行到此节点前暂停
    checkpointer=memory                   # 状态持久化，支持恢复
)
```

执行流程：
1. `graph.invoke(input)` → 执行到 `human_feedback` 前暂停
2. 人工检查 `state["analysts"]`，决定是否修改
3. `graph.update_state(config, {"human_analyst_feedback": "..."})` → 注入反馈
4. `graph.invoke(None, config)` → 从中断点继续执行

### 3.2 动手任务

**任务：** 在阶段二的代码基础上，加入条件路由：
- `human_feedback` 后，如果有反馈内容，回到 `create_analysts` 重新生成
- 如果没有反馈，进入下一步

### 3.3 面试高频问题

1. **"interrupt_before 和直接在代码里写 input() 有什么区别？"**
   → interrupt_before 配合 checkpointer 实现了状态持久化，进程可以安全退出后再恢复，适合 Web 服务场景；input() 只能用于本地脚本
2. **"checkpointer 的作用？"**
   → 在每个节点执行后自动保存状态快照，支持暂停/恢复、时间旅行调试、错误恢复

---

## 阶段四：访谈子图（建议 2~3 天）—— **项目核心**

> **目标：** 理解并重建访谈子图，这是整个项目最复杂的部分，包含循环对话、并行搜索、条件路由。

### 4.1 子图结构

```
ask_question (分析师提问)
     │
     ├──→ search_web   ──┐
     │                    ├──→ answer_question (专家回答)
     └──→ search_baike  ──┘
                              │
                         route_messages
                          │         │
                   继续提问     保存访谈
                     │              │
                     ↓              ↓
               ask_question    write_section → END
               (回到顶部)
```

### 4.2 关键设计决策（面试必问）

**Q: 为什么 search_web 和 search_baike 要并行？**
→ 一个问题同时查两个数据源，减少等待时间。LangGraph 的 add_edge 自动处理并行：当 `answer_question` 有两个入边时，它等两个搜索都完成后才执行。

**Q: route_messages 的两种退出条件是什么？**
1. 轮次达到 `max_num_turns`（默认 2）
2. 分析师说了"非常感谢您的帮助!"

**Q: 为什么 context 用 `operator.add` 累加？**
→ 每一轮搜索的结果都追加到 context 列表，后续写 section 时可以引用所有轮次搜集的素材。

### 4.3 动手任务：重建访谈子图

**Step 1 — 实现节点函数：**
- `generate_question`: 用 analyst.persona 格式化提示词，调用 LLM 生成问题
- `search_web`: 结构化输出 SearchQuery → Tavily 搜索 → XML 格式化
- `search_baike`: 结构化输出 SearchQuery → Wikipedia 加载 → XML 格式化
- `generate_answer`: 用 context + messages 调用 LLM，返回专家回答
- `save_interview`: 把 messages 转成字符串
- `write_section`: 用 context + analyst.description 调用 LLM 生成小节

**Step 2 — 组装图：**
```python
interview_builder = StateGraph(InterviewState)
# 添加 6 个节点...
# 定义边：START → ask_question → [search_web, search_baike] → answer_question
# 条件路由：answer_question → route_messages → [ask_question, save_interview]
# save_interview → write_section → END
```

**Step 3 — 测试：**
```python
interview_graph = interview_builder.compile(checkpointer=MemorySaver())
result = interview_graph.invoke({
    "analyst": Analyst(name="Dr.Chen", role="医疗AI专家", affiliation="协和医院", description="关注AI辅助诊断"),
    "messages": [HumanMessage(content="所以你说你在写一篇关于AI医疗的文章?")],
    "max_num_turns": 2,
    "context": [],
    "interview": "",
    "sections": []
})
print(result["sections"][0])  # 应该看到一篇报告小节
```

### 4.4 面试高频问题

1. **"并行搜索的结果怎么合并？"**
   → 两个搜索节点都返回 `{"context": [doc]}`，因为 context 是 `Annotated[list, operator.add]`，LangGraph 自动用 `+` 合并
2. **"如果搜索失败怎么办？"**
   → 当前代码没有显式错误处理，面试时可以说"可以加 try-except 返回空字符串，或在节点函数中加重试逻辑"
3. **"专家回答为什么只基于 context？"**
   → 防止幻觉（hallucination），确保所有论述都有据可查，来源可追溯

---

## 阶段五：Map-Reduce 并行调度（建议 2 天）—— **架构亮点**

> **目标：** 理解 Send 机制如何实现动态并行，掌握 Map-Reduce 模式在多智能体中的应用。

### 5.1 核心机制：Send API

```python
# 源码第 859-891 行
def initiate_all_interviews(state):
    return [
        Send("conduct_interview", {
            "analyst": analyst,
            "messages": [HumanMessage(content=f"所以你说你在写一篇关于{topic}的文章?")]
        })
        for analyst in state["analysts"]
    ]
```

**Send 的作用：**
- 为每个 analyst 动态创建一个 `conduct_interview` 节点实例
- 所有实例并行执行
- 每个实例独立运行访谈子图
- 返回的 `sections` 通过 `operator.add` 自动聚合

### 5.2 Reduce 阶段：三路并行报告生成

```python
# 源码第 1162-1170 行
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")

# 三个都完成后才执行 finalize_report
builder.add_edge(
    ["write_conclusion", "write_report", "write_introduction"],
    "finalize_report"
)
```

**这个设计的精妙之处：**
- `conduct_interview` 同时连接到 3 个节点 → 自动触发并行
- `finalize_report` 有 3 个入边 → 自动等待全部完成
- 不需要手动写 `Promise.all` 或 `join()`，LangGraph 的图语义自动处理

### 5.3 动手任务

1. 在阶段三的基础上，把访谈子图作为节点嵌入主图：
   ```python
   builder.add_node("conduct_interview", interview_graph)  # 子图作为节点！
   ```
2. 用 `initiate_all_interviews` 实现条件路由 + Send 并行
3. 添加三路并行报告生成节点
4. 添加 `finalize_report` 节点，组装最终报告

### 5.4 面试高频问题

1. **"Send 和普通 add_edge 的区别？"**
   → add_edge 是静态的一对一连接；Send 是动态的，运行时根据数据量决定创建多少个实例，类似 Spark 的 map
2. **"并行节点的结果怎么聚合？"**
   → 所有并行实例返回的 `sections` 字段通过 `operator.add` 自动合并到全局状态的同一个列表
3. **"为什么报告主体、引言、结论要分三个节点并行生成？"**
   → 三个部分互不依赖，并行生成可节省约 2/3 的等待时间

---

## 阶段六：提示词工程（建议 1 天）

> **目标：** 理解 7 套提示词的设计意图，能解释每套提示词解决什么问题。

### 6.1 提示词清单

| 提示词 | 作用 | 设计亮点 |
|--------|------|----------|
| `analyst_instructions` | 生成分析师 | 支持人类反馈注入，动态调整分析师视角 |
| `question_instructions` | 分析师提问 | "有趣且具体"框架，避免泛泛而谈 |
| `search_instructions` | 生成搜索查询 | SystemMessage 形式，聚焦最后一个问题 |
| `answer_instructions` | 专家回答 | 严格限制只用 context，防幻觉，强制引用标注 |
| `section_writer_instructions` | 写小节 | 指定 Markdown 结构，去重来源，中文输出 |
| `report_writer_instructions` | 写报告主体 | 整合多视角，隐藏分析师名字，统一来源列表 |
| `intro_conclusion_instructions` | 写引言/结论 | 一个模板两种用途，100字精炼 |

### 6.2 面试角度

1. **"专家回答为什么禁止引入外部知识？"**
   → 保证研究可追溯性，每个论点都有出处，这是做"深度研究"的核心价值
2. **"source 引用格式为什么用 [1] [2] 这种？"**
   → 模仿学术论文引用格式，增加报告可信度；也方便用户溯源

---

## 阶段七：生产部署（建议 1 天）

> **目标：** 理解 Docker 部署架构和 LangGraph API Server 的工作方式。

### 7.1 架构

```
用户 → langgraph-api (8124端口)
          │
          ├── langgraph-redis (消息队列 + 缓存)
          │
          └── langgraph-postgres (状态持久化)
```

### 7.2 关键配置

```json
// langgraph.json
{
    "graphs": {
        "research_assistant": "./research_assistant.py:graph"  // 入口
    },
    "python_version": "3.12"
}
```

### 7.3 面试角度

1. **"为什么需要 Redis 和 PostgreSQL？"**
   → Redis 做消息队列（API 异步调用），PostgreSQL 做状态持久化（替代 MemorySaver），生产环境不能把状态放在内存
2. **"和本地运行的区别？"**
   → 本地用 MemorySaver（内存），生产用 SqliteSaver/PostgresSaver（数据库）；本地同步 invoke，生产通过 SDK 异步调用

---

## 面试实战准备清单

### 必须能画出的图

- [ ] 整体架构图（Map-Reduce 全景）
- [ ] 访谈子图内部流程图
- [ ] 状态传递图（ResearchGraphState 各字段在哪些节点被读写）

### 必须能回答的问题

| 类别 | 问题 |
|------|------|
| **架构** | "整个系统的架构模式是什么？" → Map-Reduce |
| **并行** | "并行是怎么实现的？" → Send API + operator.add |
| **子图** | "为什么用子图？" → 可复用、独立状态、可独立测试 |
| **人机协同** | "人类怎么介入？" → interrupt_before + checkpointer |
| **状态管理** | "operator.add 的作用？" → 并行结果自动聚合 |
| **防幻觉** | "怎么保证专家不乱说？" → 只用 context，强制引用 |
| **性能** | "API 调用次数？" → 3分析师×2轮≈6次访谈 + 3小节 + 4报告 ≈ 13次 |
| **扩展性** | "怎么支持更多数据源？" → 加搜索节点，连到 answer_question |

### 加分项（主动提及）

1. "这个项目用了一个很巧妙的设计：**子图作为节点嵌入主图**，实现了 Map-Reduce 模式的自动并行和聚合"
2. "搜索结果用 XML 格式而不是 JSON，是因为 LLM 对 XML 标签的解析能力更强"
3. "finalize_report 做了来源去重和格式清洗，这是生产级代码的细节处理"

---

## 学习路线总览

```
阶段一 (2~3天)        阶段二 (2天)         阶段三 (1~2天)
数据模型 + LLM调  →  StateGraph线性流  →  条件路由 + 人机协同
                     (最小可运行Demo)
                                                      ↓
阶段七 (1天)          阶段六 (1天)         阶段四 (2~3天)       阶段五 (2天)
生产部署          ←  提示词工程         ←  访谈子图(核心)  ←  Map-Reduce并行
                                          (最难的部分)
```

**总时长：约 11~14 天，其中阶段四是重中之重。**
