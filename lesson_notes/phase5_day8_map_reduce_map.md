# 阶段五 · Day 8：Map-Reduce（上）—— Send API + 子图嵌入主图

## 今日目标

1. 理解 Map-Reduce 架构的全貌（主图 + 子图的关系）
2. 掌握 Send API 的原理与用法
3. 实现 `initiate_all_interviews`（Map 函数）
4. 把 Day 6-7 的访谈子图嵌入主图
5. 搭出主图的基本骨架（节点 + 边）

---

## 知识点一：Map-Reduce 全景图

整个 Deep Research 系统就是一次 Map-Reduce：

```
┌──────────────────────────────────────────────────────────────────┐
│                    主图 ResearchGraphState                         │
│                                                                    │
│  START → create_analysts → human_feedback                          │
│                               │                                    │
│                          条件路由                                   │
│                         ╱          ╲                                │
│                   有反馈             无反馈                          │
│                     ↓                ↓                              │
│              create_analysts    Send × N (Map)                      │
│                                   ↓                                │
│                    ┌─────────────────────────────┐                 │
│                    │   conduct_interview (子图) ×N │ ← 并行执行      │
│                    │   每个分析师独立运行一个实例     │                 │
│                    │   输出 sections → operator.add │                │
│                    └─────────────────────────────┘                 │
│                                   ↓                                │
│                    ┌──── Reduce 三路并行 ────┐                      │
│                    │ write_report            │                      │
│                    │ write_introduction      │ ← Day 9 内容        │
│                    │ write_conclusion        │                      │
│                    └────────┬────────────────┘                      │
│                             ↓                                      │
│                    finalize_report → END                            │
└──────────────────────────────────────────────────────────────────┘
```

**Map-Reduce 三阶段：**

| 阶段 | 做什么 | 关键机制 |
|------|--------|----------|
| Map | N 个分析师并行访谈 | `Send("conduct_interview", {...})` |
| Shuffle | 各子图输出 sections 自动累加 | `Annotated[list, operator.add]` |
| Reduce | 合并 sections → 引言/主体/结论 | 三路并行 + fan-in |

---

## 知识点二：ResearchGraphState — 主图状态

```python
class ResearchGraphState(TypedDict):
    topic: str                                    # 输入：研究主题
    max_analysts: int                             # 输入：分析师数量
    human_analyst_feedback: str                   # 人机协同：反馈
    analysts: List[Analyst]                       # 中间：分析师列表
    sections: Annotated[list, operator.add]       # 关键！并行子图输出自动累加
    introduction: str                             # Reduce 输出
    content: str                                  # Reduce 输出
    conclusion: str                               # Reduce 输出
    final_report: str                             # 最终输出
```

**`sections` 是 Map 和 Reduce 之间的桥梁：**

```
子图1 返回 {"sections": ["小节1"]}
子图2 返回 {"sections": ["小节2"]}     →  operator.add  →  sections = ["小节1", "小节2", "小节3"]
子图3 返回 {"sections": ["小节3"]}
```

这就是为什么 Day 1 就学了 `Annotated[list, operator.add]` — 它是 Map-Reduce 聚合的核心！

---

## 知识点三：Send API — 动态并行

```python
from langgraph.types import Send

def initiate_all_interviews(state: ResearchGraphState):
    """Map 函数：为每个分析师创建一个并行任务"""

    # 如果有人类反馈，回到 create_analysts 重新生成
    human_analyst_feedback = state.get('human_analyst_feedback')
    if human_analyst_feedback:
        return "create_analysts"

    # 无反馈，为每个分析师创建 Send
    topic = state["topic"]
    return [
        Send("conduct_interview", {
            "analyst": analyst,
            "messages": [HumanMessage(
                content=f"所以你说你在写一篇关于{topic}的文章?"
            )]
        })
        for analyst in state["analysts"]
    ]
```

### Send 的本质

```
add_edge（静态边）：
  固定连接两个节点，编译时确定
  builder.add_edge("A", "B")  → A 执行完，B 执行一次

Send（动态边）：
  运行时根据数据量创建 N 个节点实例
  返回 [Send("B", data1), Send("B", data2), Send("B", data3)]
  → B 被执行 3 次，每次拿不同的输入
```

**类比：**
- `add_edge` = 打电话给一个人
- `Send` = 群发消息，每个人收到的内容不同

### Send 的参数

```python
Send(节点名, 输入状态字典)
```

- 第一个参数：目标节点名（字符串）
- 第二个参数：传给这个节点实例的初始状态

**注意：** Send 返回的是 **列表**，不是单个 Send。因为可以有多个并行任务。

### initiate_all_interviews 的双重角色

这个函数既是 **Map 函数**，也是 **条件路由**：

```python
# 作为条件路由
builder.add_conditional_edges(
    "human_feedback",
    initiate_all_interviews,
    ["create_analysts", "conduct_interview"]  # 可能的目标
)
```

| 返回值 | 含义 |
|--------|------|
| `"create_analysts"` (字符串) | 有反馈 → 回去重新生成分析师 |
| `[Send(...), Send(...)]` (列表) | 无反馈 → 启动并行访谈 |

---

## 知识点四：子图嵌入主图

```python
# interview_graph 是 Day 7 编译好的子图
# 直接作为 add_node 的参数！
builder.add_node("conduct_interview", interview_graph)
```

**这就是 LangGraph 子图的魔法：** 编译好的子图可以当作一个普通节点使用。

```
子图内部：ask_question → search → answer → ... → write_section → END
子图外部：看起来就是一个叫 "conduct_interview" 的黑盒节点
```

**数据流：**

```
主图传入: Send("conduct_interview", {"analyst": ..., "messages": [...]})
                                    ↓
子图接收: InterviewState 的初始状态 = {"analyst": ..., "messages": [...]}
                                    ↓
子图执行: 6个节点跑完循环
                                    ↓
子图输出: {"sections": ["小节内容"], ...}
                                    ↓
主图接收: sections 通过 operator.add 累加到 ResearchGraphState.sections
```

---

## 知识点五：主图骨架 — 所有节点和边

```python
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(ResearchGraphState)

# ===== 7 个节点 =====
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_graph)  # 子图嵌入！
builder.add_node("write_report", write_report)          # Day 9
builder.add_node("write_introduction", write_introduction)  # Day 9
builder.add_node("write_conclusion", write_conclusion)      # Day 9
builder.add_node("finalize_report", finalize_report)        # Day 9

# ===== 边 =====
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")

# 条件路由 + Send（Map 的入口）
builder.add_conditional_edges(
    "human_feedback",
    initiate_all_interviews,
    ["create_analysts", "conduct_interview"]
)

# Reduce 三路并行
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")

# 三路 fan-in
builder.add_edge(
    ["write_conclusion", "write_report", "write_introduction"],
    "finalize_report"
)

builder.add_edge("finalize_report", END)

# 编译
memory = MemorySaver()
graph = builder.compile(
    interrupt_before=['human_feedback'],
    checkpointer=memory
)
```

**图的执行路径：**

```
START → create_analysts → human_feedback (暂停，等人类)
                              ↓ (人类批准)
                        initiate_all_interviews
                              ↓ (返回 [Send, Send, Send])
                    conduct_interview × 3 (并行，各自独立运行子图)
                              ↓ (sections 累加完成)
                    ┌─────────┼─────────┐
               write_report  write_intro  write_conclusion (并行)
                    └─────────┼─────────┘
                              ↓ (fan-in，全部完成后)
                       finalize_report → END
```

---

## 动手任务

### 任务：搭建主图骨架

**在 Day 7 的完整代码基础上，添加：**

1. `ResearchGraphState` 状态类（如果还没有的话）
2. `initiate_all_interviews` 函数
3. 主图骨架（7 个节点 + 所有边）
4. 编译 + 测试运行

**Reduce 节点（write_report / write_introduction / write_conclusion / finalize_report）今天用占位函数实现，明天补充完整。**

**骨架代码见 `phase5_day8_skeleton.py`**

**验收标准：**
1. 主图能编译通过
2. 能看到完整的图结构（打印节点和边）
3. initiate_all_interviews 返回正确数量的 Send 对象

---

## 今日面试题

**Q1: "Send 和 add_edge 的区别是什么？"**
> add_edge 是静态连接，编译时就确定了 A → B 的一对一关系。Send 是动态的，运行时根据数据决定创建多少个节点实例。类比：add_edge 是固定电话线，Send 是群发消息。

**Q2: "并行子图的结果怎么聚合？"**
> 子图返回 `{"sections": ["小节"]}`，主图的 `sections` 字段定义为 `Annotated[list, operator.add]`。3 个并行子图的输出会自动用 `+` 合并成 `["小节1", "小节2", "小节3"]`。这就是 operator.add 在 Map-Reduce 中的核心作用。

**Q3: "子图怎么嵌入主图？"**
> 编译好的子图 `interview_graph` 可以直接作为 `add_node` 的第二个参数。主图通过 Send 给每个子图实例传入不同的初始状态（不同的 analyst），子图独立运行后，输出通过 operator.add 累加回主图状态。

**Q4: "initiate_all_interviews 为什么既是路由函数又是 Map 函数？"**
> 它在 `add_conditional_edges` 中使用，返回值有两种类型：字符串 `"create_analysts"`（回到重生成）和 `Send` 列表（启动并行访谈）。LangGraph 约定：返回字符串走静态边，返回 Send 列表走动态并行。

---

## 参考源码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| ResearchGraphState | `research_assistant.py` | 297-316 |
| initiate_all_interviews | `research_assistant.py` | 859-891 |
| 主图构建 | `research_assistant.py` | 1103-1184 |
| Send 导入 | `research_assistant.py` | 207 |
| fan-in 多入边 | `research_assistant.py` | 1167-1170 |
