# 阶段四 · Day 7：访谈子图（下）—— 完成剩余节点 + 组装循环

## 今日目标

1. 实现 answer_question 节点（专家基于 context 回答）
2. 实现 route_messages 路由（循环控制）
3. 实现 save_interview 节点（对话历史 → 字符串）
4. 实现 write_section 节点（访谈内容 → 报告小节）
5. 把 6 个节点 + 条件路由 + 并行搜索组装成完整子图
6. 跑通一次完整的访谈流程

---

## 知识点一：answer_question——专家只基于搜索结果回答

```python
def generate_answer(state: InterviewState):
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]  # 两个搜索节点的结果已经累加到这里

    system_message = answer_instructions.format(
        goals=analyst.persona,
        context=context
    )

    answer = llm.invoke([SystemMessage(content=system_message)] + messages)
    answer.name = "expert"  # ← 关键：标记这条消息来自"专家"

    return {"messages": [answer]}  # messages 自动累加
```

**两个关键设计：**

1. **`answer.name = "expert"`** — 给这条 AIMessage 打标签，route_messages 用它来统计专家回答了几次
2. **只用 context 回答** — 提示词里明确写了"请仅使用以下上下文"，防止幻觉

**读写：** 读 analyst + messages + context → 写 messages（累加）

---

## 知识点二：route_messages——循环的控制中心

```python
def route_messages(state: InterviewState, name: str = "expert"):
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)

    # 条件1：专家回答次数 >= 上限 → 结束
    num_responses = len([
        m for m in messages
        if isinstance(m, AIMessage) and m.name == name
    ])
    if num_responses >= max_num_turns:
        return 'save_interview'

    # 条件2：分析师主动说"非常感谢您的帮助!" → 结束
    last_question = messages[-2]  # 倒数第二条是分析师的提问
    if "非常感谢您的帮助!" in last_question.content:
        return 'save_interview'

    # 否则：继续提问
    return "ask_question"
```

**为什么用 `messages[-2]`？**

```
messages 列表（从旧到新）：
  [0] HumanMessage("你在写关于AI医疗的文章?")  ← 初始消息
  [1] AIMessage("我是张伟博士，请问...")        ← 分析师提问
  [2] AIMessage("根据文献...")                  ← 专家回答，name="expert"
  [3] AIMessage("那中国三甲医院呢？")           ← 分析师追问
  [4] AIMessage("北京协和医院...")              ← 专家回答，name="expert"

当执行到 route_messages 时，最后一条 [4] 是专家回答
倒数第二条 [3] 是分析师的提问 → 检查是否包含结束信号
```

**退出条件总结：**

| 条件 | 检查方式 | 说明 |
|------|----------|------|
| 轮次达到上限 | 统计 `name=="expert"` 的 AIMessage 数量 | 硬限制，防止无限循环 |
| 分析师主动结束 | 检查 `messages[-2]` 是否包含"非常感谢您的帮助!" | 软限制，LLM 自己判断信息足够 |

---

## 知识点三：save_interview——对话历史转字符串

```python
from langchain_core.messages import get_buffer_string

def save_interview(state: InterviewState):
    messages = state["messages"]
    interview = get_buffer_string(messages)  # 把消息列表转成一段文字
    return {"interview": interview}
```

**`get_buffer_string` 做了什么？**

```python
# 输入：消息列表
[
    HumanMessage(content="你在写关于AI医疗的文章?"),
    AIMessage(content="我是张伟博士，请问..."),
    AIMessage(content="根据文献...", name="expert"),
]

# 输出：一段文字
"Human: 你在写关于AI医疗的文章?\nAI: 我是张伟博士，请问...\nAI: 根据文献..."
```

**为什么需要这一步？** 后面的 write_section 需要把访谈内容作为上下文传给 LLM，但 LLM 不接受消息对象列表，需要纯文本。

---

## 知识点四：write_section——访谈内容变成报告小节

```python
def write_section(state: InterviewState):
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"使用这些来源撰写你的小节: {context}")
    ])

    return {"sections": [section.content]}  # sections 会被主图的 operator.add 累加
```

**读写：** 读 interview + context + analyst → 写 sections

**这个节点是子图和主图的桥梁：** 子图返回 `{"sections": ["小节内容"]}`，主图的 `ResearchGraphState.sections` 通过 `operator.add` 自动累加所有子图的输出。

---

## 知识点五：组装完整子图

```python
interview_builder = StateGraph(InterviewState)

# 6 个节点
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_baike", search_baike)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# 边
interview_builder.add_edge(START, "ask_question")

# 并行搜索（fan-out）
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_baike")

# 汇聚到回答（fan-in）
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_baike", "answer_question")

# 条件路由（循环控制）
interview_builder.add_conditional_edges(
    "answer_question",
    route_messages,
    ['ask_question', 'save_interview']
)

# 收尾
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# 编译
interview_graph = interview_builder.compile()
```

**三种模式汇聚在一张图里：**

```
并行：ask_question → [search_web, search_baike] → answer_question
循环：answer_question → route_messages → ask_question（回到开头）
条件路由：route_messages → "ask_question" 或 "save_interview"
```

---

## 提示词补充：section_writer_instructions

```python
section_writer_instructions = """你是一名资深技术写作者。

你的任务是基于一组来源文档，撰写一段简洁、易读的报告小节。

1. 先分析来源文档内容：
- 每个文档的名称在文档开头，以 <Document 标签呈现。

2. 使用 Markdown 制作小节结构：
- 用 ## 作为小节标题
- 用 ### 作为小节内的小标题

3. 按结构撰写：
 a. 标题（## 头）
 b. 摘要（### 头）
 c. 参考来源（### 头）

4. 标题需要贴合分析师的关注点并具有吸引力：
{focus}

5. 关于摘要部分：
- 先给出与分析师关注点相关的背景/上下文
- 强调访谈中获得的新颖、有趣或令人意外的洞见
- 使用到来源文档时，按使用顺序创建编号
- 不要提及访谈者或专家的名字
- 控制在约 400 字以内
- 在报告正文中使用数字引用（如 [1]、[2]）
- **重要：生成的小节内容必须全部使用中文**

6. 在参考来源部分：
- 列出报告中使用到的全部来源
- 每个来源单独一行
- 合并重复来源"""
```

---

## 动手任务

### 任务：完成访谈子图的全部代码

**在 Day 6 的代码基础上，添加：**

1. `generate_answer` 节点函数
2. `route_messages` 路由函数
3. `save_interview` 节点函数
4. `write_section` 节点函数
5. 补全图的边（条件路由 + 收尾）
6. 运行完整访谈

**骨架代码见 `phase4_day7_skeleton.py`**

**验收标准：**
1. 分析师提问 → 并行搜索 → 专家回答 → 循环 2 轮
2. 2 轮结束后保存访谈 → 生成报告小节
3. 最终打印出报告小节内容

**注意：** 如果没有 Tavily API Key，search_web 会报错。可以在 search_web 中加个 try-except 返回空字符串作为降级方案。

---

## 今日面试题

**Q1: "route_messages 的两种退出条件是什么？"**
> 硬限制：专家回答次数达到 max_num_turns（统计 name="expert" 的 AIMessage 数量）。软限制：分析师在提问中包含"非常感谢您的帮助!"。两个条件满足任一个就结束循环。

**Q2: "为什么用 messages[-2] 而不是 messages[-1] 检查结束信号？"**
> 因为 route_messages 在 answer_question 之后执行。此时 messages 最后一条是专家的回答（name="expert"），倒数第二条才是分析师的提问。结束信号写在分析师的提问里。

**Q3: "write_section 返回的 sections 怎么传递到主图？"**
> 子图的 InterviewState.sections 返回 `["小节内容"]`。在主图中，conduct_interview 节点的输出会映射到 ResearchGraphState.sections。因为 sections 是 `Annotated[list, operator.add]`，多个并行子图的 sections 自动累加。

**Q4: "get_buffer_string 的作用是什么？"**
> 把消息对象列表（HumanMessage/AIMessage）转换成纯文本字符串。因为后续的 write_section 需要把访谈内容作为文本传给 LLM，LLM 不接受消息对象。

---

## 参考源码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| generate_answer | `research_assistant.py` | 730-759 |
| save_interview | `research_assistant.py` | 762-782 |
| route_messages | `research_assistant.py` | 785-821 |
| write_section | `research_assistant.py` | 828-856 |
| answer_instructions | `research_assistant.py` | 384-410 |
| section_writer_instructions | `research_assistant.py` | 416-466 |
| 子图构建 | `research_assistant.py` | 1053-1100 |
