# 阶段四 · Day 6：访谈子图（上）—— 理解架构 + 实现前三个节点

## 今日目标

1. 理解访谈子图的完整架构（6 节点 + 循环 + 并行）
2. 实现 generate_question 节点（分析师提问）
3. 实现 search_web 节点（Web 搜索）
4. 实现 search_baike 节点（百科搜索）
5. 理解并行搜索的 fan-out / fan-in 模式

---

## 知识点一：访谈子图的全貌

这是整个项目最复杂的图，包含三种模式：**循环 + 并行 + 条件路由**

```
                 ┌──────────────────────────────────────────────┐
                 │               访谈子图                        │
                 │                                                │
START → ask_question ──┬──→ search_web  ────┐                   │
         ↑              │                    ├──→ answer_question │
         │              └──→ search_baike ──┘          │        │
         │                    （并行搜索）               │        │
         │                                              ↓        │
         │                                        route_messages │
         │                                         │          │  │
         │                                    继续提问    保存访谈 │  │
         │                                         │          ↓  │
         └─────────────────────────────────────────┘   write_section
                                                            │
                                                            ↓
                                                           END
                 └──────────────────────────────────────────────┘
```

**三种模式：**
- **并行（fan-out/fan-in）：** ask_question 同时触发 search_web 和 search_baike，两个搜索都完成后才进 answer_question
- **循环：** route_messages 决定回到 ask_question 还是进入 save_interview
- **条件路由：** 和阶段三一样，根据状态决定走哪条路

---

## 知识点二：fan-out / fan-in 怎么实现

```python
# fan-out：一个节点连到两个节点（并行触发）
builder.add_edge("ask_question", "search_web")
builder.add_edge("ask_question", "search_baike")

# fan-in：两个节点都连到同一个节点（等两个都完成才执行）
builder.add_edge("search_web", "answer_question")
builder.add_edge("search_baike", "answer_question")
```

**LangGraph 的自动处理：**
- `answer_question` 有两个入边 → LangGraph 自动等两个搜索都完成后再执行
- 两个搜索节点都返回 `{"context": [doc]}` → 因为 `context` 是 `operator.add`，自动合并

```
search_web  返回 {"context": ["Web搜索结果"]}
search_baike 返回 {"context": ["百科搜索结果"]}
                          ↓ operator.add 自动合并
context = ["Web搜索结果", "百科搜索结果"]
```

---

## 知识点三：提示词模板

访谈子图用到 4 套提示词，先认识它们：

### 1. question_instructions（分析师提问用）

```python
question_instructions = """你是一名分析师，需要通过访谈专家来了解一个具体主题。

你的目标是提炼与该主题相关的「有趣且具体」的洞见。

1. 有趣（Interesting）：让人感到意外或非显而易见的观点。
2. 具体（Specific）：避免泛泛而谈，包含专家提供的具体案例或细节。

以下是你的关注主题与目标设定：{goals}

请先用符合你人设的名字进行自我介绍，然后提出你的第一个问题。

持续追问，逐步深入，逐步完善你对该主题的理解。

当你认为信息已充分，请以这句话结束访谈：「非常感谢您的帮助!」

请始终保持与你的人设与目标一致的说话方式。"""
```

**注意 `{goals}` 占位符：** 运行时用 `analyst.persona` 填充，让 LLM 知道自己扮演谁。

### 2. search_instructions（生成搜索词用）

```python
search_instructions = SystemMessage(content="""你将获得一段分析师与专家之间的对话。

你的目标是基于这段对话，为Web搜索生成一条结构良好的查询语句。

首先，通读整段对话。

特别关注分析师最后提出的问题。

将这个最终问题转化为结构良好的 Web 搜索查询。""")
```

**注意：** 这是 SystemMessage，不是字符串模板，不需要 `.format()`。

### 3. answer_instructions（专家回答用）

```python
answer_instructions = """你是一位被分析师访谈的专家。

以下是分析师的关注领域：{goals}。

你的目标是回答访谈者提出的问题。

回答问题时，请仅使用以下上下文：

{context}

回答须遵循如下要求：
1. 只使用上下文中提供的信息。
2. 不要引入上下文之外的信息...
3. 在涉及具体论断时，标注引用来源编号 [1] [2]...
4. 在答案结尾处按顺序列出引用来源。"""
```

**两个占位符：** `{goals}` 填 analyst.persona，`{context}` 填搜索结果。

---

## 知识点四：三个节点的实现解析

### generate_question（源码第 640-664 行）

```python
def generate_question(state: InterviewState):
    analyst = state["analyst"]
    messages = state["messages"]

    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)

    return {"messages": [question]}  # messages 自动累加
```

**读写：** 读 analyst + messages → 写 messages（累加）

**关键：** `[SystemMessage(...)] + messages` 把系统提示词放在最前面，对话历史跟在后面。LLM 先知道"你是谁"（persona），再看到"之前聊了什么"（messages）。

### search_web（源码第 667-694 行）

```python
def search_web(state: InterviewState):
    # 1. 对话 → 搜索词
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state['messages'])

    # 2. 搜索词 → 搜索结果
    search_docs = tavily_search.invoke(search_query.search_query)

    # 3. 格式化为 XML
    formatted = "\n\n---\n\n".join([
        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
        for doc in search_docs
    ])

    return {"context": [formatted]}  # context 自动累加
```

**读写：** 读 messages → 写 context（累加）

**三步转换：** 对话历史 → 搜索词 → 搜索结果 → XML 格式

### search_baike（源码第 697-727 行）

```python
def search_baike(state: InterviewState):
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state['messages'])

    search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()

    formatted = "\n\n---\n\n".join([
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
        for doc in search_docs
    ])

    return {"context": [formatted]}
```

**和 search_web 的区别：**
- search_web 用 Tavily（实时 Web 搜索），结果格式是 `{"url": "...", "content": "..."}`
- search_baike 用 WikipediaLoader（百科），结果是 LangChain Document 对象，有 `metadata["source"]` 和 `page_content`

---

## 动手任务

### 任务：实现访谈子图的前半部分（3 个节点 + 并行搜索）

**目标：** 搭出 `ask_question → [search_web, search_baike] → answer_question` 的并行搜索流程（暂不加循环）。

**骨架代码见 `phase4_day6_skeleton.py`**

**你只需要实现：**
1. `generate_question` 节点函数
2. `search_web` 节点函数（Tavily 搜索）
3. `search_baike` 节点函数（Wikipedia 搜索）
4. 图的构建（节点 + 边）

**暂不实现：** answer_question / route_messages / save_interview / write_section（Day 7 内容）

**验收标准：**
- 运行后 ask_question 生成一个问题
- search_web 和 search_baike 并行执行
- 最终 context 里有两类搜索结果（Web + 百科）
- 打印出搜索到的内容摘要

**注意：** Tavily 需要 API Key。如果你没有，在 `.env` 里加上 `TAVILY_API_KEY`。注册地址：https://tavily.com（免费 1000 次搜索）

---

## 今日面试题

**Q1: "并行搜索的结果怎么合并？"**
> 两个搜索节点都返回 `{"context": [doc]}`，因为 context 是 `Annotated[list, operator.add]`，LangGraph 自动用 `+` 合并。answer_question 执行时拿到的 context 包含两个来源的结果。

**Q2: "为什么搜索结果用 XML 格式？"**
> LLM 对 XML 标签的解析和理解能力比 JSON 更强。用 `<Document href="...">内容</Document>` 格式，LLM 能清楚区分"来源"和"内容"，方便后续引用标注。

**Q3: "fan-in 怎么实现？两个搜索都完成才执行 answer_question 是怎么做到的？"**
> answer_question 有两个入边（从 search_web 和 search_baike 各一条）。LangGraph 的执行引擎检测到一个节点有多个未完成的入边时，会等所有入边的源节点都执行完毕后才触发该节点。这是图执行引擎的标准 fan-in 行为。

---

## 参考源码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| generate_question | `research_assistant.py` | 640-664 |
| search_web | `research_assistant.py` | 667-694 |
| search_baike | `research_assistant.py` | 697-727 |
| question_instructions | `research_assistant.py` | 348-364 |
| search_instructions | `research_assistant.py` | 370-378 |
| 访谈子图构建 | `research_assistant.py` | 1053-1100 |
