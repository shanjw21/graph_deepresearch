# 阶段五 · Day 9：Map-Reduce（下）—— Reduce 三路并行 + finalize_report

## 今日目标

1. 实现 `write_report`（sections → 报告主体）
2. 实现 `write_introduction`（sections → 引言）
3. 实现 `write_conclusion`（sections → 结论）
4. 实现 `finalize_report`（引言 + 主体 + 结论 → 完整报告）
5. 端到端运行完整系统，生成一份研究报告

---

## 知识点一：Reduce 阶段的全貌

```
                    conduct_interview × N（Map 完成）
                              ↓
              sections = ["小节1", "小节2", "小节3"]（operator.add 已累加）
                              ↓
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
        write_report   write_introduction  write_conclusion
        (主体内容)      (引言)              (结论)
              │               │               │
              └───────────────┼───────────────┘
                              ↓  （fan-in：全部完成后才执行）
                       finalize_report
                       (引言 + 主体 + 结论 → 拼装)
                              ↓
                        final_report → END
```

**三个 Reduce 节点的共同点：** 都读 `sections` 和 `topic`，互不依赖，所以并行执行。

**finalize_report 的特殊之处：** 不调用 LLM，纯字符串拼装。

---

## 知识点二：sections 预处理 — 列表转字符串

三个 Reduce 节点的第一步都一样：把 `sections` 列表合成一段文本。

```python
sections = state["sections"]  # ["小节1", "小节2", "小节3"]

formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
# → "小节1\n\n小节2\n\n小节3"
```

**为什么需要这一步？** sections 是 list，但 LLM 需要纯文本作为提示词的输入。

---

## 知识点三：write_report — 备忘录整合成报告主体

```python
report_writer_instructions = """你是一名技术写作者，正在为如下主题撰写报告：

{topic}

你拥有一支分析师团队。每位分析师完成了两件事：
1. 围绕一个具体子主题，访谈了一位专家。
2. 将发现写成一份备忘录（memo）。

你的任务：
1. 你将收到分析师们的备忘录集合。
2. 仔细思考每份备忘录的洞见。
3. 将它们整合为简洁的总体总结，串联起所有备忘录的中心观点。
4. 把每份备忘录的关键信息归纳成一个连贯的单一叙述。

报告格式要求：
1. 使用 Markdown 格式。
2. 报告不要有任何前言。
3. 不使用任何小标题。
4. 报告以一个标题开头：## Insights
5. 报告中不要提及任何分析师的名字。
6. 保留备忘录中的引用标注（如 [1]、[2]）。
7. 汇总最终来源列表，并以 ## Sources 作为小节标题。
8. 按顺序列出来源且不要重复。

以下是分析师提供的备忘录，请基于此撰写报告：

{context}"""

def write_report(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    system_message = report_writer_instructions.format(
        topic=topic,
        context=formatted_str_sections
    )

    report = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="基于这些备忘录撰写一份报告。")
    ])

    return {"content": report.content}
```

**关键设计：**
- 输入：`sections`（所有子图输出）+ `topic`
- 输出：`content` 字段
- 报告以 `## Insights` 开头，`## Sources` 结尾 → finalize_report 会利用这个结构

---

## 知识点四：write_introduction & write_conclusion — 共用一套提示词

两个函数逻辑几乎一模一样，只是 HumanMessage 不同：

```python
intro_conclusion_instructions = """你是一名技术写作者，正在完成主题为 {topic} 的报告。

你将获得报告的全部小节。

你的任务是撰写简洁而有说服力的引言或结论。

由用户告知写引言还是结论。

两者均不需要任何前言。

目标约 100 字：
- 引言：精炼预览各小节要点
- 结论：精炼回顾各小节要点

使用 Markdown 格式。

引言要求：创建一个有吸引力的标题，并用 # 作为标题头。

引言小节标题使用：## 引言
结论小节标题使用：## 结论

撰写时可参考以下小节内容：{formatted_str_sections}"""
```

```python
def write_introduction(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    instructions = intro_conclusion_instructions.format(
        topic=topic,
        formatted_str_sections=formatted_str_sections
    )

    intro = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content="撰写报告引言")
    ])

    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    instructions = intro_conclusion_instructions.format(
        topic=topic,
        formatted_str_sections=formatted_str_sections
    )

    conclusion = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content="撰写报告结论")
    ])

    return {"conclusion": conclusion.content}
```

**为什么两个函数不合并？** 因为在 LangGraph 中，一个节点只能返回自己的输出字段。write_introduction 写 `introduction`，write_conclusion 写 `conclusion`，必须分开。

---

## 知识点五：finalize_report — 纯字符串拼装（不调用 LLM）

```python
def finalize_report(state: ResearchGraphState):
    content = state["content"]

    # 清理格式：去掉 ## Insights 标题
    if content.startswith("## Insights"):
        content = content.strip("## Insights")

    # 从主体中分离出 Sources 部分
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    # 拼装：引言 + 分隔线 + 主体 + 分隔线 + 结论
    final_report = (
        state["introduction"] +
        "\n\n---\n\n" +
        content +
        "\n\n---\n\n" +
        state["conclusion"]
    )

    # 加回 Sources
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources

    return {"final_report": final_report}
```

**为什么要把 Sources 分离出来？**

```
write_report 输出的 content 长这样：
  ## Insights
  报告正文内容...

  ## Sources
  [1] 来源1
  [2] 来源2

finalize_report 的目标格式：
  # 报告标题          ← 引言（write_introduction 生成）
  ## 引言
  引言内容...

  ---

  报告正文内容...     ← content（去掉 ## Insights 标题）

  ---

  ## 结论            ← 结论（write_conclusion 生成）
  结论内容...

  ## Sources         ← 来源（从 content 中提取，放到最后）
  [1] 来源1
```

Sources 是全报告共享的，应该放在最后。所以需要从 content 中提取出来再拼到末尾。

---

## 动手任务

### 任务：实现 4 个 Reduce 节点 + 端到端测试

**在 Day 8 的代码基础上：**

1. 添加 `report_writer_instructions` 提示词
2. 添加 `intro_conclusion_instructions` 提示词
3. 实现 `write_report`
4. 实现 `write_introduction`
5. 实现 `write_conclusion`
6. 实现 `finalize_report`
7. 取消测试 3 的注释，端到端运行完整系统

**骨架代码见 `phase5_day9_skeleton.py`**

**验收标准：**
1. 3 个分析师并行访谈完成
2. 三路 Reduce 并行生成引言/主体/结论
3. final_report 输出完整 Markdown 报告
4. 报告结构：标题 → 引言 → 主体 → 结论 → Sources

---

## 今日面试题

**Q1: "write_introduction 和 write_conclusion 为什么不合并成一个节点？"**
> 两个函数逻辑一样但输出字段不同（`introduction` vs `conclusion`）。LangGraph 中一个节点只能返回自己的状态字段。分开也是为了并行——互不依赖，同时执行节省时间。

**Q2: "finalize_report 为什么不调用 LLM？"**
> 它只是字符串拼装：把引言、主体、结论按格式拼接，加上分隔线和 Sources。不需要 LLM 的推理能力，纯 Python 字符串操作就够了，省了一次 API 调用。

**Q3: "为什么要把 Sources 从 content 中分离出来？"**
> `write_report` 生成的主体内容里包含 `## Sources`，但在最终报告中，Sources 应该放在所有内容（引言 + 主体 + 结论）之后。所以 finalize_report 先把 Sources 从 content 中拆出来，拼完后再追加到末尾。

**Q4: "整个系统总共多少次 LLM 调用？"**
> 大约 13 次：create_analysts 1次 + 访谈 3分析师×2轮×3节点(提问+搜索+回答)=18次 + write_section×3=3次 + write_report 1次 + write_introduction 1次 + write_conclusion 1次 ≈ 24次。其中 Map 阶段（访谈）占大头，Reduce 阶段只有 3 次。

---

## 参考源码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| report_writer_instructions | `research_assistant.py` | 472-506 |
| intro_conclusion_instructions | `research_assistant.py` | 512-536 |
| write_report | `research_assistant.py` | 894-927 |
| write_introduction | `research_assistant.py` | 930-962 |
| write_conclusion | `research_assistant.py` | 965-997 |
| finalize_report | `research_assistant.py` | 1000-1043 |
