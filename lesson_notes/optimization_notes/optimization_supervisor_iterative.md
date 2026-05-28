# 进阶优化：Supervisor 迭代委派架构

## 背景知识

### 当前架构的局限性

`phase5_day9_skeleton.py` 使用的是 **一次性 Map-Reduce** 模式：

```
create_analysts → Send(所有分析师) → conduct_interview → write_report/intro/conclusion → finalize_report
```

这是一个 **DAG（有向无环图）**，流程线性向前，没有回头的机会。问题在于：

1. **分析师数量固定**：一开始生成 3 个，就不能再加了
2. **没有质量反馈**：访谈结果好不好、覆盖全不全，没有人评估
3. **不能补漏**：如果 3 个分析师都没覆盖到某个重要角度，报告就会缺失

### 官方教程的 Supervisor 循环

`deep_research_from_scratch` 的 Supervisor 模式是一个**循环决策器**：

```
supervisor → 委派研究 → 等结果 → 评估 → 不够？再委派 → ... → 够了 → ResearchComplete
```

核心区别：
- **动态**：根据中间结果决定下一步
- **迭代**：可以多次委派，逐步深入
- **有退出条件**：`ResearchComplete` 工具调用或最大迭代次数

### LangGraph 中的循环实现方式

LangGraph 支持循环图，通过 `conditional_edges` 实现：

```python
builder.add_conditional_edges(
    "evaluate",
    decide_next_step,
    {
        "conduct_more_research": "delegate_more_analysts",
        "finalize": "write_report"
    }
)
builder.add_edge("delegate_more_analysts", "conduct_interview")
builder.add_edge("conduct_interview", "evaluate")  # 形成循环
```

---

## 架构设计

### 新图结构

```
START → create_analysts → human_feedback → conduct_interview
                                              ↓
                                       accumulate_sections
                                              ↓
                                       evaluate_coverage ←──┐
                                              ↓             │
                                    ┌── needs_more ─────────┘
                                    ↓
                              delegate_more_analysts
                                    ↓
                              conduct_interview (循环回到 accumulate)
                                    ↓
                               enough_coverage
                                    ↓
                          write_report / intro / conclusion
                                    ↓
                              finalize_report → END
```

### 关键改动点

1. **新增 `evaluate_coverage` 节点**：LLM 评估已有 sections 是否覆盖了主题的关键方面
2. **新增 `delegate_more_analysts` 节点**：基于缺失角度，生成补充分析师
3. **新增循环边**：`conduct_interview → evaluate_coverage → delegate_more_analysts → conduct_interview`
4. **迭代次数保护**：`max_supervisor_iterations` 防止无限循环
5. **状态累积**：`accumulate_sections` 将每个分析师的结果累加到全局 state

### 状态管理

需要在 `ResearchGraphState` 中新增：
- `supervisor_iterations: int` — 当前迭代次数
- `max_supervisor_iterations: int = 3` — 最大迭代次数
- `coverage_gaps: str` — 评估发现的缺失角度
- `all_sections: Annotated[list, operator.add]` — 累积的所有 sections（与 sections 分开）

---

## 实现步骤

### 步骤 1：状态定义

```python
class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    all_sections: Annotated[list, operator.add]  # 累积
    supervisor_iterations: int
    max_supervisor_iterations: int
    coverage_gaps: str
    introduction: str
    content: str
    conclusion: str
    final_report: str
```

### 步骤 2：评估覆盖度节点

```python
evaluate_prompt = """你是一名研究质量评估专家。

研究主题：{topic}

已有的研究小节：
{sections}

请评估：
1. 已有小节覆盖了主题的哪些关键方面？
2. 还缺少哪些重要角度？
3. 是否需要补充研究？

如果需要补充研究，请以「需要补充：[具体角度]」格式列出缺失角度。
如果覆盖已充分，请回复「覆盖已充分，可以进入报告撰写阶段」。"""

def evaluate_coverage(state: ResearchGraphState):
    sections = state.get("sections", [])
    sections_text = "\n\n".join(str(s) for s in sections)
    
    prompt = evaluate_prompt.format(
        topic=state["topic"],
        sections=sections_text or "暂无，尚未完成任何研究"
    )
    
    response = llm.invoke([
        SystemMessage(content="你是一名研究质量评估专家。"),
        HumanMessage(content=prompt)
    ])
    
    needs_more = "需要补充" in response.content
    gaps = ""
    if needs_more:
        # 提取补充角度
        gaps = response.content
    
    return {
        "coverage_gaps": gaps,
        "supervisor_iterations": state.get("supervisor_iterations", 0) + 1
    }
```

### 步骤 3：路由决策

```python
def route_after_evaluate(state: ResearchGraphState):
    """基于评估结果决定下一步。"""
    # 检查是否达到最大迭代次数
    iterations = state.get("supervisor_iterations", 0)
    max_iter = state.get("max_supervisor_iterations", 3)
    
    if iterations >= max_iter:
        return "enough_coverage"
    
    # 检查是否需要补充
    gaps = state.get("coverage_gaps", "")
    if "需要补充" in gaps:
        return "needs_more"
    
    return "enough_coverage"
```

### 步骤 4：补充分析师节点

```python
delegate_more_prompt = """你是一名研究主管。现有研究对主题「{topic}」的分析还缺少以下角度：

{coverage_gaps}

请生成 1-2 位新的分析师来补充这些缺失的角度。每位分析师应专注于一个具体方面。

研究主题：{topic}"""

def delegate_more_analysts(state: ResearchGraphState):
    gaps = state.get("coverage_gaps", "")
    
    structured_llm = llm.with_structured_output(Perspectives, method="function_calling")
    result = structured_llm.invoke([
        SystemMessage(content=delegate_more_prompt.format(
            topic=state["topic"],
            coverage_gaps=gaps
        )),
        HumanMessage(content="生成补充分析师")
    ])
    
    return {
        "analysts": result.analysts,
        "sections": []  # 清空 sections，让新分析师重新写入
    }
```

### 步骤 5：图结构

```python
builder = StateGraph(ResearchGraphState)

# 添加节点
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_graph)
builder.add_node("evaluate_coverage", evaluate_coverage)
builder.add_node("delegate_more_analysts", delegate_more_analysts)
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

# 边
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges(
    "human_feedback",
    initiate_all_interviews,
    ["create_analysts", "conduct_interview"]
)
builder.add_edge("conduct_interview", "evaluate_coverage")

# 循环边
builder.add_conditional_edges(
    "evaluate_coverage",
    route_after_evaluate,
    {
        "needs_more": "delegate_more_analysts",
        "enough_coverage": "write_report"
    }
)
builder.add_edge("delegate_more_analysts", "conduct_interview")  # 循环

# Reduce 并行
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")

builder.add_edge(
    ["write_report", "write_introduction", "write_conclusion"],
    "finalize_report"
)
builder.add_edge("finalize_report", END)
```

---

## 关键注意事项

1. **边冲突问题**：`conduct_interview` 有两条出边 — 一条到 `evaluate_coverage`，一条到 `write_report/intro/conclusion`。LangGraph 允许多条边，但需要确保路由逻辑正确。更安全的做法是用一个条件路由替代：

```python
builder.add_conditional_edges(
    "conduct_interview",
    route_after_interview,
    ["evaluate_coverage", "write_report"]
)
```

2. **sections 累积 vs 清空**：`conduct_interview` 的 `sections` 使用 `operator.add` 自动累加。但 `delegate_more_analysts` 不应清空已有 sections，而是追加新的。

3. **`Send` 与循环的兼容**：在补充分析师时，需要再次用 `Send()` 分发新分析师到 `conduct_interview`。这与第一次分发逻辑相同，只是分析师列表不同。

4. **死循环保护**：`max_supervisor_iterations` 是硬保护。此外，`evaluate_coverage` 的 prompt 应该倾向于说"够了"而不是"不够"，避免 LLM 过于苛刻导致永远不满足。

5. **成本考量**：每次迭代增加一次 LLM 评估调用 + 可能的新分析师调用。建议 `max_supervisor_iterations = 2`（即最多 2 轮补充）。
