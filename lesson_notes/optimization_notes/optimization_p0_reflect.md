# P0 优化：think_tool 反思停顿节点

## 背景知识

### 什么是反思停顿（Reflective Pause）？

反思停顿是一种 Agent 设计模式，让 LLM 在执行动作后停下来做一次**元认知评估**（metacognitive assessment），而不是盲目地继续下一步。

在 Deep Research 场景中：
- **没有反思**：分析师按固定轮数访谈，即使第 1 轮就问完了所有关键信息，还要硬凑第 2、3 轮
- **有反思**：分析师在每轮后判断"信息够了吗"，够了就结束，不够就继续

### 官方教程的实现

官方 `research_agent.py` 中的 `think_tool` 是一个"哑工具"（dumb tool）——它不做任何实际工作，只返回一条确认消息：

```python
@tool
def think_tool(reflection: str) -> str:
    return f"Reflection recorded: {reflection}"
```

它的价值不在于返回值，而在于**强制 LLM 在调用时结构化自己的思考**。LLM 必须填写 `reflection` 参数，这个过程本身就促使它评估当前状态。

### LangGraph 中的条件路由

反思节点的核心是利用 `conditional_edges` 做动态路由：

```
answer_question → reflect → (继续? ask_question : 结束? save_interview)
```

这比固定轮数的优势在于：
- 简单问题：1-2 轮就结束，省 token
- 复杂问题：可以深入追问，不浪费已有信息
- 质量可控：LLM 自己判断何时停止

---

## 实现步骤

### 步骤 1：定义反思提示词

```python
reflect_instructions = """你是一名分析师，刚刚完成一轮专家访谈。

请评估当前进展，回答以下问题：
1. 我获得了哪些关键信息？
2. 还缺少什么重要信息？
3. 我已经有了足够的洞见来写报告小节吗？
4. 我应该继续追问还是结束访谈？

如果信息已充分，请以「信息已充分，可以结束访谈」结尾。
如果还需要更多信息，请以「需要继续追问」结尾。

保持反思简洁（约 100 字）。"""
```

### 步骤 2：添加 reflect 节点

```python
def reflect(state: InterviewState):
    """分析师在每轮访谈后进行元认知评估。"""
    analyst = state["analyst"]
    messages = state["messages"]
    system_prompt = reflect_instructions
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="请评估当前访谈进展。")
    ])
    
    # 判断是否应该结束
    should_end = any(keyword in response.content for keyword in [
        "信息已充分", "可以结束", "足够", "sufficient", "enough"
    ])
    
    return {
        "messages": [AIMessage(content=response.content, name="reflection")],
        "should_end_reflection": should_end
    }
```

### 步骤 3：修改路由逻辑

```python
def route_after_reflect(state: InterviewState):
    """基于反思决定继续还是结束。"""
    # 检查反思节点的决定
    messages = state["messages"]
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.name == "reflection":
            if any(keyword in m.content for keyword in [
                "信息已充分", "可以结束", "足够", "sufficient", "enough"
            ]):
                return "save_interview"
    
    # 兜底：如果反思没有明确结论，检查最大轮数
    max_num_turns = state.get("max_num_turns", 2)
    expert_responses = sum(1 for m in messages if isinstance(m, AIMessage) and m.name == "expert")
    if expert_responses >= max_num_turns:
        return "save_interview"
    
    return "ask_question"
```

### 步骤 4：更新子图结构

```python
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_baike", search_baike)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("reflect", reflect)           # 新增
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_baike")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_baike", "answer_question")
interview_builder.add_edge("answer_question", "reflect")  # 改为先到反思
interview_builder.add_conditional_edges(
    "reflect",
    route_after_reflect,
    ["ask_question", "save_interview"]
)
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)
```

### 步骤 5：状态定义需要加 `should_end_reflection` 字段

```python
class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list
    should_end_reflection: bool = False  # 新增
```

---

## 关键注意事项

1. **关键词匹配不完美**：LLM 可能用不同措辞表达"结束"。可以用更可靠的方式：让 LLM 输出结构化 JSON（`with_structured_output`），直接返回 `should_continue: bool`。

2. **兜底保护**：仍然保留 `max_num_turns` 作为安全上限，防止 LLM 无限循环。

3. **Token 成本**：每次反思增加一次 LLM 调用。如果 `max_num_turns` 很小（2-3），增加的开销约 30-50%。可以通过用小模型（如 gpt-4o-mini）做反思来降低成本。
