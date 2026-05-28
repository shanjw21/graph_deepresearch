# P2 优化：网页内容摘要

## 背景知识

### 为什么需要摘要？

在 Deep Research 中，Tavily 和 Wikipedia 返回的内容通常非常长：
- Tavily 单条结果可能包含 2000-5000 字的网页原始内容
- 一次搜索 3 条结果 = 最多 15000 字
- 多轮搜索 + 百科搜索 = 上下文可能超过 30000 字

直接把这些原始内容塞进 prompt 的问题：
- **LLM 注意力衰减**：上下文越长，中间部分的信息越容易被忽略（"lost in the middle" 现象）
- **Token 成本高**：长上下文 = 更多输入 token
- **回答质量下降**：过多的冗余信息会干扰 LLM 提取关键信息

### 官方教程的实现

官方用 GPT-4.1-mini 做摘要，通过 `with_structured_output(Summary)` 返回结构化输出：

```python
class Summary(BaseModel):
    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")
```

摘要策略：
- 保留主要论点、关键数据、重要引用
- 压缩到原始内容的 25-30%
- 保留来源 URL 和标题

### 在你的系统中的位置

摘要应该在搜索节点返回结果之前做。流程：

```
搜索 API → 原始结果 → 摘要 → 格式化 → 存入 context
```

### 要不要用独立小模型？

官方用 GPT-4.1-mini，成本低（约 $0.4/1M tokens）。如果你已经在用 OpenAI 的模型，直接复用同一个 `llm` 就行。如果追求更低成本，可以用更小的模型（如 gpt-4o-mini）。

---

## 实现步骤

### 步骤 1：定义摘要提示词

```python
summarize_instructions = """你是一名研究助手，需要将网页内容压缩为简短摘要。

请执行以下操作：
1. 提取主要论点（1-2 句）
2. 保留关键数据、数字和事实
3. 保留最多 2 条重要引用
4. 去除重复、广告、导航等无关内容

输出格式：
- 摘要：[1-2 句话]
- 关键信息：[3-5 个要点，用 • 分隔]

原始内容：
{content}"""
```

### 步骤 2：添加摘要函数

```python
def summarize_content(content: str, max_length: int = 300) -> str:
    """用 LLM 将长网页内容压缩为简短摘要。"""
    # 如果内容本身就很短，不摘要
    if len(content) < max_length:
        return content
    
    system_prompt = summarize_instructions.format(content=content[:4000])  # 截断防止太长
    summary = llm.invoke([
        SystemMessage(content="你是一名研究助手，需要将网页内容压缩为简短摘要。提取主要论点、关键数据和事实。"),
        HumanMessage(content=f"请将以下内容压缩为{max_length}字以内的摘要：\n{system_prompt}")
    ])
    return summary.content[:max_length]
```

### 步骤 3：在搜索节点中调用

```python
def search_web(state: InterviewState):
    seen_urls = extract_urls_from_context(state.get("context", []))

    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    try:
        search_docs = tavily_search.invoke(search_query.search_query)

        new_docs = []
        new_urls = []
        for doc in search_docs:
            url = doc.get("url", "") if isinstance(doc, dict) else ""
            if url and url not in seen_urls:
                new_docs.append(doc)
                new_urls.append(url)

        # P2 改动：对每条内容做摘要
        summarized_docs = []
        for doc in new_docs:
            title = doc.get("title", "未知来源")
            content = doc.get("content", "")
            if len(content) > 500:  # 只对长内容做摘要
                content = summarize_content(content, max_length=300)
            summarized_docs.append(f'<Document href="{doc.get("url", "")}" title="{title}">\n{content}\n</Document>')

        formatted_docs = "\n\n---\n\n".join(summarized_docs)

    except Exception as e:
        formatted_docs = f"<Document />Web搜索暂不可用</Document>"

    return {"context": [formatted_docs], "seen_urls": new_urls}
```

---

## 关键注意事项

1. **摘要 vs 原文的权衡**：摘要会丢失细节。官方教程的做法是保留 `raw_notes`（原始内容）和 `compressed_research`（摘要），在不同阶段用不同的版本。你的系统中 `write_section` 阶段其实可以用摘要版本（省 token），而最终报告生成时用原始版本（保留细节）。

2. **截断保护**：摘要函数的输入应该截断到合理长度（如 4000 字），否则极长网页会浪费 token。

3. **摘要长度**：建议 200-400 字。太短会丢失关键信息，太长就失去了摘要的意义。

4. **成本计算**：每次搜索 3 条结果，每条摘要约 300 字 = 约 900 字输入 + 300 字输出 = 约 1200 tokens。如果每轮访谈 2 次搜索，每个分析师约 2400 tokens，3 个分析师约 7200 tokens。成本增加不多。
