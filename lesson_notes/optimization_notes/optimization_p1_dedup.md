# P1 优化：搜索结果 URL 去重

## 背景知识

### 为什么要去重？

在多智能体 Deep Research 系统中，多个分析师独立搜索时经常会访问相同的网页。例如：
- 分析师 A 搜索 "AI 医疗 诊断 准确率" → 找到 `https://example.com/ai-diagnosis`
- 分析师 B 搜索 "医疗影像 AI 临床数据" → 也找到 `https://example.com/ai-diagnosis`

如果不做去重，同一篇内容会被多次塞进 prompt，造成：
- **上下文浪费**：相同的 2000 字内容重复出现
- **信息偏重**：重复内容在 LLM 眼中权重翻倍
- **Token 成本增加**：同样的内容付两次费

### 官方教程的实现

官方 `utils.py` 中的去重逻辑非常简单：

```python
def deduplicate_search_results(search_results: List[dict]) -> dict:
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result
    return unique_results
```

核心思想：用 **URL 作为 dict 的 key**，天然去重。

### 在你的系统中的位置

去重可以在两个层面做：

1. **搜索层去重**：在 `search_web` / `search_baike` 节点内，每次搜索前检查 URL 是否已出现过
2. **全局去重**：在主图 state 中维护一个 `seen_urls` 集合，所有分析师共享

**推荐做法**：在 `InterviewState` 中用 `context` 字段附带 URL 信息，然后在搜索节点中解析已有 context 提取 URL，过滤重复。

但更简单且有效的做法是：**在每个搜索节点中维护一个跨调用的 seen_urls 集合**，或者在 state 中增加一个 `seen_urls` 字段。

---

## 实现步骤

### 步骤 1：状态定义增加 seen_urls

```python
class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list
    seen_urls: Annotated[list, operator.add]  # 新增：已见过的 URL 列表
```

### 步骤 2：提取 URL 的工具函数

```python
def extract_urls_from_context(context_list: list) -> set:
    """从 context 列表中提取所有 URL。"""
    import re
    urls = set()
    for ctx in context_list:
        # 匹配 href="..." 和 source="..." 属性
        found = re.findall(r'href="([^"]+)"', ctx)
        found += re.findall(r'source="([^"]+)"', ctx)
        urls.update(found)
    return urls
```

### 步骤 3：修改搜索节点

```python
def search_web(state: InterviewState):
    # 获取已见过的 URL
    seen_urls = set()
    for ctx in state.get("context", []):
        seen_urls.update(extract_urls_from_context([ctx]))

    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    try:
        search_docs = tavily_search.invoke(search_query.search_query)

        # 去重过滤
        new_docs = []
        new_urls = []
        for doc in search_docs:
            url = doc.get("url", "")
            if url and url not in seen_urls:
                new_docs.append(doc)
                new_urls.append(url)

        formatted_docs = "\n\n---\n\n".join([
            f'<Document href="{doc["url"]}" />\n{doc["content"]}\n</Document>'
            for doc in new_docs
        ])

        if not new_docs:
            formatted_docs = f"<Document />搜索结果的URL均已见过，查询: {search_query.search_query}\n</Document>"

    except Exception as e:
        formatted_docs = f"<Document />Web搜索暂不可用，查询: {search_query.search_query}\n</Document>"

    return {
        "context": [formatted_docs],
        "seen_urls": new_urls if new_docs else []
    }
```

### 步骤 4：search_baike 同理

```python
def search_baike(state: InterviewState):
    seen_urls = set()
    for ctx in state.get("context", []):
        seen_urls.update(extract_urls_from_context([ctx]))

    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    try:
        search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()

        new_docs = []
        new_urls = []
        for doc in search_docs:
            url = doc.metadata.get("source", "")
            if url and url not in seen_urls:
                new_docs.append(doc)
                new_urls.append(url)

        formatted_docs = "\n\n---\n\n".join([
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/> \n{doc.page_content}\n</Document>'
            for doc in new_docs
        ])

        if not new_docs:
            formatted_docs = f"<Document />百科搜索结果的URL均已见过，查询: {search_query.search_query}\n</Document>"

    except Exception as e:
        formatted_docs = f"<Document />百科搜索暂不可用，查询: {search_query.search_query}\n</Document>"

    return {
        "context": [formatted_docs],
        "seen_urls": new_urls if new_docs else []
    }
```

---

## 关键注意事项

1. **跨分析师去重**：当前的 `seen_urls` 在 `InterviewState` 中，只在单个分析师子图内共享。要实现跨分析师去重，需要在 `ResearchGraphState` 中维护全局 `seen_urls`，但这需要在子图和主图之间传递状态，较复杂。**单子图内去重**已经是性价比最高的改进。

2. **URL 标准化**：有些 URL 只是看起来不同（如末尾 `/` 差异、查询参数差异）。如果要做精确去重，可以先做 URL 标准化（去掉查询参数、normalize 路径）。但一般场景下字符串比较就够了。

3. **降级处理**：当所有搜索结果都已见过时，不要直接返回空，而是给一个提示信息，让 LLM 知道"这些 URL 已经看过了"，LLM 可能会换搜索词。
