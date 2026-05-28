# P4 优化：MCP（Model Context Protocol）文件搜索集成

## 背景知识

### 什么是 MCP？

MCP（Model Context Protocol）是一个**标准协议**，让 LLM 应用可以统一访问各种外部工具和数据源。类比：

- **USB** 是硬件接口的标准协议 → 任何 USB 设备插到任何电脑都能用
- **MCP** 是 AI 工具接口的标准协议 → 任何 MCP 服务器连到任何 AI 应用都能用

官方教程用 MCP 实现了**本地文件系统访问**，让研究智能体可以搜索、读取本地文档。

### 你的系统当前有什么

`phase5_day9_skeleton.py` 中的研究工具是：
- **Tavily API**：互联网搜索
- **WikipediaLoader**：维基百科搜索

**缺少**：本地文档搜索能力。如果你有行业报告、论文 PDF、内部文档等本地文件，当前系统无法利用。

### MCP 集成后的能力

| 能力 | Tavily | Wikipedia | MCP 文件系统 |
|------|--------|-----------|-------------|
| 数据来源 | 互联网 | 维基百科 | 本地文件 |
| 搜索方式 | 关键词搜索 | 关键词搜索 | 文件搜索 + 读取 |
| 适用场景 | 最新公开信息 | 百科知识 | 私有文档、内部资料 |
| 需要网络 | 是 | 是 | 否（本地） |
| 成本 | 按次收费 | 免费 | 免费（仅需 LLM token） |

### MCP 架构核心概念

```
┌─────────────┐     stdio/HTTP     ┌─────────────────┐
│  LLM Agent  │ ←────────────────→ │  MCP Server     │
│  (Python)   │    JSON-RPC协议    │  (Node.js等)    │
└─────────────┘                     └─────────────────┘
     │                                     │
     │ MultiServerMCPClient               │ 文件系统/数据库/API
     │ (langchain-mcp-adapters)           │
     ▼                                     ▼
  工具调用                              实际执行
```

**关键要点**：
1. MCP 服务器是**独立进程**（如 `npx @modelcontextprotocol/server-filesystem`）
2. 通过 `MultiServerMCPClient` 连接和获取工具
3. MCP 工具**必须用 async** 调用（因为是进程间通信）
4. 工具列表是动态的——服务器决定提供哪些工具

### 两种 Transport 模式

| 模式 | 适用场景 | 配置 |
|------|----------|------|
| **stdio**（本地） | 本地文件系统、本地数据库 | `"command": "npx"`, `"args": [...]` |
| **http**（远程） | 远程 API、SaaS 服务 | `"url": "https://..."`, `"transport": "http"` |

官方教程用 `stdio` 模式启动本地文件系统服务器，允许 Agent 读取指定目录下的文件。

---

## 实现步骤

### 步骤 1：安装依赖

```bash
pip install langchain-mcp-adapters
# MCP filesystem server 需要 npx (Node.js)
# 如果没有 Node.js，可以用 brew install node
```

### 步骤 2：配置 MCP 客户端

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

# 指定要暴露给 Agent 的本地目录
research_docs_path = "/path/to/your/research/docs"

mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            research_docs_path
        ],
        "transport": "stdio"
    }
}

# 懒加载客户端（避免在图编译时就启动子进程）
_mcp_client = None

def get_mcp_client():
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MultiServerMCPClient(mcp_config)
    return _mcp_client
```

### 步骤 3：修改搜索节点 — 混合搜索策略

原来的 `search_web` 和 `search_baike` 保留，新增 `search_local_files` 节点：

```python
async def search_local_files(state: InterviewState):
    """使用 MCP 文件系统搜索本地研究文档。"""
    seen_urls = extract_urls_from_context(state.get("context", []))

    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    tools = mcp_tools + [think_tool]

    model_with_tools = llm.bind_tools(tools)

    # 让 LLM 决定搜索什么
    search_query = llm.with_structured_output(SearchQuery, method="function_calling").invoke(
        [search_instructions] + state["messages"]
    )

    # 让 LLM 通过 MCP 工具搜索文件
    # 注意：这里需要异步执行 MCP 工具
    ...
```

### 步骤 4：LLM 调用和工具节点需要改为 async

因为 MCP 工具必须异步调用，所以 `llm_call` 和 `tool_node` 都需要改为 `async def`：

```python
async def llm_call(state: InterviewState):
    """LLM 调用，绑定 MCP 工具。"""
    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    tools = mcp_tools + [think_tool]
    model_with_tools = llm.bind_tools(tools)
    ...

async def tool_node(state: InterviewState):
    """工具节点，异步执行 MCP 工具。"""
    tool_calls = state["researcher_messages"][-1].tool_calls

    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    tools = mcp_tools + [think_tool]
    tools_by_name = {tool.name: tool for tool in tools}

    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        if tool_call["name"] == "think_tool":
            observation = tool.invoke(tool_call["args"])
        else:
            observation = await tool.ainvoke(tool_call["args"])
        observations.append(observation)
    ...
```

### 步骤 5：提示词适配

新增 `research_agent_prompt_with_mcp`，告诉 Agent 它现在可以访问本地文件：

```python
research_agent_prompt_with_mcp = """你是一名研究助理，可以通过本地文件系统和本地搜索来收集信息。

你有以下工具：
- **MCP 文件系统工具**：搜索和读取本地研究文档
- **Tavily 搜索**：互联网搜索
- **Wikipedia 搜索**：维基百科搜索
- **think_tool**：反思和规划

策略：
1. 先用 MCP 文件工具查看本地有哪些文档
2. 搜索相关文件内容
3. 如果本地资料不足，再用 Tavily/Wikipedia 补充
"""
```

---

## 在你的系统中的最佳集成方式

**推荐方案**：不是替换 Tavily/Wikipedia，而是**并行使用**三种数据源：

```
ask_question → search_web (并行)
             → search_baike (并行)
             → search_local_files (MCP, 并行)
             → answer_question (综合三路结果)
```

这样每个分析师可以同时从互联网、百科、本地文档三个渠道获取信息。

---

## 关键注意事项

1. **Node.js 依赖**：MCP filesystem server 需要 `npx`，确保已安装 Node.js
2. **目录权限**：MCP 服务器只能访问配置中指定的目录，不会暴露整个文件系统
3. **异步要求**：所有涉及 MCP 的节点必须用 `async def`，图编译时 LangGraph 会自动处理
4. **懒加载客户端**：用 `get_mcp_client()` 函数懒加载，避免在导入时就启动子进程
5. **远程 MCP 服务器**：除了本地文件系统，还可以连接远程 MCP 服务（如 Asana、Slack、GitHub 等），配置方式类似但用 `"transport": "http"`
6. **成本**：MCP 工具本身免费（无 API 费用），但 LLM 调用工具的 token 照常计费
