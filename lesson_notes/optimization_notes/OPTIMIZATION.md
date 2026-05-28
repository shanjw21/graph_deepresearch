# Deep Research 优化笔记

> 基于 `deep_research_from_scratch`（LangGraph 官方教程）与 `lesson_notes/phase5_day9_skeleton.py`（分析师访谈 Map-Reduce）的对比分析。

---

## 优化点总览

| 优先级 | 优化点 | 工作量 | 效果 |
|--------|--------|--------|------|
| **P0** | `think_tool` 反思停顿节点 | 加 1 个节点 + 1 个提示词 | 访谈质量↑，无效 token↓ |
| **P1** | 搜索结果 URL 去重 | `search_web/baike` 中加 URL set | 重复内容↓ |
| **P2** | 网页内容摘要 | 调用小模型摘要网页 | 上下文质量↑ |
| **P3** | 最终报告语言自适应 | prompt 改一句话 | 多语言支持 |
| **P4** | MCP 本地文件搜索集成 | 新增异步节点 + MCP 配置 | 数据源↑（本地文档） |
| **进阶** | Supervisor 迭代委派 | 架构改动 | 灵活性↑，但复杂度高 |

---

## P0: think_tool 反思停顿节点

### 背景
官方教程中，每个研究智能体在搜索后必须调用 `think_tool`，回答四个问题：找到了什么、还缺什么、证据够了吗、继续还是停止。这相当于让 AI 做一次**战略判断**，避免盲目搜索。

### 当前问题
`phase5_day9_skeleton.py` 中访谈轮数固定为 `max_num_turns`，分析师无法判断"信息是否够了"就结束。

### 改进方案
在访谈子图中插入 `reflect` 节点，让分析师在每轮问答后做一次元认知评估，动态决定是否继续访谈。

**详见**: `optimization_p0_reflect.md` + `optimization_p0_reflect.py`

---

## P1: 搜索结果 URL 去重

### 背景
官方教程的 `utils.py` 中有 `deduplicate_search_results()` 函数，通过 URL 集合去重。多个分析师搜索相似主题时，很容易拿到相同网页。

### 当前问题
`search_web` 和 `search_baike` 直接返回原始结果，不做去重。

### 改进方案
维护一个 `seen_urls` 集合，过滤掉已经出现过的 URL。

**详见**: `optimization_p1_dedup.md` + `optimization_p1_dedup.py`

---

## P2: 网页内容摘要

### 背景
官方教程用 `summarize_webpage_content()` 将长网页内容用 GPT-4.1-mini 压缩，保留关键信息同时节省上下文。

### 当前问题
Tavily 和 Wikipedia 的原始内容直接拼接进 prompt，可能非常长。

### 改进方案
在搜索结果格式化后，加一层小模型摘要。

**详见**: `optimization_p2_summarize.md` + `optimization_p2_summarize.py`

---

## P3: 最终报告语言自适应

### 背景
官方教程在 `final_report_generation_prompt` 中要求模型检测用户输入语言并用相同语言输出报告。

### 当前问题
`phase5_day9_skeleton.py` 中硬编码"全部使用中文"。

### 改进方案
在 prompt 模板中加入语言检测指令，根据用户输入自动切换输出语言。

**详见**: `optimization_p3_language.md` + `optimization_p3_language.py`

---

## P4: MCP 本地文件搜索集成

### 背景
官方教程用 MCP（Model Context Protocol）实现了本地文件系统访问，让研究智能体可以搜索、读取本地文档。MCP 是一个标准协议，类比 USB —— 任何 MCP 服务器连到任何 AI 应用都能用。

### 当前问题
`phase5_day9_skeleton.py` 只有 Tavily（互联网搜索）和 Wikipedia（百科搜索），无法利用本地文档（行业报告、论文 PDF、内部资料等）。

### 改进方案
1. 用 `langchain-mcp-adapters` 连接本地文件系统 MCP 服务器
2. 新增 `search_local_files` 异步节点，与 `search_web` + `search_baike` 三路并行
3. 所有涉及 MCP 的节点必须 `async def`（MCP 协议要求异步进程间通信）

**详见**: `optimization_p4_mcp.md` + `optimization_p4_mcp.py`

---

## 进阶: Supervisor 迭代委派

### 背景
官方教程的 Supervisor 可以循环多轮 — 第一轮派 2 个子智能体，拿到结果后发现还缺信息，再派第 3 个。而 Map-Reduce 是一次性分发所有分析师。

### 当前问题
`Send()` 一次性分发所有分析师，没有"回头看是否需要补充"的机制。

### 改进方案
将一次性 `Send()` 改为循环委派模式，需要重构主图结构。

**详见**: `optimization_deferred_supervisor.md` + `optimization_deferred_supervisor.py`

---

*生成日期: 2026-05-28*
