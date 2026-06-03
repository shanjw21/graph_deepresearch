# 项目经验：Deep Research 智能研究助手系统

## 项目概述

基于 LangGraph 框架从零实现的多智能体研究系统，通过模拟"分析师访谈专家"的工作模式，自动化完成深度研究并生成结构化报告。系统采用 Map-Reduce 架构，支持多分析师并行访谈、人机协同审核、多数据源检索等能力。

**技术栈**：Python / LangGraph / LangChain / OpenAI API / Pydantic / PostgreSQL / Tavily / Wikipedia / MCP

**项目周期**：2024.04 - 2024.05

---

## 核心架构

```
用户输入 → 生成分析师 → 人类审核 → Map(并行访谈) → Reduce(报告生成) → 最终报告
                                    ↓
                              ┌─────┴─────┐
                              │ 访谈子图    │
                              │ (循环+并行) │
                              └───────────┘
```

### 主图（Main Graph）- Map-Reduce 模式

- **Map 阶段**：`Send()` 动态分发 N 个分析师到并行访谈子图，每个子图独立运行
- **Reduce 阶段**：三路并行（write_report / write_introduction / write_conclusion），最后 finalize_report 组装

### 访谈子图（Interview Subgraph）- 循环+并行

- **并行搜索**：fan-out/fan-in 模式，search_web + search_baike 同时执行，结果通过 `operator.add` 自动合并
- **循环控制**：route_messages 实现条件路由，支持轮次限制和主动结束两种退出机制
- **6 节点设计**：ask_question → [search_web, search_baike] → answer_question → route_messages → save_interview → write_section

---

## 技术亮点

### 1. 状态管理设计

- **TypedDict + Annotated**：定义多层状态（GenerateAnalystsState / InterviewState / ResearchGraphState），通过 `Annotated[list, operator.add]` 实现自动累加
- **Pydantic 结构化输出**：Analyst / Perspectives / SearchQuery 等数据模型，配合 `with_structured_output()` 确保 LLM 输出格式可控

### 2. 人机协同（Human-in-the-Loop）

- **interrupt_before**：在 human_feedback 节点前拦截，保存状态到 checkpointer，进程可安全退出
- **checkpointer 状态持久化**：支持 MemorySaver（开发）/ PostgresSaver（生产），每个节点执行后自动保存快照
- **update_state 注入反馈**：人类审核后通过 `graph.update_state()` 修改状态，再 `invoke(None, config)` 恢复执行

### 3. 动态并行（Send）

- `Send("conduct_interview", {...})` 动态创建 N 个并行子图实例
- 每个子图独立运行，互不等待，结果通过 `operator.add` 自动累加到主图状态

### 4. 生产级 Checkpointer

- **PostgreSQL 持久化**：`PostgresSaver` 自动创建 checkpoints 表，JSONB 存储完整状态快照
- **Migration 管理**：通过 Alembic 追踪表结构变更，支持版本控制和回滚
- **无状态服务架构**：暂停和恢复可在不同进程/HTTP 请求中，通过 thread_id 关联

---

## 优化点（已设计，部分已实现）

### P0: think_tool 反思停顿

- **问题**：固定轮数访谈，无法动态判断信息是否充分
- **方案**：在 answer_question 后插入 reflect 节点，LLM 做元认知评估，动态决定继续或结束
- **效果**：访谈质量↑，无效 token↓（简单问题 1-2 轮结束，复杂问题可深入）

### P1: 搜索结果 URL 去重

- **问题**：多个分析师搜索相似主题时，容易拿到相同网页
- **方案**：维护 seen_urls 集合，过滤重复 URL
- **效果**：减少重复内容，提升上下文质量

### P2: 网页内容摘要

- **问题**：原始搜索结果直接拼接进 prompt，可能非常长
- **方案**：用小模型（GPT-4o-mini）压缩网页内容，保留关键信息
- **效果**：上下文质量↑，token 成本↓

### P3: 报告语言自适应

- **问题**：硬编码"全部使用中文"，无法支持多语言
- **方案**：prompt 中加入语言检测指令，根据用户输入自动切换输出语言
- **效果**：支持中/英文等多语言报告生成

### P4: MCP 本地文件搜索

- **问题**：只有互联网搜索，无法利用本地文档（行业报告、论文 PDF、内部资料）
- **方案**：集成 MCP（Model Context Protocol）文件系统服务器，新增 search_local_files 异步节点
- **效果**：三路并行搜索（Web + 百科 + 本地），数据源更丰富

### 进阶: Supervisor 迭代委派

- **问题**：一次性 Send 分发，没有"回头看是否需要补充"的机制
- **方案**：新增 evaluate_coverage 评估节点 + delegate_more_analysts 补充节点，实现循环委派
- **效果**：动态补充缺失角度，报告覆盖更全面

---

## 面试话术

### 项目介绍（1 分钟）

> "这是一个基于 LangGraph 的多智能体研究系统。核心思路是模拟'分析师访谈专家'的工作流程：先用 LLM 生成 N 个不同视角的分析师，每个分析师独立进行多轮访谈（提问→搜索→回答→追问），最后将所有访谈结果整合成一份结构化报告。
>
> 技术上用了 LangGraph 的三个核心能力：**子图复用**（访谈子图被 N 个分析师并行调用）、**人机协同**（interrupt_before + checkpointer 实现暂停/恢复）、**动态分发**（Send 实现 Map-Reduce 并行）。"

### 技术深度（被追问时）

> "**状态管理**：用 TypedDict 定义多层状态，`Annotated[list, operator.add]` 实现自动累加，避免手动合并。
>
> **并行搜索**：fan-out/fan-in 模式，两个搜索节点同时执行，LangGraph 自动等两个都完成后再执行下一个节点。
>
> **循环控制**：route_messages 实现条件路由，两种退出机制——硬限制（轮次上限）和软限制（LLM 主动结束）。
>
> **生产部署**：checkpointer 从 MemorySaver 升级到 PostgresSaver，用 Alembic 管理表结构变更，支持无状态 HTTP 服务架构。"

### 优化思路（展示思考深度）

> "系统有几个优化方向：
>
> 1. **访谈质量**：加入 think_tool 反思节点，让 LLM 动态判断信息是否充分，而不是固定轮数
> 2. **搜索质量**：URL 去重 + 网页摘要，减少重复和冗余信息
> 3. **数据源扩展**：通过 MCP 协议集成本地文件搜索，三路并行（Web + 百科 + 本地）
> 4. **架构升级**：从一次性 Map-Reduce 改为 Supervisor 迭代委派，动态补充缺失角度"

---

## 项目成果

- 从零实现完整 LangGraph 多智能体系统，涵盖子图、并行、循环、人机协同等核心模式
- 设计 5 个优化方案（P0-P4 + 进阶），覆盖质量、成本、数据源、架构四个维度
- 积累 LangGraph 状态管理、checkpointer 持久化、MCP 协议集成等生产级经验

---

*适用岗位：AI Agent 开发工程师 / LLM 应用工程师 / 智能体架构师*
