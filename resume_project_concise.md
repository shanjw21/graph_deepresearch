# Deep Research 智能研究助手系统

**技术栈**：Python / LangGraph / LangChain / OpenAI / Pydantic / PostgreSQL / Tavily / MCP

## 项目简介

基于 LangGraph 从零实现的多智能体研究系统，通过"分析师访谈专家"模式自动化完成深度研究并生成结构化报告。采用 Map-Reduce 架构，支持多分析师并行访谈、人机协同审核、多数据源检索。

## 核心工作

### 1. 多智能体架构设计

- 设计主图 + 访谈子图的双层架构，主图控制整体流程，子图封装访谈逻辑
- 实现 Map-Reduce 模式：`Send()` 动态分发 N 个分析师并行访谈，结果通过 `operator.add` 自动累加
- 访谈子图融合三种模式：**并行搜索**（fan-out/fan-in）、**循环控制**（条件路由）、**子图复用**

### 2. 状态管理与人机协同

- 使用 `TypedDict + Annotated[list, operator.add]` 定义多层状态，实现自动累加
- Pydantic 结构化输出（Analyst / SearchQuery），配合 `with_structured_output()` 确保 LLM 输出可控
- `interrupt_before` + checkpointer 实现暂停/恢复，支持 `update_state` 注入人类反馈

### 3. 生产级持久化

- PostgreSQL Checkpointer：自动创建 checkpoints 表，JSONB 存储状态快照
- Alembic Migration 管理表结构变更，支持版本控制和回滚
- 无状态服务架构：暂停/恢复可在不同进程，通过 thread_id 关联

### 4. 系统优化（设计 + 部分实现）

| 优化点 | 方案 | 效果 |
|--------|------|------|
| P0 反思停顿 | 插入 reflect 节点，LLM 动态判断访谈结束时机 | 访谈质量↑，token↓ |
| P1 URL 去重 | seen_urls 集合过滤重复搜索结果 | 重复内容↓ |
| P2 网页摘要 | 小模型压缩搜索结果 | 上下文质量↑ |
| P3 语言自适应 | prompt 语言检测指令 | 多语言支持 |
| P4 MCP 集成 | 本地文件搜索，三路并行 | 数据源扩展 |
| 进阶 迭代委派 | Supervisor 循环 + 覆盖度评估 | 报告完整性↑ |

## 技术亮点

- **子图复用**：访谈子图被 N 个分析师并行调用，代码复用率高
- **动态并行**：`Send()` 运行时决定并行数量，非编译时硬编码
- **循环+条件路由**：route_messages 实现双退出机制（轮次硬限制 + LLM 软限制）
- **checkpointer 状态机**：每个节点执行后自动保存快照，支持任意时刻恢复

## 个人职责

独立完成系统设计、核心代码实现、优化方案设计与验证。

---

*适用岗位：AI Agent 开发 / LLM 应用工程师 / 智能体架构*
