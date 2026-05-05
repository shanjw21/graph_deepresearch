# AI 智能体系统 — 面试项目学习

> 基于 LangGraph 构建 Map-Reduce 多智能体深度研究系统，同时用 Langfuse 建立完整的在线/离线评估体系和安全防护机制。

## 项目概述

两个项目联动作为 AI 智能体方向面试项目：

| 项目 | 技术栈 | 核心能力 |
|------|--------|---------|
| **Deep Research 多智能体系统** | LangGraph + GPT-4o + Tavily | Map-Reduce 并行访谈 + 自动报告生成 |
| **Langfuse 质量评估与安全监控** | Langfuse + LLM Guard | 在线追踪 / 离线评估 / LLM-as-a-Judge / 安全防护 |

## 学习路线（7 阶段 / 12 天）

```
阶段一 → 阶段二 → 阶段三 → 阶段四 → 阶段五 → 阶段六 → 阶段七
基础砖块   单线工作流  条件路由   访谈子图   Map-Reduce  质量评估   安全防护
(2天)     (2天)    (1天)    (2天)★    (2天)★    (2天)★    (1天)
```

★ 标记为面试核心

## 目录结构

```
lesson_notes/          # 12 天课程讲义 + 骨架代码
├── phase1_day1_*      # Pydantic 结构化输出
├── phase1_day2_*      # TypedDict 状态模型
├── phase2_day3_*      # StateGraph 单线工作流
├── phase2_day4_*      # Langfuse 追踪接入
├── phase3_day5_*      # 条件路由 + 人机协同
├── phase4_day6_*      # 访谈子图（上）
├── phase4_day7_*      # 访谈子图（下）
├── phase5_day8_*      # Map-Reduce — Map 并行
├── phase5_day9_*      # Map-Reduce — Reduce 聚合
├── phase6_day10_*     # 在线评估 — Langfuse 追踪与评分
├── phase6_day11_*     # 离线评估 + LLM-as-a-Judge
└── phase7_day12_*     # 安全防护 — 暴力检测/PII脱敏/注入防护

codes/                 # 基础实验代码
├── structured_output.py
├── s02_state_rebuild.py
└── s04_langfuse.py

INTERVIEW_LEARNING_GUIDE.md   # 完整学习指南（含面试话术）
```

## 技术栈

- **工作流编排**：LangGraph StateGraph
- **LLM**：OpenAI GPT-4o (ChatOpenAI)
- **搜索**：Tavily + Wikipedia
- **可观测性**：Langfuse SDK v4
- **评估**：Langfuse Dataset + Experiment SDK + LLM-as-a-Judge
- **安全防护**：LLM Guard (BanTopics / Anonymize / PromptInjection)

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/shanjw21/graph_deepresearch.git

# 安装依赖
pip install langchain-openai langgraph langfuse llm-guard python-dotenv

# 配置 .env
cp .env.example .env
# 填入 BASE_URL, API_KEY, MODEL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
```

## 学习方式

每天的学习流程：
1. 读 `lesson_notes/phaseX_dayY_*.md` 讲义理解概念
2. 在 `lesson_notes/phaseX_dayY_skeleton.py` 中实现 TODO 部分
3. 运行验证，有疑问时追问
4. 完成后 review 代码

## 参考

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [Langfuse 文档](https://langfuse.com/docs)
- [LLM Guard 文档](https://llm-guard.com/)
