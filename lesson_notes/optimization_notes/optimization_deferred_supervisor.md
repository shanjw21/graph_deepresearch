# 暂缓优化：Supervisor 迭代委派

## 背景知识

### 什么是 Supervisor 迭代委派？

在 `deep_research_from_scratch` 中，Supervisor 模式的核心是**动态决策**：

```
Supervisor 收到研究简报
    → 思考：需要研究哪些子主题？
    → 委派 1-3 个子智能体并行研究
    → 等待结果
    → 思考：结果够了吗？还缺什么？
    → 如果不够：再委派新的子智能体
    → 如果够了：标记 ResearchComplete
```

这是一个**循环**过程，Supervisor 可以根据中间结果动态调整研究方向。

### 你的系统当前的做法

`phase5_day9_skeleton.py` 中用的是 `Send()` 一次性分发：

```
create_analysts → 生成 3 个分析师
    → Send() 一次性分发所有分析师到 conduct_interview
    → 所有访谈并行执行
    → 3 个 Reduce 节点并行写报告
    → finalize_report 组装
```

这是一个**DAG（有向无环图）**，一旦分发就不能回头。

### 两者的权衡

| 维度 | Map-Reduce（当前） | Supervisor 迭代（官方） |
|------|-------------------|------------------------|
| 灵活性 | 低，一次性分发 | 高，可动态调整 |
| 复杂度 | 低，图结构清晰 | 高，需要循环控制 |
| Token 成本 | 可预测 | 不确定，取决于迭代次数 |
| 适用场景 | 主题明确的报告 | 开放探索性研究 |
| 调试难度 | 低 | 高（循环可能无限） |

---

## 为什么暂缓

1. **架构改动大**：需要将 `Send()` 的 Map-Reduce 改为 Supervisor 循环，涉及主图结构重构
2. **收益不确定**：对于"AI 医疗"这类主题明确的研究，一次性分发 3 个不同角度的分析师已经覆盖了大部分需求
3. **调试成本高**：循环委派可能导致无限循环，需要加更多安全保护

---

## 如果未来要实现

### 大致思路

将主图从 DAG 改为循环：

```python
def supervisor_evaluate(state: ResearchGraphState):
    """Supervisor 评估当前进展，决定是否需要更多研究。"""
    sections = state.get("sections", [])
    
    # 如果已经有足够多的小节，进入 Reduce
    if len(sections) >= state["max_analysts"]:
        return "reduce_phase"
    
    # 否则，生成新的分析师并继续 Map
    return "map_phase"


builder.add_conditional_edges(
    "conduct_interview",
    supervisor_evaluate,
    {
        "reduce_phase": ["write_report", "write_introduction", "write_conclusion"],
        "map_phase": "create_additional_analysts"  # 新的节点
    }
)
```

### 需要的额外工作

1. **新增 `create_additional_analysts` 节点**：基于已有 sections，判断还缺什么角度
2. **新增 `supervisor_evaluate` 节点**：LLM 评估研究完整性
3. **迭代次数上限**：防止无限循环（如 `max_supervisor_iterations = 3`）
4. **状态管理复杂化**：需要区分"第一批分析师"和"补充分析师"

### 什么时候值得做

- 研究主题非常开放（如"帮我了解 AI 领域的最新动态"）
- 需要深度探索而不是广度覆盖
- 用户期望系统自动发现遗漏的角度

对于"写一份关于 X 的报告"这类任务，当前的 Map-Reduce 已经足够。
