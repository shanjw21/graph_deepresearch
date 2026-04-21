# AI 智能体系统 — 面试项目学习指南

> **两大项目联动：构建多智能体系统 + 建立质量评估体系**
> 项目一教会你"怎么造"，项目二教会你"怎么验"——面试中这两手都要硬。

---

## 两个项目的关系

```
┌─────────────────────────────────────────────────────────────────┐
│                        面试故事线                                 │
│                                                                   │
│  "我基于 LangGraph 构建了一个 Map-Reduce 多智能体深度研究系统，       │
│   同时使用 Langfuse 建立了完整的在线/离线评估体系和安全监控机制。"     │
│                                                                   │
│  ┌──────────────────┐          ┌──────────────────────────┐      │
│  │ 项目一：          │   →→→    │ 项目二：                  │      │
│  │ Deep Research     │  追踪评估 │ Langfuse Agent           │      │
│  │ 多智能体系统       │          │ 质量评估与安全监控          │      │
│  │ (怎么造)          │          │ (怎么验)                  │      │
│  └──────────────────┘          └──────────────────────────┘      │
│                                                                   │
│  技术关键词: LangGraph / Map-Reduce / Send / Langfuse /           │
│             LLM-as-a-Judge / PII脱敏 / Prompt Injection 防护     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 全局技术栈总览

| 层级 | 技术 | 作用 |
|------|------|------|
| **工作流编排** | LangGraph StateGraph | 有向图建模，节点+边定义工作流 |
| **LLM 调用** | OpenAI GPT-4o + ChatOpenAI | 推理引擎，结构化输出 |
| **数据搜索** | Tavily + Wikipedia | 实时Web搜索 + 知识库检索 |
| **可观测性** | Langfuse SDK v3 | Trace追踪、Span打点、评分记录 |
| **在线评估** | Langfuse CallbackHandler | 实时成本/延迟/用户反馈监控 |
| **离线评估** | Langfuse Dataset + LLM-as-a-Judge | 基准数据集评测 + 自动打分 |
| **安全防护** | LLM Guard (BanTopics/Anonymize/PromptInjection) | 暴力检测/PII脱敏/注入防护 |
| **部署** | Docker Compose + Redis + PostgreSQL | 容器化生产部署 |

---

## 学习路线图

```
阶段一 ──→ 阶段二 ──→ 阶段三 ──→ 阶段四 ──→ 阶段五 ──→ 阶段六 ──→ 阶段七
基础砖块    单线工作流   条件路由     访谈子图     Map-Reduce   质量评估     安全防护
(2天)      (2天)      (1-2天)    (2-3天)★    (2天)★     (2-3天)★    (2天)
                                                                   
 ↑ 项目一：Deep Research 多智能体系统 ↑           ↑ 项目二：Langfuse 评估 ↑

总时长：约 14-17 天
★ 标记为面试核心，必须深度掌握
```

---

## 阶段一：基础砖块（2 天）

> **目标：** 掌握 Pydantic 结构化输出 + LangGraph 状态模型

### 核心概念

| 概念 | 代码位置 | 面试话术 |
|------|----------|----------|
| Pydantic 模型 | `research_assistant.py:215-278` | "用 Pydantic 定义 LLM 的输出 schema，通过 function calling 约束返回格式" |
| `with_structured_output` | `research_assistant.py:582` | "LLM 返回结构化数据而不是自由文本，避免解析失败" |
| TypedDict 状态 | `ResearchGraphState`, `InterviewState` | "用 TypedDict 定义图的节点间传递状态，比 dataclass 更轻量" |
| `operator.add` 累加 | `Annotated[list, operator.add]` | "并行节点返回的结果自动合并，这是 Map-Reduce 聚合的核心机制" |
| MessagesState | `InterviewState(MessagesState)` | "LangGraph 内置的消息列表状态，自动追加消息" |

### 动手任务

**任务 1：重建数据模型（不看源码）**

根据描述，自己写出 `Analyst`、`Perspectives`、`SearchQuery`、`ResearchGraphState`、`InterviewState` 这 5 个类：
- `Analyst`：4 个字段 + `@property persona`
- `ResearchGraphState`：9 个字段，`sections` 用 `operator.add`
- `InterviewState`：继承 `MessagesState`，`context` 用 `operator.add`

**任务 2：最小结构化输出 Demo**

```python
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI

class Analyst(BaseModel):
    name: str = Field(description="分析师姓名")
    role: str = Field(description="角色定位")
    affiliation: str = Field(description="隶属机构")

class Perspectives(BaseModel):
    analysts: List[Analyst]

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(Perspectives)
result = structured_llm.invoke("为'AI在医疗中的应用'生成3个分析师")
print(result.analysts)
```

运行后修改：增加 `description` 字段，观察输出变化。

### 面试高频问题

1. **"为什么用 Pydantic 而不是让 LLM 返回 JSON？"**
   → `with_structured_output` 底层走 function calling，Pydantic 自动校验类型，比手动解析 JSON 字符串更可靠
2. **"temperature=0 的意义？"**
   → 保证报告生成的确定性和可复现性
3. **"operator.add 在状态中的作用？"**
   → 并行节点返回的 list 自动用 `+` 合并到全局状态，Map-Reduce 的基础

---

## 阶段二：单线工作流（2 天）

> **目标：** 理解 StateGraph，能用节点+边搭出线性+条件分支工作流

### 核心概念

| 概念 | 代码位置 | 说明 |
|------|----------|------|
| `StateGraph` | `research_assistant.py:1125` | 用有向图建模工作流 |
| `add_node` | `research_assistant.py:1132-1138` | 每个节点是纯函数：接收状态，返回 partial dict |
| `add_edge` | `research_assistant.py:1144` | 定义数据流向 |
| `compile + invoke` | `research_assistant.py:1181` | 编译为可执行图 |
| 条件路由 | `research_assistant.py:615-633` | `add_conditional_edges` 根据状态选择路径 |

### 动手任务

**任务 1：搭出邮件分类 Agent（参考 Langfuse 项目 03 notebook）**

这是一个比 Deep Research 更简单的 LangGraph Agent，适合入门：

```
START → read_email → classify_email ─┬→ handle_spam → END
                                     └→ drafting_response → notify → END
```

关键代码骨架：
```python
class EmailState(TypedDict):
    email: Dict[str, Any]
    is_spam: Optional[bool]
    draft_response: Optional[str]
    messages: List[Dict[str, Any]]

def route_email(state: EmailState) -> str:
    return "spam" if state["is_spam"] else "legitimate"

email_graph = StateGraph(EmailState)
# 添加 5 个节点...
# add_conditional_edges("classify_email", route_email, {"spam": "handle_spam", "legitimate": "drafting_response"})
```

**任务 2：加入 Langfuse 追踪**

```python
from langfuse import observe
from langfuse.langchain import CallbackHandler

@observe()
def process_email(email_data):
    langfuse_handler = CallbackHandler()
    result = compiled_graph.invoke(
        input={...},
        config={"callbacks": [langfuse_handler]}  # 自动追踪每个节点
    )
    return result
```

### 面试高频问题

1. **"节点函数的返回值格式？"** → 返回 `dict`，key 对应状态字段，LangGraph 自动 merge
2. **"为什么要用 StateGraph 而不是直接调 LLM？"** → 条件路由、并行执行、状态持久化、人机协同
3. **"Langfuse CallbackHandler 的原理？"** → 利用 LangChain 的回调机制，在每次 LLM 调用前后自动记录 token、延迟、成本

---

## 阶段三：条件路由与人机协同（1~2 天）

> **目标：** 掌握 `add_conditional_edges` + `interrupt_before` + `checkpointer`

### 核心机制

```
create_analysts → human_feedback ─┬─ 有反馈 → 回到 create_analysts
                                  └─ 无反馈 → 启动并行访谈
```

**interrupt_before（源码第 1181-1184 行）：**
```python
graph = builder.compile(
    interrupt_before=['human_feedback'],  # 执行到此节点前暂停
    checkpointer=memory                   # 状态持久化
)
```

执行流程：
1. `graph.invoke(input)` → 执行到 `human_feedback` 前暂停
2. 人工检查 `state["analysts"]`
3. `graph.update_state(config, {"human_analyst_feedback": "..."})` → 注入反馈
4. `graph.invoke(None, config)` → 从中断点继续

### 动手任务

在阶段二的邮件 Agent 基础上：
1. 在 `classify_email` 前加入 `interrupt_before`，让人类可以覆盖 AI 的分类结果
2. 使用 `MemorySaver` 作为 checkpointer
3. 实现暂停 → 人工审核 → 注入反馈 → 继续执行的完整流程

### 面试高频问题

1. **"interrupt_before 和直接写 input() 的区别？"**
   → interrupt_before 配合 checkpointer 实现状态持久化，进程可以退出后恢复，适合 Web 服务
2. **"checkpointer 的作用？"**
   → 节点执行后自动保存状态快照，支持暂停/恢复、时间旅行调试

---

## 阶段四：访谈子图（2~3 天）—— 项目一核心

> **目标：** 重建访谈子图，理解循环对话+并行搜索+条件路由

### 子图结构

```
ask_question (分析师提问)
     ├──→ search_web   ──┐
     │                    ├──→ answer_question (专家回答)
     └──→ search_baike  ──┘
                              │
                         route_messages
                          │         │
                   继续提问     保存访谈
                     ↓              ↓
               ask_question    write_section → END
```

### 关键设计决策

**Q: 为什么两个搜索要并行？**
→ 一个问题同时查两个数据源。LangGraph 的 add_edge 自动处理：`answer_question` 有两个入边时，等两个搜索都完成才执行。

**Q: route_messages 的两种退出条件？**
1. 轮次达到 `max_num_turns`（默认 2）
2. 分析师说了"非常感谢您的帮助!"

**Q: context 为什么用 `operator.add`？**
→ 每轮搜索结果追加到 context 列表，后续写 section 可引用所有素材。

### 动手任务

**Step 1 — 实现 6 个节点函数：**
- `generate_question`：用 analyst.persona 格式化提示词
- `search_web`：结构化输出 SearchQuery → Tavily → XML 格式化
- `search_baike`：结构化输出 SearchQuery → Wikipedia → XML 格式化
- `generate_answer`：用 context + messages 生成专家回答，`answer.name = "expert"`
- `save_interview`：`get_buffer_string(messages)` 转成字符串
- `write_section`：用 context + analyst.description 生成报告小节

**Step 2 — 组装图并测试：**
```python
interview_builder = StateGraph(InterviewState)
# 添加 6 个节点
# 定义边和条件路由
# 测试单个访谈的完整执行
```

### 面试高频问题

1. **"并行搜索结果怎么合并？"** → 两个节点都返回 `{"context": [doc]}`，`operator.add` 自动合并
2. **"专家回答为什么只基于 context？"** → 防幻觉，确保所有论述有据可查，来源可追溯
3. **"搜索结果为什么用 XML 格式？"** → LLM 对 XML 标签的解析能力强于 JSON

---

## 阶段五：Map-Reduce 并行调度（2 天）—— 架构亮点

> **目标：** 理解 Send API + Reduce 三路并行 + finalize_report 组装

### Send API（源码第 859-891 行）

```python
def initiate_all_interviews(state):
    return [
        Send("conduct_interview", {
            "analyst": analyst,
            "messages": [HumanMessage(content=f"所以你说你在写一篇关于{topic}的文章?")]
        })
        for analyst in state["analysts"]
    ]
```

**Send 的本质：** 运行时根据数据量动态创建节点实例，类似 Spark 的 map。

### Reduce 三路并行（源码第 1162-1170 行）

```python
# 三个写入任务自动并行
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")

# finalize_report 有 3 个入边，自动等待全部完成
builder.add_edge(
    ["write_conclusion", "write_report", "write_introduction"],
    "finalize_report"
)
```

### 动手任务

1. 把访谈子图嵌入主图：`builder.add_node("conduct_interview", interview_graph)`
2. 用 `initiate_all_interviews` 实现 Send 并行
3. 实现三路并行报告生成 + `finalize_report` 组装
4. 端到端运行完整系统，生成一份研究报告

### 面试高频问题

1. **"Send 和 add_edge 的区别？"** → add_edge 是静态连接；Send 是动态的，运行时决定创建多少实例
2. **"并行结果怎么聚合？"** → `sections` 字段通过 `operator.add` 自动合并
3. **"为什么引言/主体/结论分三个节点？"** → 互不依赖，并行可节省约 2/3 等待时间
4. **"API 调用次数？"** → 3分析师 × 2轮 ≈ 6次访谈 + 3小节 + 4报告 ≈ 13次

---

## 阶段六：智能体质量评估（2~3 天）—— 项目二核心

> **目标：** 掌握在线评估 + 离线评估 + LLM-as-a-Judge，能说出"怎么验证我的系统是好的"

### 6.1 评估体系全景

```
┌────────────────────────────────────────────────────────────┐
│                    智能体评估体系                             │
│                                                              │
│  ┌─────────── 在线评估（生产环境）─────────────┐              │
│  │                                              │              │
│  │  1. 成本追踪 → 每个 LLM 调用的 token 消耗     │              │
│  │  2. 延迟监控 → 每个节点的执行时间              │              │
│  │  3. 用户反馈 → 点赞/点踩，score_trace()       │              │
│  │  4. LLM 评审 → 实时自动评估输出质量            │              │
│  └──────────────────────────────────────────────┘              │
│                                                              │
│  ┌─────────── 离线评估（开发阶段）─────────────┐              │
│  │                                              │              │
│  │  1. 基准数据集 → HF datasets 上传到 Langfuse  │              │
│  │  2. 批量运行 → 不同模型/提示词的对比实验       │              │
│  │  3. LLM-as-a-Judge → Answer Correctness 评分 │              │
│  │  4. 结果对比 → 多模型性能横向比较              │              │
│  └──────────────────────────────────────────────┘              │
└────────────────────────────────────────────────────────────┘
```

### 6.2 在线评估实战

**任务 1：给邮件 Agent 加上 Langfuse 追踪**

```python
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

langfuse = get_client()

@observe()
def process_email(email_data):
    with langfuse.start_as_current_span(name="email-processing") as span:
        langfuse_handler = CallbackHandler()
        result = compiled_graph.invoke(
            input={...},
            config={"callbacks": [langfuse_handler]}
        )
        span.update_trace(input={"email": email_data}, output=result)
    return result
```

**任务 2：三种评分方式（面试必背）**

```python
# 方案一：span 直接评分（推荐，适合即时评分）
with langfuse.start_as_current_span(name="workflow") as span:
    # ... 执行逻辑 ...
    span.score_trace(name="user-feedback", value=1, data_type="NUMERIC")

# 方案二：全局方法（适合深层函数）
with langfuse.start_as_current_span(name="workflow") as span:
    langfuse.score_current_trace(name="accuracy", value=0.95)

# 方案三：trace_id 直接评分（适合异步/批量/用户延迟反馈）
langfuse.create_score(
    trace_id="xxx",  # 从数据库中取
    name="user-feedback", value=1
)
```

### 6.3 离线评估实战

**任务 3：创建基准数据集并批量评测**

```python
# 1. 从 HuggingFace 加载数据集
from datasets import load_dataset
dataset = load_dataset("junzhang1207/search-dataset", split="train")

# 2. 上传到 Langfuse
langfuse.create_dataset(name="qa-benchmark", description="问答基准")
for row in dataset.sample(30):
    langfuse.create_dataset_item(
        dataset_name="qa-benchmark",
        input={"text": row["question"]},
        expected_output={"text": row["expected_answer"]}
    )

# 3. 批量运行 Agent
dataset = langfuse.get_dataset('qa-benchmark')
for item in dataset.items:
    with item.run(run_name="run_gpt-4o") as root_span:
        output = my_agent(str(item.input), langfuse_handler)
        root_span.score_trace(name="user-feedback", value=1)

# 4. 在 Langfuse 控制台配置 LLM-as-a-Judge 评估器
#    自动对比 agent 输出 vs expected_output → 给出 Answer Correctness 分数
```

### 6.4 LLM-as-a-Judge 配置流程

```
1. 创建评估器 → 选择"托管评估器" → Answer Correctness
2. 设置裁判模型 → gpt-4o-mini（成本低）
3. 选择数据源 → 数据集运行
4. 变量映射：
   - {{ground_truth}} ← Dataset item → Expected output
   - {{answer}} ← Trace → Output
5. 执行评估 → 查看评分结果
```

### 面试高频问题

1. **"在线评估和离线评估的区别？"**
   → 在线是生产环境实时监控（成本/延迟/用户反馈）；离线是开发阶段用基准数据集系统性测试
2. **"LLM-as-a-Judge 的原理？"**
   → 用一个独立的 LLM 作为裁判，对比 agent 输出和标准答案，自动打分。本质是"用 AI 评估 AI"
3. **"三种评分方式分别适合什么场景？"**
   → 方案一：即时评分（AI 自动评估）；方案二：深层函数评分；方案三：异步评分（用户延迟反馈/批量评估）
4. **"怎么对比不同模型的效果？"**
   → 同一数据集上运行不同模型，用 LLM-as-a-Judge 统一评分，在 Langfuse 控制台横向对比

---

## 阶段七：安全防护（2 天）

> **目标：** 掌握三种安全防护模式，面试中展现"生产级思维"

### 7.1 三种安全威胁及防护

```
┌───────────────────────────────────────────────────────────────┐
│                     LLM 安全防护体系                            │
│                                                                 │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│  │ 禁止主题检测  │  │ PII 脱敏/复原     │  │ 提示词注入检测    │   │
│  │ BanTopics    │  │ Anonymize/       │  │ PromptInjection  │   │
│  │              │  │ Deanonymize      │  │                  │   │
│  │ 检测暴力/     │  │ 人名/机构名/      │  │ 检测"忽略指令"   │   │
│  │ 敏感话题     │  │ 邮箱/电话脱敏     │  │ "奶奶把戏"等攻击  │   │
│  └─────────────┘  └──────────────────┘  └─────────────────┘   │
│         ↓                  ↓                    ↓               │
│    Langfuse 记录风险分 + Langfuse 追踪脱敏过程 + 风险分记录      │
└───────────────────────────────────────────────────────────────┘
```

### 7.2 禁止主题检测（暴力内容防护）

```python
from llm_guard.input_scanners import BanTopics

violence_scanner = BanTopics(topics=["violence"], threshold=0.5)
sanitized, is_valid, risk_score = violence_scanner.scan(user_input)

if risk_score > 0.5:
    return "内容不安全，已拦截"

# 记录到 Langfuse
with langfuse.start_as_current_span(name="security-check") as span:
    span.score(name="input-violence", value=risk_score)
```

**原理：** 使用零样本分类模型（roberta-base-zeroshot-v2.0-c），你只需定义标签，模型就能理解语义并判断。

### 7.3 PII 脱敏/复原（隐私保护）

```
原始输入 → Anonymize（脱敏）→ LLM 处理 → Deanonymize（复原）→ 最终输出
"Kelly Hyman"    →  [REDACTED_PERSON_2]  →  保留占位符    →  "Kelly Hyman"
                                       ↑
                            Vault 维护映射关系
```

```python
from llm_guard.input_scanners import Anonymize
from llm_guard.output_scanners import Deanonymize
from llm_guard.vault import Vault

vault = Vault()

# 脱敏：识别 PERSON, ORGANIZATION, EMAIL, PHONE
scanner = Anonymize(vault, entity_types=["PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "PHONE_NUMBER"])
sanitized, is_valid, risk = scanner.scan(prompt)

# LLM 处理脱敏后的文本...
answer = llm.invoke(sanitized)

# 复原：查 Vault 映射表，把占位符换回原始值
de_scanner = Deanonymize(vault)
final_output, _, _ = de_scanner.scan(sanitized, answer)
```

### 7.4 提示词注入检测

```python
from llm_guard.input_scanners import PromptInjection

scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)
sanitized, is_valid, risk_score = scanner.scan(user_input)

if risk_score > 0.5:
    return "检测到提示词注入，已拦截"
```

**原理：** 使用 deberta-v3-base-prompt-injection-v2 模型，能识别直接注入攻击。
**局限：** 对中文"奶奶把戏"等社会工程学攻击可能漏报，需要配合 LLM 自身的安全对齐。

### 面试高频问题

1. **"PII 脱敏为什么需要 Vault？"**
   → Vault 维护占位符到原始值的映射，保证脱敏后的响应能正确还原
2. **"LLM Guard 的零样本分类是什么意思？"**
   → 模型已学会语义理解，你只需定义标签（如"violence"），它就能判断，不需要额外训练
3. **"提示词注入检测有什么局限？"**
   → 对中文复杂语境、社会工程学攻击可能漏报，需要多层防护（检测模型 + LLM 自身安全对齐 + 人工审核）
4. **"这些安全措施怎么和 Langfuse 结合？"**
   → 把风险分数通过 `span.score()` 记录到 trace，在 Langfuse 控制台可以按风险分过滤和告警

---

## 面试实战准备清单

### 必须能画出的图

- [ ] Deep Research 整体架构图（Map-Reduce 全景）
- [ ] 访谈子图内部流程图（循环+并行搜索）
- [ ] 状态传递图（ResearchGraphState 各字段在哪些节点被读写）
- [ ] 评估体系全景图（在线评估 vs 离线评估）
- [ ] PII 脱敏/复原流程图

### 必须能回答的问题

| 类别 | 问题 |
|------|------|
| **架构** | "整个系统用什么架构模式？" → Map-Reduce |
| **并行** | "并行怎么实现的？" → Send API + operator.add |
| **子图** | "为什么用子图？" → 可复用、独立状态、可独立测试 |
| **人机协同** | "人类怎么介入？" → interrupt_before + checkpointer |
| **防幻觉** | "怎么保证专家不乱说？" → 只用 context，强制引用 [1][2] |
| **可观测性** | "怎么监控 Agent 质量？" → Langfuse Trace/Span/Score |
| **在线评估** | "生产环境怎么评估？" → 成本追踪+延迟监控+用户反馈+LLM评审 |
| **离线评估** | "上线前怎么测试？" → 基准数据集+LLM-as-a-Judge+模型对比 |
| **安全** | "怎么防护提示词注入？" → LLM Guard 检测 + LLM 安全对齐 + 人工审核 |
| **隐私** | "怎么保护用户数据？" → PII 脱敏(Anonymize) → LLM处理 → 复原(Deanonymize) |

### 加分话术（主动提及）

1. "我的系统用了一个巧妙的设计：**子图作为节点嵌入主图**，实现 Map-Reduce 的自动并行和聚合"
2. "我不仅构建了多智能体系统，还用 **Langfuse 建立了完整的评估闭环**：在线监控成本/延迟，离线用基准数据集 + LLM-as-a-Judge 做质量评测"
3. "生产环境需要多层安全防护：**LLM Guard 做第一层拦截，LLM 自身安全对齐做第二层，Langfuse 记录风险分做持续监控**"
4. "PII 脱敏用 Vault 维护映射关系，保证脱敏后的响应能正确还原，**OpenAI 服务器只看到占位符，永远不接触原始敏感数据**"

---

## 两个项目如何串联讲

面试时可以这样组织你的项目介绍：

> "我的项目分为两部分：
>
> **第一部分是构建**——我基于 LangGraph 构建了一个 Map-Reduce 模式的多智能体深度研究系统。它会动态生成 3-5 个不同视角的分析师，通过 Send API 并行执行访谈子图，每个子图内部实现了循环对话+并行搜索（Tavily+维基百科）。所有访谈完成后，Reduce 阶段三路并行生成报告主体、引言和结论，最后组装成完整报告。
>
> **第二部分是评估**——我使用 Langfuse 建立了完整的质量评估体系。在线评估方面，通过 CallbackHandler 自动追踪每个 LLM 调用的 token 和延迟，支持用户反馈评分。离线评估方面，我从 HuggingFace 加载基准数据集，批量运行不同模型的 Agent，用 LLM-as-a-Judge 自动评估 Answer Correctness。同时我还实现了安全防护，包括暴力内容检测、PII 脱敏复原、提示词注入检测，所有风险分数都记录在 Langfuse 中做持续监控。"

---

## 文件索引

| 文件 | 内容 | 学习阶段 |
|------|------|----------|
| `02-agent-multi-role/deepresearch/deployment/research_assistant.py` | 多智能体系统完整源码（1185行） | 阶段一~五 |
| `04-agent-evaluation/langfuse/code/03_trace_and_evaluation_langgraph_agents.ipynb` | 邮件Agent + 追踪 + 在线/离线评估 | 阶段二~六 |
| `04-agent-evaluation/langfuse/code/04_llm_security_monitoring.ipynb` | 安全防护（暴力/PII/注入） | 阶段七 |
| `04-agent-evaluation/langfuse/docs/LLM-as-a-Judge 评估器.md` | LLM-as-a-Judge 配置指南 | 阶段六 |
