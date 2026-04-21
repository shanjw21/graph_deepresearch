# 阶段三 · Day 5：条件路由 + 人机协同（interrupt_before + checkpointer）

## 今日目标

1. 理解 `add_conditional_edges` 的用法——让工作流"分叉"
2. 掌握 `interrupt_before` + `checkpointer`——让工作流"暂停等人"
3. 实现"人类审核分析师 → 不满意就重新生成"的循环
4. 理解 invoke / get_state / update_state 三步交互模式

---

## 知识点一：从线性到分支

Day 3 搭的是线性流：
```
START → create_analysts → human_feedback → END
```

今天要把它变成带循环的分支流：
```
START → create_analysts → human_feedback ─┬─ 有反馈 → 回到 create_analysts
                                          └─ 无反馈 → END
```

关键变化：`human_feedback` 后面不是固定的 END，而是**根据状态动态决定**走哪条路。

---

## 知识点二：条件路由函数

条件路由函数**不是节点**，它是一个返回字符串的普通函数：

```python
def should_continue(state: GenerateAnalystsState):
    if state.get('human_analyst_feedback'):
        return "create_analysts"  # 有反馈 → 回去重新生成
    return END                      # 无反馈 → 结束
```

**规则：**
- 返回值必须是节点名字符串或 `END`
- 它读状态但不修改状态
- 它告诉 LangGraph "下一步走哪个节点"

---

## 知识点三：add_conditional_edges

把固定边换成条件边：

```python
# Day 3 的固定边（今天删掉）
builder.add_edge("human_feedback", END)

# Day 5 的条件边（今天加上）
builder.add_conditional_edges(
    "human_feedback",              # 从哪个节点出发
    should_continue,               # 路由函数
    ["create_analysts", END]       # 所有可能的目标节点
)
```

**对比：**
```
add_edge("A", "B")                    → A 完了必定走 B
add_conditional_edges("A", fn, [...]) → A 完了，fn 返回什么就走什么
```

---

## 知识点四：interrupt_before + checkpointer

### 问题：人类怎么介入？

之前的 `human_feedback` 节点是空的，执行到它直接跳过。现在我们要让图**在执行到它之前暂停**。

### 解法：compile 时配置

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(
    interrupt_before=['human_feedback'],  # 执行到这个节点前暂停
    checkpointer=memory                   # 状态持久化
)
```

### 三步交互模式

```
第一步：启动（执行到 human_feedback 前暂停）
  config = {"configurable": {"thread_id": "test-001"}}
  graph.invoke({"topic": "AI医疗", ...}, config)

第二步：人类查看状态
  state = graph.get_state(config)
  print(state.values["analysts"])  # 看到生成的分析师

第三步 A：满意，放行（不注入反馈）
  graph.invoke(None, config)  # None = 不改输入，从暂停处继续
  → should_continue 发现没反馈 → END

第三步 B：不满意，注入反馈后继续
  graph.update_state(config, {"human_analyst_feedback": "请加一个法律分析师"})
  graph.invoke(None, config)
  → should_continue 发现有反馈 → 回到 create_analysts
  → 重新生成 → 又到 human_feedback 前暂停
  → 人类再次审核...
```

**注意 config 参数：** 必须用 `thread_id` 标识会话，checkpointer 根据 thread_id 存取状态。

---

## 知识点五：get_state 返回什么

```python
state = graph.get_state(config)

state.values        # 当前完整状态 dict
state.next          # 下一个要执行的节点名（列表）
state.config        # 当前配置

# 示例
print(state.values["analysts"])   # [Analyst(...), Analyst(...), Analyst(...)]
print(state.next)                  # ['human_feedback'] — 说明暂停在 human_feedback 前
```

## 知识点六: graph.invoke(None,config)功能
```python
graph.invoke(input, config)
```
input:要更新到状态中的值
config:配置，包含thread_id等
传None的作用:
当传入None时，是在告诉langGraph:不向状态中添加任何新值，从上次保存的检查点继续执行.
**核心作用:**不更新任何状态字段，触发图从已保存的检查点恢复执行，常用于中断恢复，人机协同审批后继续等场景

## 知识点七：graph.update_state() + invoke(None,config)组合获取人类反馈

标准流程：
```markdown
图执行到interrupt_before暂停
外部获取人类反馈
graph.update_state(config,{"human_anlyst_feedback":"反馈内容"})
graph.invoke(None,config) 从暂停处继续，路由函数反馈后决定走哪条路
```

---

## 动手任务

### 任务：在 Day 3 代码基础上加入条件路由和人机协同

**骨架代码：**

```python
"""
阶段三 · Day 5 任务：条件路由 + 人机协同

你的任务：
1. 添加 should_continue 路由函数
2. 把固定边换成条件边
3. 给 compile 加上 interrupt_before 和 checkpointer
4. 补全主流程中的三步交互代码（标记 TODO 的部分）

基于 Day 3 的代码改造
"""

import os
from typing import List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver  # 新增

load_dotenv()

base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model = os.getenv("MODEL")


# ============================================
# 数据模型（和之前一样）
# ============================================

class Analyst(BaseModel):
    affiliation: str = Field(description="分析师的主要隶属机构或组织")
    name: str = Field(description="分析师姓名")
    role: str = Field(description="分析师在研究主题中的具体角色定位")
    description: str = Field(description="分析师的关注焦点、关切点和动机的详细描述")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="包含所有分析师角色和隶属机构的综合列表")


class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]


analyst_instructions = """你需要创建一组 AI 分析师人设。请严格遵循以下指引：

1. 先审阅研究主题：
{topic}

2. 查看（可选的）编辑反馈，它将指导分析师的人设创建：

{human_analyst_feedback}

3. 基于上述文档与/或反馈，识别最值得关注的主题。

4. 选出前 {max_analysts} 个主题。

5. 为每个主题分配一位分析师。"""

llm = ChatOpenAI(model=model, temperature=0, base_url=base_url, api_key=api_key)


# ============================================
# 节点函数
# ============================================

def create_analysts(state: GenerateAnalystsState):
    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_feedback = state["human_analyst_feedback"]
    structured_llm = llm.with_structured_output(Perspectives, method="function_calling")
    system_message = analyst_instructions.format(
        topic=topic,
        max_analysts=max_analysts,
        human_analyst_feedback=human_feedback
    )
    result = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="生成分析师集合。")
    ])
    return {"analysts": result.analysts}


def human_feedback(state: GenerateAnalystsState):
    pass


# ============================================
# 路由函数 —— 你来实现
# ============================================

# TODO: 写一个 should_continue 函数
# 提示：
#   - 参数是 state: GenerateAnalystsState
#   - 检查 state.get('human_analyst_feedback')
#   - 如果有值 → return "create_analysts"
#   - 如果没值 → return END



# ============================================
# 构建图 —— 你来实现
# ============================================

builder = StateGraph(GenerateAnalystsState)

# TODO: 添加节点
# builder.add_node("create_analysts", create_analysts)
# builder.add_node("human_feedback", human_feedback)

# TODO: 添加边
# START → create_analysts → human_feedback
# builder.add_edge(...)

# TODO: 添加条件边（替换原来的 builder.add_edge("human_feedback", END)）
# builder.add_conditional_edges(
#     "human_feedback",
#     should_continue,
#     ["create_analysts", END]
# )

# TODO: 编译图（加上 interrupt_before 和 checkpointer）
# memory = MemorySaver()
# graph = builder.compile(
#     interrupt_before=["human_feedback"],
#     checkpointer=memory
# )


# ============================================
# 运行：三步交互模式
# ============================================

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test-001"}}

    # --- 第一步：启动，执行到 human_feedback 前暂停 ---
    print("=== 第一步：生成分析师 ===")
    graph.invoke({
        "topic": "人工智能在医疗领域的应用",
        "max_analysts": 3,
        "human_analyst_feedback": "",
        "analysts": []
    }, config)

    # --- 第二步：查看生成的分析师 ---
    print("\n=== 第二步：查看生成的分析师 ===")
    # TODO: 用 graph.get_state(config) 获取状态
    # 打印每个分析师的 persona
    # 提示：
    #   state = graph.get_state(config)
    #   for a in state.values["analysts"]:
    #       print(a.persona)



    # --- 第三步 A：满意，直接放行 ---
    # TODO: 取消下面这行的注释来测试"放行"
    # graph.invoke(None, config)
    # print("已放行，工作流结束")

    # --- 第三步 B：不满意，注入反馈 ---
    # TODO: 取消下面几行的注释来测试"反馈后重新生成"
    # graph.update_state(config, {"human_analyst_feedback": "请增加一个关注法律合规的分析师"})
    # graph.invoke(None, config)
    #
    # 查看重新生成的分析师
    # state = graph.get_state(config)
    # print("\n=== 重新生成后的分析师 ===")
    # for a in state.values["analysts"]:
    #     print(a.persona)
    #
    # 最后放行
    # graph.invoke(None, config)
    # print("已放行，工作流结束")
```

**验收步骤：**

1. 运行，确认在 `human_feedback` 前暂停（第一步有输出，第二步能看到分析师）
2. 测试"放行"：取消第三步 A 的注释，确认正常结束
3. 再运行一次，测试"反馈"：取消第三步 B 的注释，确认重新生成后分析师有变化

**面试考点：**
- interrupt_before 不是暂停代码执行，而是把状态存到 checkpointer，下次用同一 thread_id 恢复
- 生产环境中，暂停和恢复可能是不同进程（Web 服务 A 暂停，Web 服务 B 恢复）
两次调用可能在不同的HTTP请求中，通过config中的thread_id关联
---

## 深度理解：interrupt_before + checkpointer 的底层原理

### 核心问题：为什么需要"暂停"？

普通程序是同步执行的：

```
invoke() → 节点1 → 节点2 → 节点3 → return 结果
         ←────────── 一个进程内一气呵成 ──────────→
```

但 human-in-the-loop 要求：**执行到某一步时停下来，等人类（可能几分钟后、甚至明天）再继续。** 进程不能一直挂着等。

### LangGraph 的实现原理

```
时间线 →

第一次调用 graph.invoke(input, config)
─────────────────────────────────
  │
  ├─ 执行 create_analysts → 生成分析师
  │
  ├─ 即将执行 human_feedback...
  │
  ├─ 检测到 interrupt_before=['human_feedback']
  │     → 把当前状态 {"analysts": [...], "topic": "AI医疗"}
  │     → 存入 checkpointer（用 thread_id 做 key）
  │     → 返回给调用者（不等了）
  │
  ▼ 进程结束，内存释放


                          （5分钟后...人类审核完了）


第二次调用 graph.invoke(None, config)
─────────────────────────────────
  │
  ├─ config 里有 thread_id="test-001"
  │
  ├─ 从 checkpointer 加载状态 → {"analysts": [...], ...}
  │
  ├─ 从上次暂停的位置继续 → 执行 human_feedback → should_continue
  │
  ▼ 返回最终结果
```

**三行代码背后的完整机制：**

```python
# compile 时做的事：
#   1. 构建图结构
#   2. 注册一个"拦截器"：每次执行节点前检查是否在 interrupt_before 列表中
#   3. 绑定 checkpointer：每个节点执行后自动保存状态快照

graph = builder.compile(
    interrupt_before=['human_feedback'],  # 注册拦截点
    checkpointer=memory                   # 绑定存储后端
)

# invoke 时做的事：
#   1. 执行节点前 → 检查是否命中 interrupt_before
#   2. 命中 → 保存状态到 checkpointer → 提前返回
#   3. 没命中 → 正常执行节点 → 执行后也保存状态 → 继续下一个节点

# invoke(None, config) 时做的事：
#   1. input 是 None → 不创建新状态
#   2. 从 checkpointer 加载上次保存的状态
#   3. 从暂停位置继续执行
```

### 对比：如果用纯 OpenAI SDK 实现

本质上是**手动做 LangGraph 帮你做的事**：

```
                    ┌─────────────────────┐
                    │    数据库 / Redis     │
                    │  存储中间状态          │
                    └──────┬──────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                       │
    ▼                      ▼                       ▼
 请求1: 生成           请求2: 人类审核         请求3: 继续
 ──────────────      ──────────────         ──────────────
 1.调LLM生成分析师    1.从DB读取状态          1.从DB读取状态
 2.存DB:             2.人类看结果            2.注入反馈到状态
   {                 3.人类写反馈            3.调LLM重新生成
     analysts:[...], 4.更新DB:              4.更新DB
     status:"等待审核"   feedback:"加法律"   5.status="完成"
   }                 5.返回"已更新"
 3.返回"暂停"
```

```python
# 纯 OpenAI SDK 实现 human-in-the-loop（伪代码）

import redis, json

db = redis.Redis()

def step1_generate(topic):
    """请求1：生成分析师，存DB，返回给人类审核"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"为{topic}生成3个分析师"}]
    )
    analysts = parse_response(response)

    # 手动存状态 ← LangGraph 的 checkpointer 自动做这一步
    db.set("session:001", json.dumps({
        "topic": topic,
        "analysts": [a.__dict__ for a in analysts],
        "status": "waiting_feedback"  # 手动标记状态 ← LangGraph 用 interrupt_before 自动做
    }))

    return analysts


def step2_review(session_id):
    """请求2：人类查看结果"""
    state = json.loads(db.get(f"session:{session_id}"))
    return state["analysts"]


def step3_continue(session_id, feedback):
    """请求3：注入反馈，继续执行"""
    state = json.loads(db.get(f"session:{session_id}"))

    if feedback:  # 手动判断路由 ← LangGraph 的 should_continue 自动做
        state["feedback"] = feedback
        response = openai.chat.completions.create(...)
        state["analysts"] = parse_response(response)

    state["status"] = "done"
    db.set(f"session:{session_id}", json.dumps(state))
    return state
```

### 一张图总结：LangGraph 自动化了什么

```
纯 OpenAI SDK                        LangGraph
─────────────                        ─────────

手动存状态到 DB                    → checkpointer 自动存
手动标记 status="waiting"          → interrupt_before 自动暂停
手动写 if/else 路由                 → add_conditional_edges 声明式定义
手动从 DB 读状态恢复                → invoke(None, config) 自动恢复
手动写 3 个 API 端点                → 一个 graph 对象搞定

你写的 15 行代码                    → 纯 SDK 可能需要 100+ 行
```

---

## 今日面试题

**Q1: "interrupt_before 和直接写 input() 有什么区别？"**
> interrupt_before 配合 checkpointer 实现状态持久化。暂停后进程可以安全退出，下次用同一 thread_id 调用就能恢复。input() 只能用于本地脚本，进程退出状态就丢了。

**Q2: "update_state 和直接传新输入有什么区别？"**
> update_state 是原地修改已有状态中的某个字段（比如只改 human_analyst_feedback），不影响其他字段。直接传新输入会覆盖整个输入。update_state 更适合"微调"而不是"重来"。

**Q3: "为什么需要 thread_id？"**
> checkpointer 用 thread_id 区分不同的会话。多个用户同时使用系统时，每个用户有自己的 thread_id，状态互不干扰。

---

## 参考源码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| should_continue 路由 | `research_assistant.py` | 615-633 |
| interrupt_before 编译 | `research_assistant.py` | 1180-1184 |
| initiate_all_interviews（高级路由） | `research_assistant.py` | 859-891 |
