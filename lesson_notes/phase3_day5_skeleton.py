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
# 数据模型
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
    # create_analysts在这里读取human_feedback内容
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

# 条件路由函数，不是节点，是一个返回字符串的普通函数
def should_continue(state:GenerateAnalystsState):
    """
        1)返回值必须是节点名称或END
        2)读状态但不修改状态
        3)告诉langraph 下一步走哪个节点
    """
    human_feedback = state.get('human_analyst_feedback')
    if human_feedback:
        return "create_analysts"
    return END



# ============================================
# 构建图 —— 你来实现
# ============================================

builder = StateGraph(GenerateAnalystsState)


# TODO: 添加节点
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)

# TODO: 添加边：START → create_analysts → human_feedback
builder.add_edge(START,"create_analysts")
builder.add_edge("create_analysts","human_feedback")


# TODO: 添加条件边（替换原来的 builder.add_edge("human_feedback", END)）
# builder.add_conditional_edges(
#     "human_feedback",
#     should_continue,
#     ["create_analysts", END]
# )
# 添加条件边，
builder.add_conditional_edges(
    "human_feedback", # 从哪个节点出发
    should_continue,  # 路由函数
    [END,"create_analysts"] # 所有可能的目标节点
)

# TODO: 编译图（加上 interrupt_before 和 checkpointer）
# memory = MemorySaver()
# graph = builder.compile(
#     interrupt_before=["human_feedback"],
#     checkpointer=memory
# )

# 实现 human in loop, 通过在编译图上增加interrupt_before节点 和 checkpointer
memory = MemorySaver()
graph = builder.compile(
    interrupt_before=['human_feedback'], #执行到整个节点暂停
    checkpointer=memory # 状态持久化
)



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
    # TODO: 用 graph.get_state(config) 获取状态，打印每个分析师的 persona
    # 提示：
    #   state = graph.get_state(config)
    #   for a in state.values["analysts"]:
    #       print(a.persona)

    state = graph.get_state(config=config)
    # state.values() 当前完整状态dict
    # state.next 下一个要执行的节点名称列表
    # state.config 当前配置
    for analyst in state.values["analysts"]:
        print(analyst.persona)
    print(f"下一个要执行的节点列表为: {state.next}")


    # --- 第三步 A：满意，直接放行 ---
    # TODO: 取消注释测试"放行"
    # graph.invoke(None, config)
    # print("已放行，工作流结束")


    # --- 第三步 B：不满意，注入反馈后重新生成 ---
    # TODO: 取消注释测试"反馈"
    graph.update_state(config, {"human_analyst_feedback": "请增加一个关注法律合规的分析师"})
    graph.invoke(None, config)
    
    state = graph.get_state(config)
    print("\n=== 重新生成后的分析师 ===")
    for a in state.values["analysts"]:
        print(a.persona)
    
    graph.invoke(None, config)
    print("已放行，工作流结束")
