"""
阶段二 · Day 4 任务：给分析师工作流加上 Langfuse 追踪

你的任务：
1. 配置 .env 中的 Langfuse Key
2. 验证 Langfuse 连接
3. 补全 Langfuse 追踪代码（标记 TODO 的部分）
4. 运行后到 Langfuse 控制台查看追踪记录
"""

import os
from typing import List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph


# ============================================
# Step 1: 加载环境变量
# ============================================
load_dotenv()

base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model = os.getenv("MODEL")


# ============================================
# Step 2: 验证 Langfuse 连接
# ============================================
# TODO: 导入 get_client，创建 langfuse 客户端，验证连接
# 提示：
from langfuse import get_client
langfuse = get_client()
print("连接成功" if langfuse.auth_check() else "连接失败")



# ============================================
# Step 3: 数据模型 + 状态模型（和 Day 3 一样）
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


def human_feedback_node(state: GenerateAnalystsState):
    pass


builder = StateGraph(GenerateAnalystsState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback_node)
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_edge("human_feedback", END)
graph = builder.compile()


# ============================================
# Step 4: 运行 + Langfuse 追踪
# ============================================

if __name__ == "__main__":
    topic = "人工智能在医疗领域的应用"

    # TODO: 实现 Langfuse 追踪
    # 提示：
    #   1. 导入 CallbackHandler: from langfuse.langchain import CallbackHandler
    #   2. 创建 handler = CallbackHandler()
    #   3. 把 handler 传入 config={"callbacks": [handler]}
    #   4. 调用 graph.invoke(input_dict, config)
    #   5. 调用 langfuse.flush() 确保数据发送
    #
    # 注意：LangGraph 的 invoke 签名是
    #   graph.invoke(input, config=None)
    #   config 参数是第二个参数，不是 key-value

    # result = TODO: 你来实现

    from langfuse.langchain import CallbackHandler
    handler = CallbackHandler()
    config = {"callbacks":[handler]}
    input_dict = {
        "topic": "人工智能在医疗领域的应用",
        "max_analysts": 3,
        "human_analyst_feedback": "",
        "analysts": []
    }
    result = graph.invoke(input_dict,config=config)

    print("生成的分析师：")
    for i, analyst in enumerate(result["analysts"], 1):
        print(f"\n--- 分析师 {i} ---")
        print(analyst.persona)

    # TODO: 刷新 langfuse 缓冲区，确保数据发送到服务器
    # 提示：langfuse.flush()
    langfuse.flush()

    print("\n请到 Langfuse 控制台查看追踪记录！")
