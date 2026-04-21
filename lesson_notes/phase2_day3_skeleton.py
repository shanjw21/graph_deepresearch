"""
阶段二 · Day 3 任务：生成分析师工作流

你的任务：
1. 补全 create_analysts 节点函数（标记 TODO 的部分）
2. 补全图的构建（标记 TODO 的部分）
3. 运行并验证输出

参考源码：research_assistant.py 第 563-598 行
"""

import os
from typing import List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model = os.getenv("MODEL")


# ============================================
# Step 1: 数据模型（阶段一 Day 1 学过的）
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


# ============================================
# Step 2: 状态模型（阶段一 Day 2 学过的）
# ============================================

class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]


# ============================================
# Step 3: 提示词模板
# ============================================

analyst_instructions = """你需要创建一组 AI 分析师人设。请严格遵循以下指引：

1. 先审阅研究主题：
{topic}

2. 查看（可选的）编辑反馈，它将指导分析师的人设创建：

{human_analyst_feedback}

3. 基于上述文档与/或反馈，识别最值得关注的主题。

4. 选出前 {max_analysts} 个主题。

5. 为每个主题分配一位分析师。"""


# ============================================
# Step 4: 初始化 LLM
# ============================================

llm = ChatOpenAI(model=model, temperature=0,base_url=base_url,api_key=api_key)


# ============================================
# Step 5: 节点函数 —— 你来实现
# ============================================

def create_analysts(state: GenerateAnalystsState):
    """
    TODO: 实现这个函数

    提示：
    1. 从 state 中读取 topic, max_analysts, human_analyst_feedback
    2. 用 llm.with_structured_output(Perspectives) 创建结构化 LLM
    3. 用 analyst_instructions.format(...) 格式化提示词
    4. 调用 structured_llm.invoke([SystemMessage(...), HumanMessage(...)])
    5. 返回 {"analysts": 结果.analysts}
    """
    # TODO: 在这里写你的代码
    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_feedback = state["human_analyst_feedback"]
    structured_llm = llm.with_structured_output(Perspectives, method="function_calling")
    system_message = analyst_instructions.format(topic = topic,max_analysts = max_analysts,human_analyst_feedback=human_feedback)
    result = structured_llm.invoke([SystemMessage(content=system_message), HumanMessage(content="生成分析师集合")])
    return {"analysts": result.analysts}


def human_feedback(state: GenerateAnalystsState):
    """空节点，作为人类审核的停靠点"""
    pass


# ============================================
# Step 6: 构建图 —— 你来实现
# ============================================

# TODO: 创建 StateGraph，使用 GenerateAnalystsState
builder = StateGraph(GenerateAnalystsState)

# TODO: 添加节点 create_analysts 和 human_feedback
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback",human_feedback)

# TODO: 添加边：START → create_analysts → human_feedback → END
builder.add_edge(START,"create_analysts")
builder.add_edge("create_analysts","human_feedback")
builder.add_edge("human_feedback",END)

# TODO: 编译图
graph = builder.compile()


# ============================================
# Step 7: 运行测试
# ============================================

if __name__ == "__main__":
    result = graph.invoke({
        "topic": "人工智能在医疗领域的应用",
        "max_analysts": 3,
        "human_analyst_feedback": "",
        "analysts": []
    })

    print("生成的分析师：")
    for i, analyst in enumerate(result["analysts"], 1):
        print(f"\n--- 分析师 {i} ---")
        print(analyst.persona)

