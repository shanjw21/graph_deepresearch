"""
阶段四 · Day 7 任务：访谈子图（下）—— 完成剩余节点 + 组装循环

在 Day 6 的代码基础上添加：
1. generate_answer 节点
2. route_messages 路由
3. save_interview 节点
4. write_section 节点
5. 补全图的边（条件路由 + 收尾）

参考源码：research_assistant.py 第 730-856 行、1053-1100 行
"""

import os
import operator
from typing import List, Annotated
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import get_buffer_string
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph, MessagesState

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


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="用于检索的搜索查询语句")


class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list


# ============================================
# 提示词模板（4 套）
# ============================================

question_instructions = """你是一名分析师，需要通过访谈专家来了解一个具体主题。

你的目标是提炼与该主题相关的「有趣且具体」的洞见。

以下是你的关注主题与目标设定：{goals}

请先用符合你人设的名字进行自我介绍，然后提出你的第一个问题。

持续追问，逐步深入。

当你认为信息已充分，请以这句话结束访谈：「非常感谢您的帮助!」

请始终保持与你的人设与目标一致的说话方式。"""

search_instructions = SystemMessage(content="""你将获得一段分析师与专家之间的对话。

你的目标是基于这段对话，为Web搜索生成一条结构良好的查询语句。

特别关注分析师最后提出的问题。

将这个最终问题转化为结构良好的 Web 搜索查询。""")

answer_instructions = """你是一位被分析师访谈的专家。

以下是分析师的关注领域：{goals}。

你的目标是回答访谈者提出的问题。

回答问题时，请仅使用以下上下文：

{context}

回答须遵循如下要求：
1. 只使用上下文中提供的信息。
2. 不要引入上下文之外的信息。
3. 在涉及具体论断时，标注引用来源编号 [1] [2]。
4. 在答案结尾处按顺序列出引用来源。"""

section_writer_instructions = """你是一名资深技术写作者。

你的任务是基于一组来源文档，撰写一段简洁、易读的报告小节。

使用 Markdown 制作小节结构：## 标题，### 摘要，### 参考来源。

标题需要贴合分析师的关注点并具有吸引力：
{focus}

关于摘要部分：
- 先给出与分析师关注点相关的背景
- 强调访谈中获得的新颖洞见
- 使用数字引用 [1] [2]
- 控制在约 400 字以内
- **重要：全部使用中文**

在参考来源部分列出使用到的全部来源，合并重复来源。"""


# ============================================
# LLM 和搜索工具
# ============================================

llm = ChatOpenAI(model=model, temperature=0, base_url=base_url, api_key=api_key)

# TODO: 如果你没有 Tavily，可以保留这个导入但加 try-except 降级
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.document_loaders import WikipediaLoader
# tavily_search = TavilySearchResults(max_results=3)


# ============================================
# 节点函数
# ============================================

def generate_question(state: InterviewState):
    """分析师提问（Day 6 已实现，搬过来即可）"""
    # TODO: 复制你 Day 6 的实现
    pass


def search_web(state: InterviewState):
    """Web 搜索（Day 6 已实现，搬过来即可）"""
    # TODO: 复制你 Day 6 的实现
    pass


def search_baike(state: InterviewState):
    """百科搜索（Day 6 已实现，搬过来即可）"""
    # TODO: 复制你 Day 6 的实现
    pass


def generate_answer(state: InterviewState):
    """
    TODO: 实现专家回答

    提示：
    1. 从 state 读取 analyst, messages, context
    2. 用 answer_instructions.format(goals=analyst.persona, context=context) 格式化
    3. 调用 llm.invoke([SystemMessage(content=system_message)] + messages)
    4. 给 answer.name = "expert"  ← 关键！
    5. 返回 {"messages": [answer]}

    参考源码：research_assistant.py 第 730-759 行
    """
    pass


def save_interview(state: InterviewState):
    """
    TODO: 保存访谈记录

    提示：
    1. messages = state["messages"]
    2. interview = get_buffer_string(messages)
    3. 返回 {"interview": interview}

    参考源码：research_assistant.py 第 762-782 行
    """
    pass


def write_section(state: InterviewState):
    """
    TODO: 生成报告小节

    提示：
    1. 从 state 读取 interview, context, analyst
    2. 用 section_writer_instructions.format(focus=analyst.description) 格式化
    3. 调用 llm.invoke([SystemMessage(...), HumanMessage(content=f"使用这些来源撰写你的小节: {context}")])
    4. 返回 {"sections": [section.content]}

    参考源码：research_assistant.py 第 828-856 行
    """
    pass


# ============================================
# 路由函数
# ============================================

def route_messages(state: InterviewState, name: str = "expert"):
    """
    TODO: 实现循环路由

    提示：
    1. 从 state 获取 messages 和 max_num_turns（默认 2）
    2. 统计 messages 中 name==name 且类型是 AIMessage 的数量
    3. 如果数量 >= max_num_turns → return 'save_interview'
    4. 如果 messages[-2] 包含 "非常感谢您的帮助!" → return 'save_interview'
    5. 否则 → return "ask_question"

    参考源码：research_assistant.py 第 785-821 行
    """
    pass


# ============================================
# 构建完整子图 —— 你来实现
# ============================================

# builder = StateGraph(InterviewState)

# TODO: 添加 6 个节点
# builder.add_node("ask_question", generate_question)
# builder.add_node("search_web", search_web)
# builder.add_node("search_baike", search_baike)
# builder.add_node("answer_question", generate_answer)
# builder.add_node("save_interview", save_interview)
# builder.add_node("write_section", write_section)

# TODO: 添加边
# START → ask_question
# ask_question → search_web（并行）
# ask_question → search_baike（并行）
# search_web → answer_question（fan-in）
# search_baike → answer_question（fan-in）
# answer_question → route_messages → [ask_question, save_interview]（条件路由+循环）
# save_interview → write_section
# write_section → END

# TODO: 编译
# graph = builder.compile()


# ============================================
# 运行完整访谈
# ============================================

if __name__ == "__main__":
    sample_analyst = Analyst(
        affiliation="协和医院",
        name="张伟博士",
        role="医疗AI专家",
        description="关注AI在影像诊断中的应用，特别是CT和MRI的自动分析"
    )

    print("=== 开始完整访谈 ===\n")

    result = graph.invoke({
        "messages": [HumanMessage(content="所以你说你在写一篇关于AI在医疗中应用的文章?")],
        "max_num_turns": 2,
        "context": [],
        "analyst": sample_analyst,
        "interview": "",
        "sections": []
    })

    # 打印对话过程
    print("=== 访谈对话 ===")
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"  [Human] {msg.content[:80]}...")
        elif isinstance(msg, AIMessage):
            role = msg.name if msg.name else "分析师"
            print(f"  [{role}] {msg.content[:80]}...")

    # 打印搜索结果数量
    print(f"\n=== 搜索结果：{len(result['context'])} 条 ===")

    # 打印最终报告小节
    print("\n=== 报告小节 ===")
    for section in result["sections"]:
        print(section)

    print("\nDay 7 完成！访谈子图全部实现。")
