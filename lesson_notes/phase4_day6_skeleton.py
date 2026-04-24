"""
阶段四 · Day 6 任务：访谈子图（上）—— 前 3 个节点 + 并行搜索

你的任务：
1. 实现 generate_question 节点
2. 实现 search_web 节点（Tavily 搜索）
3. 实现 search_baike 节点（Wikipedia 搜索）
4. 构建图：ask_question → [search_web, search_baike] 并行 → 打印 context

暂不实现：answer_question / route_messages / save_interview / write_section

参考源码：research_assistant.py 第 640-727 行
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
# 提示词模板
# ============================================

question_instructions = """你是一名分析师，需要通过访谈专家来了解一个具体主题。

你的目标是提炼与该主题相关的「有趣且具体」的洞见。

1. 有趣（Interesting）：让人感到意外或非显而易见的观点。

2. 具体（Specific）：避免泛泛而谈，包含专家提供的具体案例或细节。

以下是你的关注主题与目标设定：{goals}

请先用符合你人设的名字进行自我介绍，然后提出你的第一个问题。

持续追问，逐步深入，逐步完善你对该主题的理解。

当你认为信息已充分，请以这句话结束访谈：「非常感谢您的帮助!」

请始终保持与你的人设与目标一致的说话方式。"""

search_instructions = SystemMessage(content="""你将获得一段分析师与专家之间的对话。

你的目标是基于这段对话，为Web搜索生成一条结构良好的查询语句。

首先，通读整段对话。

特别关注分析师最后提出的问题。

将这个最终问题转化为结构良好的 Web 搜索查询。""")


# ============================================
# 初始化 LLM 和搜索工具
# ============================================

llm = ChatOpenAI(model=model, temperature=0, base_url=base_url, api_key=api_key)

# TODO: 初始化搜索工具
# 提示：
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
tavily_search = TavilySearchResults(max_results=3)



# ============================================
# 节点函数 —— 你来实现
# ============================================

def generate_question(state: InterviewState):
    """
    TODO: 实现分析师提问

    提示：
    1. 从 state 中读取 analyst 和 messages
    2. 用 question_instructions.format(goals=analyst.persona) 格式化提示词
    3. 调用 llm.invoke([SystemMessage(content=system_message)] + messages)
    4. 返回 {"messages": [question]}

    参考源码：research_assistant.py 第 640-664 行
    """
    analyst = state["analyst"]
    messages = state["messages"]
    system_prompt = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    return {"messages":[question]}


def search_web(state: InterviewState):
    """
    TODO: 实现 Web 搜索

    提示：
    1. 用 llm.with_structured_output(SearchQuery, method="function_calling") 生成搜索词
    2. 调用 structured_llm.invoke([search_instructions] + state['messages'])
    3. 用 tavily_search.invoke(search_query.search_query) 执行搜索
    4. 格式化结果为 XML：
       '<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
    5. 返回 {"context": [formatted_docs]}

    参考源码：research_assistant.py 第 667-694 行
    """
    structured_llm = llm.with_structured_output(SearchQuery,method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])
    search_docs = tavily_search.invoke(search_query.search_query)
    formatted_docs = "\n\n---\n\n".join([
        f'<Document href="{doc["url"]}" />\n{doc["content"]}\n</Document>'
        for doc in search_docs
    ])
    return {"context":[formatted_docs]}


def search_baike(state: InterviewState):
    """
    TODO: 实现百科搜索

    提示：
    1. 和 search_web 前两步一样，生成搜索词
    2. 用 WikipediaLoader(query=search_query.search_query, load_max_docs=2).load() 搜索
    3. 格式化结果为 XML：
       '<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
    4. 返回 {"context": [formatted_docs]}

    参考源码：research_assistant.py 第 697-727 行
    """
    structured_llm = llm.with_structured_output(SearchQuery,method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])
    search_docs = WikipediaLoader(query=search_query.search_query,load_max_docs=2).load()
    formatted_docs = "\n\n---\n\n".join([
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page","")}"/> \n{doc.page_content}\n</Document>' 
        for doc in search_docs
    ])
    return {"context":[formatted_docs]}


# ============================================
# 构建图 —— 你来实现
# ============================================

builder = StateGraph(InterviewState)

# TODO: 添加 3 个节点
builder.add_node("ask_question", generate_question)
builder.add_node("search_web", search_web)
builder.add_node("search_baike", search_baike)

# TODO: 添加边
# START → ask_question
# ask_question → search_web（并行）
# ask_question → search_baike（并行）
# search_web → END（暂时的，Day 7 会改成 answer_question）
# search_baike → END（暂时的）
builder.add_edge(START,"ask_question")
builder.add_edge("ask_question","search_web")
builder.add_edge("ask_question","search_baike")
builder.add_edge("search_web",END)
builder.add_edge("search_baike",END)

# TODO: 编译
graph = builder.compile()


# ============================================
# 运行测试
# ============================================

if __name__ == "__main__":
    sample_analyst = Analyst(
        affiliation="协和医院",
        name="张伟博士",
        role="医疗AI专家",
        description="关注AI在影像诊断中的应用，特别是CT和MRI的自动分析"
    )

    result = graph.invoke({
        "messages": [HumanMessage(content="所以你说你在写一篇关于AI在医疗中应用的文章?")],
        "max_num_turns": 2,
        "context": [],
        "analyst": sample_analyst,
        "interview": "",
        "sections": []
    })

    print("=== 生成的提问 ===")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"  AI: {msg.content}")

    print(f"\n=== 搜索结果（context 有 {len(result['context'])} 条）===")
    for i, ctx in enumerate(result["context"]):
        print(f"  [{i+1}] {ctx}")

    print("\nDay 6 完成！明天加入 answer_question 和循环。")
