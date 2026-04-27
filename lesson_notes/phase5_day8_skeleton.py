"""
阶段五 · Day 8 任务：Map-Reduce（上）—— Send API + 子图嵌入主图

在 Day 7 的访谈子图基础上，搭建主图骨架：
1. ResearchGraphState 主图状态
2. create_analysts 节点（生成分析师团队）
3. initiate_all_interviews（Map 函数 + 条件路由）
4. 主图骨架（7 个节点 + 所有边）
5. Reduce 节点用占位函数（Day 9 补充）

参考源码：research_assistant.py 第 297-316 行、563-600 行、859-891 行、1103-1184 行
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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send  # ← Day 8 新增：动态并行的关键

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


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="用于检索的搜索查询语句")


class GenerateAnalystsState(TypedDict):
    """分析师生成阶段的状态（create_analysts 和 human_feedback 使用）"""
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]


class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list


class ResearchGraphState(TypedDict):
    """
    TODO: 定义主图状态

    字段：
    - topic: str                                    # 输入：研究主题
    - max_analysts: int                             # 输入：分析师数量
    - human_analyst_feedback: str                   # 人机协同：反馈
    - analysts: List[Analyst]                       # 中间：分析师列表
    - sections: Annotated[list, operator.add]       # 关键！并行子图输出自动累加
    - introduction: str                             # Reduce 输出
    - content: str                                  # Reduce 输出
    - conclusion: str                               # Reduce 输出
    - final_report: str                             # 最终输出

    参考源码：research_assistant.py 第 297-316 行
    """
    topic: str # 研究主题
    max_analysts: int # 最大分析师数量
    human_analyst_feedback: str # 生成分析师人类反馈
    analysts:List[Analyst] # 生成分析师列表
    sections:Annotated[list,operator.add] # map到各个子图后的输出
    introduction: str # reduce 输出，报告的介绍
    content: str # 报告内容
    conclusion: str # 报告结论
    final_report: str # 最终输出



# ============================================
# 提示词模板（Day 6-7 的 + 新增 analyst_instructions）
# ============================================

analyst_instructions = """你需要创建一组 AI 分析师人设。请严格遵循以下指引：

1. 先审阅研究主题：
{topic}

2. 查看（可选的）编辑反馈，它将指导分析师的人设创建：

{human_analyst_feedback}

3. 基于上述文档与/或反馈，识别最值得关注的主题。

4. 选出前 {max_analysts} 个主题。

5. 为每个主题分配一位分析师。"""

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

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
tavily_search = TavilySearchResults(max_results=3)


# ============================================
# 访谈子图节点（Day 6-7 已实现，原样搬入）
# ============================================

def generate_question(state: InterviewState):
    """分析师提问"""
    analyst = state["analyst"]
    messages = state["messages"]
    system_prompt = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    return {"messages": [question]}


def search_web(state: InterviewState):
    """Web 搜索"""
    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])
    search_docs = tavily_search.invoke(search_query.search_query)
    formatted_docs = "\n\n---\n\n".join([
        f'<Document href="{doc["url"]}" />\n{doc["content"]}\n</Document>'
        for doc in search_docs
    ])
    return {"context": [formatted_docs]}


def search_baike(state: InterviewState):
    """百科搜索"""
    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])
    search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()
    formatted_docs = "\n\n---\n\n".join([
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/> \n{doc.page_content}\n</Document>'
        for doc in search_docs
    ])
    return {"context": [formatted_docs]}


def generate_answer(state: InterviewState):
    """专家回答"""
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]
    system_messages = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_messages)] + messages)
    answer.name = "expert"
    return {"messages": [answer]}


def save_interview(state: InterviewState):
    """保存访谈记录"""
    messages = state["messages"]
    interview = get_buffer_string(messages=messages)
    return {"interview": interview}


def write_section(state: InterviewState):
    """生成报告小节"""
    context = state["context"]
    analyst = state["analyst"]
    system_messages = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([
        SystemMessage(content=system_messages),
        HumanMessage(content=f"使用这些来源撰写你的小节：{context}")
    ])
    return {"sections": [section.content]}


def route_messages(state: InterviewState, name: str = "expert"):
    """循环路由"""
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 2)
    number_response = 0
    for m in messages:
        if isinstance(m, AIMessage) and m.name == name:
            number_response += 1
    if number_response >= max_num_turns:
        return 'save_interview'
    last_question = messages[-2]
    if "非常感谢您的帮助!" in last_question.content:
        return 'save_interview'
    return 'ask_question'


# ============================================
# 编译访谈子图（Day 7 已完成）
# ============================================

interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_baike", search_baike)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_baike")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_baike", "answer_question")
interview_builder.add_conditional_edges(
    "answer_question",
    route_messages,
    ["ask_question", "save_interview"]
)
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

interview_graph = interview_builder.compile()
print("✅ 访谈子图编译完成")


# ============================================
# 主图节点（Day 8 新增）
# ============================================

def create_analysts(state: GenerateAnalystsState):
    """
    TODO: 生成分析师列表

    提示：
    1. 从 state 获取 topic, max_analysts, human_analyst_feedback
    2. 用 analyst_instructions.format(topic=..., max_analysts=..., human_analyst_feedback=...) 格式化
    3. structured_llm = llm.with_structured_output(Perspectives, method="function_calling")
    4. 调用 structured_llm.invoke([SystemMessage(content=提示词), HumanMessage(content="生成分析师")])
    5. 返回 {"analysts": result.analysts}

    参考源码：research_assistant.py 第 563-600 行
    """
    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback","")
    system_message = analyst_instructions.format(topic=topic,human_analyst_feedback=human_analyst_feedback,max_analysts=max_analysts)
    structured_llm = llm.with_structured_output(Perspectives,method="function_calling")
    result = structured_llm.invoke([SystemMessage(content=system_message),HumanMessage(content="生成分析师")])
    return {"analysts":result.analysts}


def human_feedback(state: GenerateAnalystsState):
    """
    人机协同节点（占位）

    这个节点本身不需要实现任何逻辑。
    编译时设置 interrupt_before=['human_feedback']，流程会在此暂停。
    人类通过 graph.update_state(config, {"human_analyst_feedback": "..."}) 注入反馈。
    """
    pass


def initiate_all_interviews(state: ResearchGraphState):
    """
    TODO: Map 函数 — 为每个分析师创建并行访谈任务

    提示：
    1. 检查 human_analyst_feedback，有内容则返回 "create_analysts"
    2. 无反馈时，获取 topic = state["topic"]
    3. 遍历 state["analysts"]
    4. 为每个 analyst 创建：
       Send("conduct_interview", {
           "analyst": analyst,
           "messages": [HumanMessage(content=f"所以你说你在写一篇关于{topic}的文章?")]
       })
    5. 返回 Send 列表

    参考源码：research_assistant.py 第 859-891 行
    """
    human_analyst_feedback = state["human_analyst_feedback"]
    if human_analyst_feedback:
        return "create_analysts"
    topic = state["topic"]
    return [
        Send("conduct_interview",{
                "analyst":analyst,
                "messages": [HumanMessage(content=f"所以说你在写一篇关于{topic}的文章?")]
            })
        for analyst in state["analysts"]
    ]


# Reduce 节点占位（Day 9 实现）

def write_report(state: ResearchGraphState):
    """占位：Day 9 实现报告主体"""
    return {"content": "报告主体（待 Day 9 实现）"}


def write_introduction(state: ResearchGraphState):
    """占位：Day 9 实现引言"""
    return {"introduction": "引言（待 Day 9 实现）"}


def write_conclusion(state: ResearchGraphState):
    """占位：Day 9 实现结论"""
    return {"conclusion": "结论（待 Day 9 实现）"}


def finalize_report(state: ResearchGraphState):
    """占位：Day 9 实现报告组装"""
    return {"final_report": "完整报告（待 Day 9 实现）"}


# ============================================
# 构建主图 —— 你来实现
# ============================================

builder = StateGraph(ResearchGraphState)

# TODO: 添加 7 个节点
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_graph)   # ← 子图嵌入！
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

# TODO: 添加边
# START → create_analysts
# create_analysts → human_feedback
# human_feedback → initiate_all_interviews (条件路由，返回 "create_analysts" 或 Send 列表)
# conduct_interview → write_report       (并行)
# conduct_interview → write_introduction (并行)
# conduct_interview → write_conclusion   (并行)
# [write_report, write_introduction, write_conclusion] → finalize_report  (fan-in)
# finalize_report → END

builder.add_edge(START,"create_analysts")
builder.add_edge("create_analysts","human_feedback")
builder.add_conditional_edges("human_feedback",initiate_all_interviews,["create_analysts","conduct_interview"])
builder.add_edge("conduct_interview","write_report")
builder.add_edge("conduct_interview","write_introduction")
builder.add_edge("conduct_interview","write_conclusion")
builder.add_edge(["write_report","write_introduction","write_conclusion"],"finalize_report")
builder.add_edge("finalize_report",END)

# TODO: 编译
memory = MemorySaver()
graph = builder.compile(
    interrupt_before=['human_feedback'],
    checkpointer=memory
)


# ============================================
# 测试
# ============================================

if __name__ == "__main__":
    print("=== 阶段五 Day 8: Map-Reduce 主图骨架 ===\n")

    # 测试 1: 验证 initiate_all_interviews 返回 Send 列表
    print("--- 测试 initiate_all_interviews (无反馈) ---")
    test_state = {
        "topic": "AI在医疗中的应用",
        "analysts": [
            Analyst(affiliation="协和医院", name="张伟博士", role="医疗AI专家",
                    description="关注AI在影像诊断中的应用"),
            Analyst(affiliation="清华", name="李明教授", role="数据科学家",
                    description="关注医疗数据的隐私保护"),
        ],
        "human_analyst_feedback": "",
    }

    result = initiate_all_interviews(test_state)
    if isinstance(result, list):
        print(f"✅ 返回 {len(result)} 个 Send 对象")
        for i, send in enumerate(result):
            print(f"   Send[{i}]: 节点={send.node}, analyst={send.arg['analyst'].name}")
    else:
        print(f"返回: {result}")

    # 测试 2: 有反馈时返回字符串
    print("\n--- 测试 initiate_all_interviews (有反馈) ---")
    test_state["human_analyst_feedback"] = "请增加一个关注伦理的分析师"
    result = initiate_all_interviews(test_state)
    print(f"✅ 有反馈时返回: {result}")

    # 测试 3: 完整主图运行（需要三步：invoke → update_state → invoke）
    # Day 9 实现 Reduce 节点后再测试
    # config = {"configurable": {"thread_id": "test_001"}}
    # result = graph.invoke({"topic": "AI在医疗中的应用", "max_analysts": 2}, config)
    # graph.update_state(config, {"human_analyst_feedback": ""}, as_node="human_feedback")
    # result = graph.invoke(None, config)
    # print(f"最终报告:\n{result['final_report']}")

    print("\nDay 8 骨架搭建完成！明天实现 Reduce 节点。")
