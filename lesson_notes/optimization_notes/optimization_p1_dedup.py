"""
P1 优化骨架：搜索结果 URL 去重

在 search_web 和 search_baike 节点中加入 URL 去重逻辑。
基于 lesson_notes/phase5_day9_skeleton.py 的改动版本。
"""

import os
import re
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
from langgraph.types import Send

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
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]


# ===== P1 改动 1：InterviewState 新增 seen_urls 字段 =====
class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list
    seen_urls: Annotated[list, operator.add]  # 新增：已见过的 URL 列表


class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str


# ============================================
# 提示词模板
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


# ===== P1 改动 2：URL 提取工具函数 =====
def extract_urls_from_context(context_list: list) -> set:
    """从 context 列表中提取所有已见过的 URL。"""
    urls = set()
    for ctx in context_list:
        # 匹配 href="..." 和 source="..." 属性
        found = re.findall(r'href="([^"]+)"', ctx)
        found += re.findall(r'source="([^"]+)"', ctx)
        urls.update(found)
    return urls


# ============================================
# 访谈子图节点
# ============================================

def generate_question(state: InterviewState):
    analyst = state["analyst"]
    messages = state["messages"]
    system_prompt = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    return {"messages": [question]}


# ===== P1 改动 3：search_web 加入去重 =====
def search_web(state: InterviewState):
    """搜索网页，过滤已见过的 URL。"""
    # 提取已见过的 URL
    seen_urls = extract_urls_from_context(state.get("context", []))

    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    try:
        search_docs = tavily_search.invoke(search_query.search_query)

        # 去重过滤
        new_docs = []
        new_urls = []
        for doc in search_docs:
            url = doc.get("url", "") if isinstance(doc, dict) else ""
            if url and url not in seen_urls:
                new_docs.append(doc)
                new_urls.append(url)

        if new_docs:
            formatted_docs = "\n\n---\n\n".join([
                f'<Document href="{doc["url"]}" />\n{doc["content"]}\n</Document>'
                if isinstance(doc, dict) else f'<Document />\n{doc}\n</Document>'
                for doc in new_docs
            ])
        else:
            formatted_docs = f"<Document />搜索结果的URL均已见过，查询: {search_query.search_query}，请尝试换不同的搜索词</Document>"

    except Exception as e:
        print(f"  [search_web 降级] {e}")
        formatted_docs = f"<Document />Web搜索暂不可用，查询: {search_query.search_query}\n</Document>"

    return {"context": [formatted_docs], "seen_urls": new_urls}


# ===== P1 改动 4：search_baike 加入去重 =====
def search_baike(state: InterviewState):
    """搜索百科，过滤已见过的 URL。"""
    seen_urls = extract_urls_from_context(state.get("context", []))

    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    try:
        search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()

        new_docs = []
        new_urls = []
        for doc in search_docs:
            url = doc.metadata.get("source", "")
            if url and url not in seen_urls:
                new_docs.append(doc)
                new_urls.append(url)

        if new_docs:
            formatted_docs = "\n\n---\n\n".join([
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/> \n{doc.page_content}\n</Document>'
                for doc in new_docs
            ])
        else:
            formatted_docs = f"<Document />百科搜索结果的URL均已见过，查询: {search_query.search_query}，请尝试换不同的搜索词</Document>"

    except Exception as e:
        print(f"  [search_baike 降级] {e}")
        formatted_docs = f"<Document />百科搜索暂不可用，查询: {search_query.search_query}\n</Document>"

    return {"context": [formatted_docs], "seen_urls": new_urls}


def generate_answer(state: InterviewState):
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]
    system_messages = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_messages)] + messages)
    answer.name = "expert"
    return {"messages": [answer]}


def reflect(state: InterviewState):
    """分析师在每轮访谈后进行元认知评估。"""
    messages = state["messages"]
    response = llm.invoke([
        SystemMessage(content="""你是一名分析师，刚刚完成一轮专家访谈。
请评估：我获得了哪些关键信息？还缺少什么？是否已有足够洞见来写报告小节？
如果信息已充分，请以「信息已充分，可以结束访谈」结尾。"""),
        HumanMessage(content="请评估当前访谈进展。")
    ])
    should_end = any(keyword in response.content for keyword in [
        "信息已充分", "可以结束", "足够", "sufficient", "enough"
    ])
    return {
        "messages": [AIMessage(content=response.content, name="reflection")],
        "should_end_reflection": should_end
    }


def route_after_reflect(state: InterviewState):
    """基于反思决定继续还是结束访谈。"""
    messages = state["messages"]
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.name == "reflection":
            if any(keyword in m.content for keyword in [
                "信息已充分", "可以结束", "足够", "sufficient", "enough"
            ]):
                return "save_interview"
    max_num_turns = state.get("max_num_turns", 2)
    expert_responses = sum(1 for m in messages if isinstance(m, AIMessage) and m.name == "expert")
    if expert_responses >= max_num_turns:
        return "save_interview"
    return "ask_question"


def save_interview(state: InterviewState):
    messages = state["messages"]
    interview = get_buffer_string(messages=messages)
    return {"interview": interview}


def write_section(state: InterviewState):
    context = state["context"]
    analyst = state["analyst"]
    system_messages = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([
        SystemMessage(content=system_messages),
        HumanMessage(content=f"使用这些来源撰写你的小节：{context}")
    ])
    return {"sections": [section.content]}


# ============================================
# 编译访谈子图
# ============================================

interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_baike", search_baike)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("reflect", reflect)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_baike")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_baike", "answer_question")
interview_builder.add_edge("answer_question", "reflect")
interview_builder.add_conditional_edges(
    "reflect",
    route_after_reflect,
    ["ask_question", "save_interview"]
)
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

interview_graph = interview_builder.compile()
print("✅ 访谈子图编译完成（含 URL 去重 + reflect 节点）")


# ============================================
# 主图节点（与原版相同，省略重复定义）
# ============================================

def create_analysts(state: GenerateAnalystsState):
    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")
    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts
    )
    structured_llm = llm.with_structured_output(Perspectives, method="function_calling")
    result = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="生成分析师")
    ])
    return {"analysts": result.analysts}


def human_feedback(state: GenerateAnalystsState):
    pass


def initiate_all_interviews(state: ResearchGraphState):
    human_analyst_feedback = state["human_analyst_feedback"]
    if human_analyst_feedback:
        return "create_analysts"
    topic = state["topic"]
    return [
        Send("conduct_interview", {
            "analyst": analyst,
            "messages": [HumanMessage(content=f"所以你说你在写一篇关于{topic}的文章?")]
        })
        for analyst in state["analysts"]
    ]


def write_report(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    context = "\n\n".join(str(s) for s in sections)
    report_writer_instructions = """你是一名技术写作者，正在为如下主题撰写报告：

{topic}

你拥有一支分析师团队。每位分析师完成了两件事：

1. 围绕一个具体子主题，访谈了一位专家。
2. 将发现写成一份备忘录（memo）。

你的任务：将备忘录整合为简洁的总体总结，使用 Markdown 格式，全部使用中文。

以下是分析师提供的备忘录：

{context}"""
    system_messages = report_writer_instructions.format(topic=topic, context=context)
    report = llm.invoke([SystemMessage(content=system_messages), HumanMessage(content="基于这些备忘录撰写一份报告。")])
    return {"content": report.content}


def write_introduction(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    context = "\n\n".join(str(s) for s in sections)
    intro_conclusion_instructions = """你是一名技术写作者，正在完成主题为 {topic} 的报告。
你将获得报告的全部小节。你的任务是撰写简洁而有说服力的引言。
目标约 100 字，使用 Markdown 格式，全部使用中文。
撰写时可参考以下小节内容：{formatted_str_sections}"""
    system_messages = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=context)
    introduction = llm.invoke([SystemMessage(content=system_messages), HumanMessage(content="撰写报告引言")])
    return {"introduction": introduction.content}


def write_conclusion(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    context = "\n\n".join(str(s) for s in sections)
    intro_conclusion_instructions = """你是一名技术写作者，正在完成主题为 {topic} 的报告。
你将获得报告的全部小节。你的任务是撰写简洁而有说服力的结论。
目标约 100 字，使用 Markdown 格式，全部使用中文。
撰写时可参考以下小节内容：{formatted_str_sections}"""
    system_messages = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=context)
    conclusion = llm.invoke([SystemMessage(content=system_messages), HumanMessage(content="撰写报告结论")])
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    content = state["content"]
    final_report = (
        state.get("introduction", "") +
        "\n\n---\n\n" +
        content +
        "\n\n---\n\n" +
        state.get("conclusion", "")
    )
    return {"final_report": final_report}


# ============================================
# 构建主图
# ============================================

builder = StateGraph(ResearchGraphState)

builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_graph)
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges(
    "human_feedback",
    initiate_all_interviews,
    ["create_analysts", "conduct_interview"]
)
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(
    ["write_report", "write_introduction", "write_conclusion"],
    "finalize_report"
)
builder.add_edge("finalize_report", END)

memory = MemorySaver()
graph = builder.compile(
    interrupt_before=['human_feedback'],
    checkpointer=memory
)
print("✅ 主图编译完成（含 URL 去重）")


if __name__ == "__main__":
    topic = "AI在医疗领域的应用与挑战"
    max_analysts = 2

    print(f"=== P1 优化测试（URL 去重）：{topic} ===\n")
    config = {"configurable": {"thread_id": "test_p1"}}

    result = graph.invoke({"topic": topic, "max_analysts": max_analysts}, config)
    analysts = result["analysts"]
    print(f"生成了 {len(analysts)} 个分析师")

    graph.update_state(config, {"human_analyst_feedback": ""}, as_node="human_feedback")

    for event in graph.stream(None, config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "conduct_interview":
                urls = node_output.get("seen_urls", [])
                if urls:
                    print(f"  [去重] 新发现 {len(urls)} 个唯一URL: {urls[:3]}...")
            elif node_name == "finalize_report":
                print(f"  最终报告 ({len(node_output.get('final_report', ''))} 字符)")

    print("\n✅ P1 优化测试完成！")
