"""
P0 优化骨架：think_tool 反思停顿节点

在访谈子图中插入 reflect 节点，让分析师动态决定是否继续访谈。
基于 lesson_notes/phase5_day9_skeleton.py 的改动版本。
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


# ===== P0 改动 1：InterviewState 新增 should_end_reflection 字段 =====
class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list
    should_end_reflection: bool = False  # 新增：反思节点的决定


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


# ===== P0 改动 2：新增反思提示词 =====
reflect_instructions = """你是一名分析师，刚刚完成一轮专家访谈。

请评估当前进展，回答以下问题：
1. 我获得了哪些关键信息？
2. 还缺少什么重要信息？
3. 我已经有了足够的洞见来写报告小节吗？
4. 我应该继续追问还是结束访谈？

如果信息已充分，请以「信息已充分，可以结束访谈」结尾。
如果还需要更多信息，请以「需要继续追问」结尾。

保持反思简洁（约 100 字）。"""


# ============================================
# LLM 和搜索工具
# ============================================

llm = ChatOpenAI(model=model, temperature=0, base_url=base_url, api_key=api_key)

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
tavily_search = TavilySearchResults(max_results=3)


# ============================================
# 访谈子图节点
# ============================================

def generate_question(state: InterviewState):
    analyst = state["analyst"]
    messages = state["messages"]
    system_prompt = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    return {"messages": [question]}


def search_web(state: InterviewState):
    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])
    try:
        search_docs = tavily_search.invoke(search_query.search_query)
        formatted_docs = "\n\n---\n\n".join([
            f'<Document href="{doc["url"]}" />\n{doc["content"]}\n</Document>'
            if isinstance(doc, dict) else f'<Document />\n{doc}\n</Document>'
            for doc in search_docs
        ])
    except Exception as e:
        print(f"  [search_web 降级] {e}")
        formatted_docs = f"<Document />Web搜索暂不可用，查询: {search_query.search_query}\n</Document>"
    return {"context": [formatted_docs]}


def search_baike(state: InterviewState):
    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])
    try:
        search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()
        formatted_docs = "\n\n---\n\n".join([
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/> \n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    except Exception as e:
        print(f"  [search_baike 降级] {e}")
        formatted_docs = f"<Document />百科搜索暂不可用，查询: {search_query.search_query}\n</Document>"
    return {"context": [formatted_docs]}


def generate_answer(state: InterviewState):
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]
    system_messages = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_messages)] + messages)
    answer.name = "expert"
    return {"messages": [answer]}


# ===== P0 改动 3：新增 reflect 节点 =====
def reflect(state: InterviewState):
    """分析师在每轮访谈后进行元认知评估，动态决定是否继续访谈。"""
    messages = state["messages"]

    response = llm.invoke([
        SystemMessage(content=reflect_instructions),
        HumanMessage(content="请评估当前访谈进展。")
    ])

    # 判断是否应该结束
    should_end = any(keyword in response.content for keyword in [
        "信息已充分", "可以结束", "足够", "sufficient", "enough"
    ])

    return {
        "messages": [AIMessage(content=response.content, name="reflection")],
        "should_end_reflection": should_end
    }


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


def route_messages(state: InterviewState, name: str = "expert"):
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


# ===== P0 改动 4：新增 route_after_reflect 路由函数 =====
def route_after_reflect(state: InterviewState):
    """基于反思决定继续还是结束访谈。"""
    messages = state["messages"]

    # 1. 检查反思节点的决定
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.name == "reflection":
            if any(keyword in m.content for keyword in [
                "信息已充分", "可以结束", "足够", "sufficient", "enough"
            ]):
                return "save_interview"

    # 2. 兜底：检查最大轮数
    max_num_turns = state.get("max_num_turns", 2)
    expert_responses = sum(1 for m in messages if isinstance(m, AIMessage) and m.name == "expert")
    if expert_responses >= max_num_turns:
        return "save_interview"

    # 3. 继续追问
    return "ask_question"


# ============================================
# 编译访谈子图
# ============================================

interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_baike", search_baike)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("reflect", reflect)           # 新增
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_baike")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_baike", "answer_question")
interview_builder.add_edge("answer_question", "reflect")  # 改为先到反思
interview_builder.add_conditional_edges(
    "reflect",
    route_after_reflect,
    ["ask_question", "save_interview"]
)
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

interview_graph = interview_builder.compile()
print("✅ 访谈子图编译完成（含 reflect 反思节点）")


# ============================================
# 主图节点
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


# Reduce 节点（保持不变）
def write_report(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    context = "\n\n".join(f"{section}" for section in sections)
    from langchain_core.messages import SystemMessage, HumanMessage
    report_writer_instructions = """你是一名技术写作者，正在为如下主题撰写报告：

{topic}

你拥有一支分析师团队。每位分析师完成了两件事：

1. 围绕一个具体子主题，访谈了一位专家。
2. 将发现写成一份备忘录（memo）。

你的任务：

1. 你将收到分析师们的备忘录集合。
2. 仔细思考每份备忘录的洞见。
3. 将它们整合为简洁的总体总结，串联起所有备忘录的中心观点。
4. 把每份备忘录的关键信息归纳成一个连贯的单一叙述。

**重要要求：生成的报告必须全部使用中文。**

报告格式要求：

1. 使用 Markdown 格式。
2. 报告不要有任何前言。
3. 不使用任何小标题。
4. 报告以一个标题开头：## Insights
5. 报告中不要提及任何分析师的名字。
6. 保留备忘录中的引用标注（如 [1]、[2]）。
7. 汇总最终来源列表，并以 `## Sources` 作为小节标题。
8. 按顺序列出来源且不要重复。

[1] Source 1
[2] Source 2

以下是分析师提供的备忘录，请基于此撰写报告：

{context}"""
    system_messages = report_writer_instructions.format(topic=topic, context=context)
    report = llm.invoke([SystemMessage(content=system_messages), HumanMessage(content="基于这些备忘录撰写一份报告。")])
    return {"content": report.content}


def write_introduction(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    context = "\n\n".join(f"{section}" for section in sections)
    intro_conclusion_instructions = """你是一名技术写作者，正在完成主题为 {topic} 的报告。

你将获得报告的全部小节。

你的任务是撰写简洁而有说服力的引言或结论。

由用户告知写引言还是结论。

两者均不需要任何前言。

目标约 100 字：
- 引言：精炼预览各小节要点
- 结论：精炼回顾各小节要点

使用 Markdown 格式。

**重要要求：全部使用中文。**

引言要求：创建一个有吸引力的标题，并用 # 作为标题头。

引言小节标题使用：## 引言

结论小节标题使用：## 结论

撰写时可参考以下小节内容：{formatted_str_sections}"""
    system_messages = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=context)
    introduction = llm.invoke([SystemMessage(content=system_messages), HumanMessage(content="撰写报告引言")])
    return {"introduction": introduction.content}


def write_conclusion(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    context = "\n\n".join(f"{section}" for section in sections)
    intro_conclusion_instructions = """你是一名技术写作者，正在完成主题为 {topic} 的报告。

你将获得报告的全部小节。

你的任务是撰写简洁而有说服力的引言或结论。

由用户告知写引言还是结论。

两者均不需要任何前言。

目标约 100 字：
- 引言：精炼预览各小节要点
- 结论：精炼回顾各小节要点

使用 Markdown 格式。

**重要要求：全部使用中文。**

引言要求：创建一个有吸引力的标题，并用 # 作为标题头。

引言小节标题使用：## 引言

结论小节标题使用：## 结论

撰写时可参考以下小节内容：{formatted_str_sections}"""
    system_messages = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=context)
    conclusion = llm.invoke([SystemMessage(content=system_messages), HumanMessage(content="撰写报告结论")])
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Insights" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = (
        state["introduction"] +
        "\n\n---\n\n" +
        content +
        "\n\n---\n\n" +
        state["conclusion"]
    )
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources

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
print("✅ 主图编译完成（含 reflect 反思节点）")


# ============================================
# 测试
# ============================================

if __name__ == "__main__":
    topic = "AI在医疗领域的应用与挑战"
    max_analysts = 2

    print(f"=== P0 优化测试：{topic} ===\n")

    config = {"configurable": {"thread_id": "test_p0"}}

    print("--- 第1步：生成分析师 ---")
    result = graph.invoke({
        "topic": topic,
        "max_analysts": max_analysts
    }, config)

    analysts = result["analysts"]
    print(f"生成了 {len(analysts)} 个分析师：")
    for a in analysts:
        print(f"  - {a.name}（{a.affiliation}）: {a.description[:40]}...")

    print("\n--- 第2步：批准分析师 ---")
    graph.update_state(config, {"human_analyst_feedback": ""}, as_node="human_feedback")

    print("\n--- 第3步：Map-Reduce 全流程（含反思节点）---")
    for event in graph.stream(None, config, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n📍 节点: {node_name}")
            if node_name == "conduct_interview":
                messages = node_output.get("messages", [])
                # 打印反思消息
                for msg in messages:
                    if isinstance(msg, AIMessage) and msg.name == "reflection":
                        print(f"  [反思] {msg.content[:150]}...")
            elif node_name == "finalize_report":
                report = node_output.get("final_report", "")
                print(f"  完整报告 ({len(report)} 字符)")

    result = graph.get_state(config).values
    print("\n" + "=" * 60)
    print("📋 最终报告")
    print("=" * 60)
    print(result.get("final_report", ""))
    print(f"\n✅ P0 优化测试完成！")
