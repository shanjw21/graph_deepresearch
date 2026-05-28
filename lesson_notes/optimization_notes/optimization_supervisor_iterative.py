"""
进阶优化骨架：Supervisor 迭代委派架构

在 Map-Reduce 基础上加入 Supervisor 循环评估：
1. 完成所有分析师访谈后，LLM 评估覆盖度
2. 如果有缺失角度，动态生成补充分析师
3. 最多迭代 N 轮（可配置）

基于 lesson_notes/phase5_day9_skeleton.py 的完整重构版本。
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


class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list
    seen_urls: Annotated[list, operator.add]


# ===== Supervisor 迭代架构：ResearchGraphState 新增字段 =====
class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]        # 当前轮次的新 sections
    all_sections: Annotated[list, operator.add]    # 累积的所有 sections
    supervisor_iterations: int                     # 当前迭代次数
    max_supervisor_iterations: int                 # 最大迭代次数
    coverage_gaps: str                             # 评估发现的缺失角度
    introduction: str
    content: str
    conclusion: str
    final_report: str


# ============================================
# 提示词模板
# ============================================

language_instruction = """
**重要：请检测用户输入的语言，并使用相同的语言输出报告。**
- 如果用户输入是中文，用中文写报告
- 如果用户输入是英文，用英文写报告
- 如果用户输入是其他语言，也用该语言写报告
"""

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

{language_instruction}

在参考来源部分列出使用到的全部来源，合并重复来源。"""

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

{language_instruction}

报告格式要求：

1. 使用 Markdown 格式。
2. 报告不要有任何前言。
3. 不使用任何小标题。
4. 报告以一个标题开头：## Insights
5. 报告中不要提及任何分析师的名字。
6. 保留备忘录中的引用标注（如 [1]、[2]）。
7. 汇总最终来源列表，并以 `## Sources` 作为小节标题。
8. 按顺序列出来源且不要重复。

以下是分析师提供的备忘录：

{context}"""

intro_conclusion_instructions = """你是一名技术写作者，正在完成主题为 {topic} 的报告。

你将获得报告的全部小节。

你的任务是撰写简洁而有说服力的引言或结论。

由用户告知写引言还是结论。

两者均不需要任何前言。

目标约 100 字：
- 引言：精炼预览各小节要点
- 结论：精炼回顾各小节要点

使用 Markdown 格式。

{language_instruction}

引言小节标题使用：## 引言

结论小节标题使用：## 结论

撰写时可参考以下小节内容：{formatted_str_sections}"""


# ===== Supervisor 迭代专用提示词 =====
evaluate_coverage_prompt = """你是一名研究质量评估专家。

研究主题：{topic}

已有的研究小节：
{sections}

请评估：
1. 已有小节覆盖了主题的哪些关键方面？
2. 还缺少哪些重要角度？
3. 是否需要补充研究？

如果需要补充研究，请以「需要补充：[具体角度]」格式列出缺失角度（最多 2 个）。
如果覆盖已充分，请回复「覆盖已充分，可以进入报告撰写阶段」。

注意：评估标准应该是"足够好"而不是"完美"。如果已有小节覆盖了主题的 70% 以上关键方面，就认为覆盖已充分。"""

delegate_more_prompt = """你是一名研究主管。现有研究对主题「{topic}」的分析还缺少以下角度：

{coverage_gaps}

请生成 1-2 位新的分析师来补充这些缺失的角度。每位分析师应专注于一个具体方面。

研究主题：{topic}"""


# ============================================
# LLM 和搜索工具
# ============================================

llm = ChatOpenAI(model=model, temperature=0, base_url=base_url, api_key=api_key)

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
tavily_search = TavilySearchResults(max_results=3)


# ============================================
# 工具函数
# ============================================

def extract_urls_from_context(context_list: list) -> set:
    """从 context 列表中提取所有已见过的 URL。"""
    urls = set()
    for ctx in context_list:
        found = re.findall(r'href="([^"]+)"', ctx)
        found += re.findall(r'source="([^"]+)"', ctx)
        urls.update(found)
    return urls


def summarize_content(content: str, max_length: int = 300) -> str:
    """用 LLM 将长网页内容压缩为简短摘要。"""
    if len(content) < max_length:
        return content
    truncated = content[:4000]
    system_prompt = f"请将以下内容压缩为{max_length}字以内的摘要。提取主要论点、关键数据和事实：\n{truncated}"
    summary = llm.invoke([
        SystemMessage(content="你是一名研究助手，需要将网页内容压缩为简短摘要。"),
        HumanMessage(content=system_prompt)
    ])
    return summary.content[:max_length]


# ============================================
# 访谈子图节点（含 P0 反思节点）
# ============================================

def generate_question(state: InterviewState):
    analyst = state["analyst"]
    messages = state["messages"]
    system_prompt = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    return {"messages": [question]}


def search_web(state: InterviewState):
    """搜索网页，去重 + 摘要。"""
    seen_urls = extract_urls_from_context(state.get("context", []))
    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    try:
        search_docs = tavily_search.invoke(search_query.search_query)
        new_docs, new_urls = [], []
        for doc in search_docs:
            url = doc.get("url", "") if isinstance(doc, dict) else ""
            if url and url not in seen_urls:
                new_docs.append(doc)
                new_urls.append(url)

        summarized_docs = []
        for doc in new_docs:
            title = doc.get("title", "未知来源")
            content = doc.get("content", "")
            if len(content) > 500:
                content = summarize_content(content, max_length=300)
            summarized_docs.append(
                f'<Document href="{doc.get("url", "")}" title="{title}">\n{content}\n</Document>'
            )

        formatted_docs = "\n\n---\n\n".join(summarized_docs) if summarized_docs else \
            f"<Document />搜索结果的URL均已见过</Document>"
    except Exception as e:
        formatted_docs = f"<Document />Web搜索暂不可用，查询: {search_query.search_query}\n</Document>"

    return {"context": [formatted_docs], "seen_urls": new_urls}


def search_baike(state: InterviewState):
    """搜索百科，去重 + 摘要。"""
    seen_urls = extract_urls_from_context(state.get("context", []))
    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    try:
        search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()
        new_docs, new_urls = [], []
        for doc in search_docs:
            url = doc.metadata.get("source", "")
            if url and url not in seen_urls:
                new_docs.append(doc)
                new_urls.append(url)

        summarized_docs = []
        for doc in new_docs:
            content = doc.page_content
            if len(content) > 500:
                content = summarize_content(content, max_length=300)
            summarized_docs.append(
                f'<Document source="{doc.metadata.get("source", "")}" page="{doc.metadata.get("page", "")}">\n{content}\n</Document>'
            )

        formatted_docs = "\n\n---\n\n".join(summarized_docs) if summarized_docs else \
            f"<Document />百科搜索结果的URL均已见过</Document>"
    except Exception as e:
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
        SystemMessage(content="你是一名分析师，刚刚完成一轮专家访谈。请评估：我获得了哪些关键信息？还缺少什么？是否已有足够洞见来写报告小节？如果信息已充分，请以「信息已充分，可以结束访谈」结尾。"),
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
    system_messages = section_writer_instructions.format(
        focus=analyst.description,
        language_instruction=language_instruction
    )
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
print("✅ 访谈子图编译完成（含 P0/P1/P2 全部优化）")


# ============================================
# 主图节点 — Map 部分
# ============================================

def create_analysts(state: ResearchGraphState):
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


def human_feedback(state: ResearchGraphState):
    pass


def initiate_all_interviews(state: ResearchGraphState):
    """决定是重新生成分析师还是启动访谈。"""
    human_analyst_feedback = state.get("human_analyst_feedback", "")
    if human_analyst_feedback and human_analyst_feedback.strip():
        return "create_analysts"
    topic = state["topic"]
    return [
        Send("conduct_interview", {
            "analyst": analyst,
            "messages": [HumanMessage(content=f"你好，我在写一篇关于{topic}的文章，想请教你一些相关问题。")]
        })
        for analyst in state["analysts"]
    ]


# ============================================
# 主图节点 — Supervisor 迭代评估
# ============================================

def evaluate_coverage(state: ResearchGraphState):
    """
    Supervisor 评估已有 sections 是否覆盖了主题的关键方面。

    返回评估结果和迭代计数。
    """
    sections = state.get("sections", [])
    all_sections = state.get("all_sections", [])
    combined = all_sections + sections
    sections_text = "\n\n".join(str(s) for s in combined)

    if not sections_text:
        sections_text = "暂无，尚未完成任何研究"

    prompt = evaluate_coverage_prompt.format(
        topic=state["topic"],
        sections=sections_text
    )

    response = llm.invoke([
        SystemMessage(content="你是一名研究质量评估专家，负责评估研究覆盖度。"),
        HumanMessage(content=prompt)
    ])

    needs_more = "需要补充" in response.content
    gaps = response.content if needs_more else ""

    return {
        "coverage_gaps": gaps,
        "supervisor_iterations": state.get("supervisor_iterations", 0) + 1
    }


def delegate_more_analysts(state: ResearchGraphState):
    """
    基于评估发现的缺失角度，生成补充分析师。
    """
    gaps = state.get("coverage_gaps", "")

    structured_llm = llm.with_structured_output(Perspectives, method="function_calling")
    result = structured_llm.invoke([
        SystemMessage(content=delegate_more_prompt.format(
            topic=state["topic"],
            coverage_gaps=gaps
        )),
        HumanMessage(content="生成1-2位补充分析师")
    ])

    print(f"  [Supervisor] 补充分析师：{len(result.analysts)} 位")
    for a in result.analysts:
        print(f"    - {a.name}（{a.affiliation}）: {a.description[:40]}...")

    return {
        "analysts": result.analysts
    }


def route_after_evaluate(state: ResearchGraphState):
    """基于评估结果决定下一步。"""
    iterations = state.get("supervisor_iterations", 0)
    max_iter = state.get("max_supervisor_iterations", 2)

    if iterations >= max_iter:
        print(f"  [Supervisor] 达到最大迭代次数 {max_iter}，进入报告撰写")
        return "enough_coverage"

    gaps = state.get("coverage_gaps", "")
    if "需要补充" in gaps:
        print(f"  [Supervisor] 发现缺失角度，启动补充研究")
        return "needs_more"

    print(f"  [Supervisor] 覆盖已充分，进入报告撰写")
    return "enough_coverage"


def route_after_interview(state: ResearchGraphState):
    """访谈完成后决定是进入评估还是直接进入报告。"""
    # 第一次访谈（iterations == 0）直接进入评估
    # 补充访谈后也进入评估
    return "evaluate_coverage"


# ============================================
# 主图节点 — Reduce 部分
# ============================================

def write_report(state: ResearchGraphState):
    sections = state.get("all_sections", []) + state.get("sections", [])
    topic = state["topic"]
    context = "\n\n".join(str(s) for s in sections)
    system_messages = report_writer_instructions.format(
        topic=topic,
        context=context,
        language_instruction=language_instruction
    )
    report = llm.invoke([
        SystemMessage(content=system_messages),
        HumanMessage(content="基于这些备忘录撰写一份报告。")
    ])
    return {"content": report.content}


def write_introduction(state: ResearchGraphState):
    sections = state.get("all_sections", []) + state.get("sections", [])
    topic = state["topic"]
    context = "\n\n".join(str(s) for s in sections)
    system_messages = intro_conclusion_instructions.format(
        topic=topic,
        formatted_str_sections=context,
        language_instruction=language_instruction
    )
    introduction = llm.invoke([
        SystemMessage(content=system_messages),
        HumanMessage(content="撰写报告引言")
    ])
    return {"introduction": introduction.content}


def write_conclusion(state: ResearchGraphState):
    sections = state.get("all_sections", []) + state.get("sections", [])
    topic = state["topic"]
    context = "\n\n".join(str(s) for s in sections)
    system_messages = intro_conclusion_instructions.format(
        topic=topic,
        formatted_str_sections=context,
        language_instruction=language_instruction
    )
    conclusion = llm.invoke([
        SystemMessage(content=system_messages),
        HumanMessage(content="撰写报告结论")
    ])
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    content = state["content"]
    introduction = state.get("introduction", "")
    conclusion = state.get("conclusion", "")

    final_report = introduction + "\n\n---\n\n" + content + "\n\n---\n\n" + conclusion
    return {"final_report": final_report}


# ============================================
# 构建主图（含 Supervisor 循环）
# ============================================

builder = StateGraph(ResearchGraphState)

# 添加节点
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_graph)
builder.add_node("evaluate_coverage", evaluate_coverage)
builder.add_node("delegate_more_analysts", delegate_more_analysts)
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

# 初始边
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")

# 条件边：human_feedback → create_analysts 或 conduct_interview
builder.add_conditional_edges(
    "human_feedback",
    initiate_all_interviews,
    ["create_analysts", "conduct_interview"]
)

# 访谈完成后进入评估
builder.add_edge("conduct_interview", "evaluate_coverage")

# ===== 核心：Supervisor 循环 =====
builder.add_conditional_edges(
    "evaluate_coverage",
    route_after_evaluate,
    {
        "needs_more": "delegate_more_analysts",
        "enough_coverage": "write_report"
    }
)

# 补充分析师 → 再次访谈（形成循环）
builder.add_edge("delegate_more_analysts", "conduct_interview")

# Reduce 并行节点（在 evaluate_coverage → enough_coverage 后触发）
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
print("✅ 主图编译完成（含 Supervisor 迭代委派架构）")
print("   架构: create_analysts → human_feedback → conduct_interview → evaluate_coverage")
print("          ↕ (needs_more)")
print("   循环: delegate_more_analysts → conduct_interview → evaluate_coverage")
print("          ↓ (enough_coverage)")
print("   完成: write_report/intro/conclusion → finalize_report → END")


# ============================================
# 端到端测试
# ============================================

if __name__ == "__main__":
    topic = "AI在医疗领域的应用与挑战"
    max_analysts = 2
    max_supervisor_iterations = 2

    print(f"=== Supervisor 迭代委派测试：{topic} ===\n")
    print(f"   初始分析师: {max_analysts}")
    print(f"   最大迭代次数: {max_supervisor_iterations}\n")

    config = {"configurable": {"thread_id": "test_supervisor"}}

    # 第1步：生成分析师
    print("--- 第1步：生成分析师 ---")
    result = graph.invoke({
        "topic": topic,
        "max_analysts": max_analysts,
        "max_supervisor_iterations": max_supervisor_iterations
    }, config)

    print(f"\n生成了 {len(result['analysts'])} 个分析师：")
    for a in result["analysts"]:
        print(f"  - {a.name}（{a.affiliation}）")

    # 第2步：批准
    print("\n--- 第2步：批准分析师，启动 Supervisor 循环 ---")
    graph.update_state(config, {"human_analyst_feedback": ""}, as_node="human_feedback")

    # 第3步：运行
    print("\n--- 第3步：Supervisor 迭代流程 ---")
    iteration_count = 0
    for event in graph.stream(None, config, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n{'='*50}")
            print(f"📍 节点: {node_name}")
            print(f"{'='*50}")

            if node_name == "conduct_interview":
                sections = node_output.get("sections", [])
                print(f"  生成 {len(sections)} 个小节")
                for s in sections:
                    print(f"    {str(s)[:100]}...")

            elif node_name == "evaluate_coverage":
                gaps = node_output.get("coverage_gaps", "")
                iters = node_output.get("supervisor_iterations", 0)
                iteration_count = iters
                print(f"  迭代 #{iters}")
                if gaps:
                    print(f"  缺失角度: {gaps[:200]}...")
                else:
                    print(f"  覆盖已充分")

            elif node_name == "delegate_more_analysts":
                new_analysts = node_output.get("analysts", [])
                print(f"  新增 {len(new_analysts)} 个补充分析师")

            elif node_name == "finalize_report":
                report = node_output.get("final_report", "")
                print(f"  完整报告 ({len(report)} 字符)")

            else:
                for k, v in node_output.items():
                    if isinstance(v, list):
                        print(f"  {k}: [{len(v)} 项]")
                    elif isinstance(v, str):
                        print(f"  {k}: {v[:100]}...")

    # 获取最终状态
    result = graph.get_state(config).values
    all_sections = result.get("all_sections", []) + result.get("sections", [])
    print(f"\n{'='*60}")
    print(f"📋 Supervisor 迭代总结")
    print(f"{'='*60}")
    print(f"   Supervisor 迭代次数: {result.get('supervisor_iterations', 0)}")
    print(f"   总小节数: {len(all_sections)}")
    print(f"   最终报告: {len(result.get('final_report', ''))} 字符")
    print(f"\n{'='*60}")
    print("📋 最终报告")
    print(f"{'='*60}")
    print(result.get("final_report", ""))
    print(f"{'='*60}")

    print(f"\n✅ Supervisor 迭代委派测试完成！")
