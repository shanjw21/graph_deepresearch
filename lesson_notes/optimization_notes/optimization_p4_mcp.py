"""
P4 优化骨架：MCP 文件搜索集成

在访谈子图中加入 MCP 本地文件搜索节点，与 Tavily + Wikipedia 并行。
关键改动：
1. 新增 MCP 客户端配置（懒加载）
2. 新增 search_local_files 异步节点
3. llm_call / tool_node 改为 async（MCP 要求）
4. 三路并行搜索：web + baike + local_files

基于完整优化版本（P0-P3 + Supervisor）的叠加。
"""

import os
import re
import operator
from typing import List, Annotated
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, filter_messages
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


class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    all_sections: Annotated[list, operator.add]
    supervisor_iterations: int
    max_supervisor_iterations: int
    coverage_gaps: str
    introduction: str
    content: str
    conclusion: str
    final_report: str


# ============================================
# 提示词模板
# ============================================

language_instruction = """
**重要：请检测用户输入的语言，并使用相同的语言输出报告。**
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

你的目标是基于这段对话，为Web搜索和本地文件搜索生成一条结构良好的查询语句。

特别关注分析师最后提出的问题。

将这个最终问题转化为结构良好的搜索查询。""")

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

你的任务：将备忘录整合为简洁的总体总结，使用 Markdown 格式。

{language_instruction}

以下是分析师提供的备忘录：

{context}"""

intro_conclusion_instructions = """你是一名技术写作者，正在完成主题为 {topic} 的报告。

你将获得报告的全部小节。你的任务是撰写简洁而有说服力的引言或结论。

目标约 100 字，使用 Markdown 格式。

{language_instruction}

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

如果需要补充研究，请以「需要补充：[具体角度]」格式列出缺失角度。
如果覆盖已充分，请回复「覆盖已充分，可以进入报告撰写阶段」。"""

delegate_more_prompt = """你是一名研究主管。现有研究对主题「{topic}」的分析还缺少以下角度：

{coverage_gaps}

请生成 1-2 位新的分析师来补充这些缺失的角度。"""


# ===== P4 改动 1：MCP 研究 Agent 提示词 =====
research_agent_prompt_with_mcp = """你是一名研究助理，可以通过本地文件系统和本地搜索来收集信息。

<Tasks>
1. 先用 MCP 文件工具查看本地有哪些文档
2. 搜索相关文件内容
3. 如果本地资料不足，再用外部搜索补充
</Tasks>

<Available Tools>
- **MCP 文件系统工具**：搜索和读取本地研究文档
  - list_allowed_directories: 查看可访问的目录
  - list_directory: 列出目录中的文件
  - search_files: 搜索包含特定内容的文件
  - read_file: 读取文件内容
  - read_multiple_files: 同时读取多个文件
- **think_tool**: 反思和规划
</Available Tools>

<Instructions>
像一名有文档库访问权限的研究员那样思考：

1. 先了解可用文件 — 用 list_allowed_directories 和 list_directory
2. 识别相关文件 — 用 search_files 查找匹配主题的文件
3. 有策略地阅读 — 从最相关的文件开始
4. 每次阅读后暂停评估 — 信息够了吗？还缺什么？
5. 有信心回答时停止 — 不要为完美继续阅读
</Instructions>

<Hard Limits>
- 简单查询：最多 3-4 次文件操作
- 复杂查询：最多 6 次文件操作
- 连续 2 次文件读取返回相似信息时立即停止
</Hard Limits>

<Show Your Thinking>
每次读取文件后用 think_tool 分析：
- 我找到了什么关键信息？
- 还缺什么？
- 应该继续读取还是给出答案？
- 始终引用你使用的文件名
</Show Your Thinking>"""


# ============================================
# LLM 和搜索工具
# ============================================

llm = ChatOpenAI(model=model, temperature=0, base_url=base_url, api_key=api_key)

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
tavily_search = TavilySearchResults(max_results=3)


# ===== P4 改动 2：MCP 客户端配置（懒加载） =====
from langchain_mcp_adapters.client import MultiServerMCPClient

# 指定要暴露给 Agent 的本地文档目录
# 这里用项目下的 lesson_notes 目录作为示例
current_dir = os.path.dirname(os.path.abspath(__file__))
research_docs_path = current_dir  # 可以改成你存放研究文档的目录

mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",  # 自动安装（如果需要）
            "@modelcontextprotocol/server-filesystem",
            research_docs_path  # 可访问的目录路径
        ],
        "transport": "stdio"  # 通过 stdin/stdout 通信
    }
}

# 懒加载 MCP 客户端 — 避免在导入时就启动子进程
_mcp_client = None

def get_mcp_client():
    """获取或初始化 MCP 客户端。"""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MultiServerMCPClient(mcp_config)
    return _mcp_client


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
# 访谈子图节点（含 P4 MCP 集成）
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


# ===== P4 改动 3：新增 MCP 本地文件搜索节点（异步） =====
async def search_local_files(state: InterviewState):
    """
    使用 MCP 文件系统搜索本地研究文档。

    流程：
    1. 获取 MCP 工具
    2. 让 LLM 决定搜索什么（用 search_query）
    3. 用 MCP 工具搜索和读取文件
    4. 格式化结果为 context
    """
    seen_urls = extract_urls_from_context(state.get("context", []))

    # 获取 MCP 工具
    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    tools_by_name = {tool.name: tool for tool in mcp_tools}

    # 决定搜索什么
    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    print(f"  [MCP] 搜索本地文件: {search_query.search_query}")

    try:
        # 第一步：查看有哪些目录和文件
        list_dir_tool = tools_by_name.get("list_allowed_directories")
        if list_dir_tool:
            dir_listing = await list_dir_tool.ainvoke({})
        else:
            dir_listing = "No directory listing available"

        # 第二步：搜索相关文件
        search_tool = tools_by_name.get("search_files")
        if search_tool:
            search_result = await search_tool.ainvoke({
                "path": research_docs_path,
                "pattern": search_query.search_query[:50]  # 限制长度
            })
        else:
            search_result = "Search not available"

        # 第三步：读取相关文件
        read_tool = tools_by_name.get("read_file")
        file_contents = []
        file_names = []

        if search_result and isinstance(search_result, str):
            # 从搜索结果中提取文件名
            # 简单策略：尝试读取看起来像文件的行
            for line in search_result.split("\n")[:10]:  # 最多读 10 个
                if ".md" in line or ".txt" in line or ".pdf" in line:
                    # 提取文件路径
                    import re as _re
                    matches = _re.findall(r'[\w/.\-]+\.\w+', line)
                    for match in matches:
                        if match not in file_names and len(file_names) < 3:
                            file_names.append(match)
                            if read_tool:
                                try:
                                    content = await read_tool.ainvoke({"path": match})
                                    if len(content) > 500:
                                        content = summarize_content(content, max_length=300)
                                    file_contents.append(content)
                                except Exception:
                                    pass  # 跳过读取失败的文件

        # 格式化结果
        if file_contents:
            formatted_docs = "\n\n---\n\n".join([
                f'<Document source="local_file:{name}" >\n{content}\n</Document>'
                for name, content in zip(file_names, file_contents)
            ])
        else:
            # 即使没找到文件，也把目录结构给 LLM 参考
            formatted_docs = (
                f"<Document source=\"local_dir_listing\">\n"
                f"本地目录结构：\n{dir_listing}\n\n"
                f"搜索查询: {search_query.search_query}\n"
                f"搜索结果: 未找到完全匹配的文件\n"
                f"</Document>"
            )

    except Exception as e:
        print(f"  [MCP 降级] {e}")
        formatted_docs = f"<Document />本地文件搜索暂不可用，查询: {search_query.search_query}\n</Document>"

    # 本地文件的"URL"用 file:// 前缀标记
    local_urls = [f"file://{name}" for name in file_names]
    return {"context": [formatted_docs], "seen_urls": local_urls}


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


# ===== P4 改动 4：MCP 专用的研究子图（替代原有的 Tavily/Wikipedia 子图） =====
#
# 这是一个独立的 MCP 研究子图，可以单独使用，用于仅基于本地文件的研究。
# 它的结构与 research_agent_mcp.py 相同：llm_call → tool_node → compress_research

async def mcp_llm_call(state: InterviewState):
    """MCP 研究子图的 LLM 调用节点。"""
    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    tools = mcp_tools + []  # 只使用 MCP 工具，不额外加 think_tool

    model_with_tools = llm.bind_tools(tools)

    system_prompt = research_agent_prompt_with_mcp.format(date="2026年5月28日")

    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=system_prompt)] + state["messages"]
            )
        ]
    }


async def mcp_tool_node(state: InterviewState):
    """MCP 研究子图的工具节点，异步执行 MCP 工具。"""
    tool_calls = state["messages"][-1].tool_calls

    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    tools = mcp_tools
    tools_by_name = {tool.name: tool for tool in tools}

    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name.get(tool_call["name"])
        if tool:
            observation = await tool.ainvoke(tool_call["args"])
        else:
            observation = f"Tool {tool_call['name']} not found"
        observations.append(
            ToolMessage(
                content=str(observation),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": observations}


def mcp_should_continue(state: InterviewState):
    """决定 MCP 子图是否继续搜索。"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "mcp_tool_node"
    return "mcp_compress"


def mcp_compress(state: InterviewState):
    """压缩 MCP 研究结果为简洁摘要。"""
    messages = state["messages"]
    tool_messages = filter_messages(messages, include_types="tool")

    # 拼接所有工具输出
    combined = "\n\n---\n\n".join([
        f"<Document source=\"mcp:{m.name}\">\n{m.content}\n</Document>"
        for m in tool_messages
    ])

    if not combined:
        combined = "<Document />MCP 研究未找到相关信息</Document>"

    if len(combined) > 1000:
        combined = summarize_content(combined, max_length=500)

    return {"context": [combined]}


# 编译 MCP 研究子图
mcp_researcher_builder = StateGraph(InterviewState)
mcp_researcher_builder.add_node("mcp_llm_call", mcp_llm_call)
mcp_researcher_builder.add_node("mcp_tool_node", mcp_tool_node)
mcp_researcher_builder.add_node("mcp_compress", mcp_compress)

mcp_researcher_builder.add_edge(START, "mcp_llm_call")
mcp_researcher_builder.add_conditional_edges(
    "mcp_llm_call",
    mcp_should_continue,
    {
        "mcp_tool_node": "mcp_tool_node",
        "mcp_compress": "mcp_compress"
    }
)
mcp_researcher_builder.add_edge("mcp_tool_node", "mcp_llm_call")
mcp_researcher_builder.add_edge("mcp_compress", END)

mcp_research_graph = mcp_researcher_builder.compile()
print("✅ MCP 研究子图编译完成")


# ============================================
# 编译访谈子图
# ============================================

interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_baike", search_baike)
interview_builder.add_node("search_local_files", search_local_files)  # P4 新增
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("reflect", reflect)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

interview_builder.add_edge(START, "ask_question")

# 三路并行搜索
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_baike")
interview_builder.add_edge("ask_question", "search_local_files")

# 所有搜索完成后进入回答
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_baike", "answer_question")
interview_builder.add_edge("search_local_files", "answer_question")

interview_builder.add_edge("answer_question", "reflect")
interview_builder.add_conditional_edges(
    "reflect",
    route_after_reflect,
    ["ask_question", "save_interview"]
)
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

interview_graph = interview_builder.compile()
print("✅ 访谈子图编译完成（含 P4 MCP 本地文件搜索）")
print("   搜索架构: ask_question → search_web (并行)")
print("                        → search_baike (并行)")
print("                        → search_local_files (MCP, 并行)")
print("                        → answer_question (综合三路结果)")


# ============================================
# 主图节点（Supervisor 迭代架构）
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


def evaluate_coverage(state: ResearchGraphState):
    """Supervisor 评估已有 sections 是否覆盖了主题的关键方面。"""
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
    """基于评估发现的缺失角度，生成补充分析师。"""
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

    return {"analysts": result.analysts}


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


# Reduce 节点
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
# 构建主图（含 Supervisor 循环 + MCP）
# ============================================

builder = StateGraph(ResearchGraphState)

builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_graph)
builder.add_node("evaluate_coverage", evaluate_coverage)
builder.add_node("delegate_more_analysts", delegate_more_analysts)
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
builder.add_edge("conduct_interview", "evaluate_coverage")
builder.add_conditional_edges(
    "evaluate_coverage",
    route_after_evaluate,
    {
        "needs_more": "delegate_more_analysts",
        "enough_coverage": "write_report"
    }
)
builder.add_edge("delegate_more_analysts", "conduct_interview")
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
print("✅ 主图编译完成（含 P0-P4 全部优化 + MCP + Supervisor 循环）")
print("   完整架构:")
print("   Scope: create_analysts → human_feedback")
print("   Map:   Send(分析师) → conduct_interview")
print("          ↳ search_web + search_baike + search_local_files(MCP) [三路并行]")
print("   Eval:  conduct_interview → evaluate_coverage")
print("   Loop:  evaluate_coverage → delegate_more → conduct_interview (循环)")
print("   Reduce: write_report + intro + conclusion → finalize_report → END")


# ============================================
# 测试
# ============================================

if __name__ == "__main__":
    topic = "AI在医疗领域的应用与挑战"
    max_analysts = 2
    max_supervisor_iterations = 1

    print(f"=== P4 MCP 集成测试：{topic} ===\n")
    print(f"   初始分析师: {max_analysts}")
    print(f"   最大迭代次数: {max_supervisor_iterations}")
    print(f"   本地文档目录: {research_docs_path}\n")

    config = {"configurable": {"thread_id": "test_p4_mcp"}}

    print("--- 第1步：生成分析师 ---")
    result = graph.invoke({
        "topic": topic,
        "max_analysts": max_analysts,
        "max_supervisor_iterations": max_supervisor_iterations
    }, config)

    print(f"\n生成了 {len(result['analysts'])} 个分析师")

    print("\n--- 第2步：批准分析师 ---")
    graph.update_state(config, {"human_analyst_feedback": ""}, as_node="human_feedback")

    print("\n--- 第3步：运行（含 MCP 本地搜索）---")
    for event in graph.stream(None, config, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n📍 节点: {node_name}")
            if node_name == "finalize_report":
                report = node_output.get("final_report", "")
                print(f"  完整报告 ({len(report)} 字符)")

    result = graph.get_state(config).values
    all_sections = result.get("all_sections", []) + result.get("sections", [])

    print(f"\n{'='*60}")
    print(f"📋 测试总结")
    print(f"{'='*60}")
    print(f"   Supervisor 迭代次数: {result.get('supervisor_iterations', 0)}")
    print(f"   总小节数: {len(all_sections)}")
    print(f"   最终报告: {len(result.get('final_report', ''))} 字符")

    print(f"\n✅ P4 MCP 集成测试完成！")
