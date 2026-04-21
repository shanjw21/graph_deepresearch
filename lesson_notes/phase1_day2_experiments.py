"""
阶段一 · Day 2 动手实验：TypedDict 状态模型 + operator.add 累加机制

使用方法：
1. 确保已安装依赖：pip install langgraph langchain-openai
2. 设置环境变量：export OPENAI_API_KEY=your_key
3. 运行：python phase1_day2_experiments.py
"""

import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState


# ============================================
# 先准备 Analyst 模型（昨天学的）
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


# ============================================
# 实验 1：验证 operator.add 的行为
# ============================================
print("=" * 60)
print("实验 1：operator.add 行为模拟")
print("=" * 60)


def simulate_state_update(current_state: dict, node_output: dict):
    """模拟 LangGraph 的状态更新逻辑"""
    result = dict(current_state)
    for key, new_value in node_output.items():
        if key in ("sections", "context"):
            result[key] = result.get(key, []) + new_value
        else:
            result[key] = new_value
    return result


# 模拟 3 个并行访谈节点返回 sections
state = {"topic": "AI医疗", "sections": []}

state = simulate_state_update(state, {"sections": ["## AI诊断进展\n根据研究..."]})
print(f"  分析师1完成后: sections 有 {len(state['sections'])} 个小节")

state = simulate_state_update(state, {"sections": ["## AI伦理挑战\n伦理问题..."]})
print(f"  分析师2完成后: sections 有 {len(state['sections'])} 个小节")

state = simulate_state_update(state, {"sections": ["## AI成本分析\n成本效益..."]})
print(f"  分析师3完成后: sections 有 {len(state['sections'])} 个小节")

print(f"\n  最终 sections 内容:")
for i, s in enumerate(state["sections"]):
    print(f"    [{i+1}] {s[:30]}...")

# 对比：普通字段的行为
state = simulate_state_update(state, {"topic": "新主题"})
print(f"\n  普通字段 topic 被覆盖为: {state['topic']}")


# ============================================
# 实验 2：验证真实 operator.add 的合并行为
# ============================================
print("\n" + "=" * 60)
print("实验 2：真实 operator.add 合并")
print("=" * 60)

# operator.add 对 list 的作用就是 +
list_a = ["文档A"]
list_b = ["文档B", "文档C"]
merged = operator.add(list_a, list_b)
print(f"  operator.add({list_a}, {list_b}) = {merged}")

# 模拟 context 累加：每轮搜索追加新文档
context = []
round1_web = ["<Document>搜索结果1</Document>"]
round1_baike = ["<Document>百科结果1</Document>"]
context = operator.add(context, round1_web)
context = operator.add(context, round1_baike)
print(f"  第1轮搜索后 context 有 {len(context)} 个文档")

round2_web = ["<Document>搜索结果2</Document>"]
round2_baike = ["<Document>百科结果2</Document>"]
context = operator.add(context, round2_web)
context = operator.add(context, round2_baike)
print(f"  第2轮搜索后 context 有 {len(context)} 个文档")
print(f"  这就是 write_section 时能引用的全部素材")


# ============================================
# 实验 3：完整状态模型验证
# ============================================
print("\n" + "=" * 60)
print("实验 3：状态模型定义验证")
print("=" * 60)


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


# 验证 InterviewState 继承了 MessagesState
print(f"  InterviewState 的 __annotations__: {InterviewState.__annotations__.keys()}")
print(f"  包含 'messages' 字段（继承自 MessagesState）: {'messages' in InterviewState.__annotations__}")

# 验证可以正常创建实例
sample_analyst = Analyst(
    affiliation="协和医院",
    name="张伟博士",
    role="医疗AI专家",
    description="关注AI在影像诊断中的应用"
)

interview_state: InterviewState = {
    "messages": [HumanMessage(content="你好")],
    "max_num_turns": 2,
    "context": [],
    "analyst": sample_analyst,
    "interview": "",
    "sections": []
}

print(f"\n  InterviewState 实例创建成功:")
print(f"    analyst.name: {interview_state['analyst'].name}")
print(f"    max_num_turns: {interview_state['max_num_turns']}")
print(f"    messages 数量: {len(interview_state['messages'])}")

research_state: ResearchGraphState = {
    "topic": "AI在医疗领域的应用",
    "max_analysts": 3,
    "human_analyst_feedback": "",
    "analysts": [sample_analyst],
    "sections": [],
    "introduction": "",
    "content": "",
    "conclusion": "",
    "final_report": ""
}

print(f"\n  ResearchGraphState 实例创建成功:")
print(f"    topic: {research_state['topic']}")
print(f"    analysts 数量: {len(research_state['analysts'])}")


# ============================================
# 实验 4：messages 累加行为模拟
# ============================================
print("\n" + "=" * 60)
print("实验 4：MessagesState 的 messages 累加")
print("=" * 60)

messages = []

# 模拟访谈循环中的消息追加
# 节点 generate_question 返回
messages = messages + [AIMessage(content="AI在医疗影像方面有什么进展？", name="analyst")]
print(f"  分析师提问后: {len(messages)} 条消息")

# 节点 generate_answer 返回
messages = messages + [AIMessage(content="根据文献，CT影像的AI辅助诊断准确率已达到95%", name="expert")]
print(f"  专家回答后: {len(messages)} 条消息")

# 第二轮
messages = messages + [AIMessage(content="那这些研究中有哪些是在中国做的？", name="analyst")]
messages = messages + [AIMessage(content="北京协和医院在2023年发表了相关研究...", name="expert")]
print(f"  第二轮完成后: {len(messages)} 条消息")

print(f"\n  完整对话历史:")
for i, msg in enumerate(messages):
    role = "分析师" if msg.name == "analyst" else "专家"
    print(f"    [{i+1}] {role}: {msg.content[:40]}...")


print("\n" + "=" * 60)
print("全部实验完成！阶段一结束。")
print("=" * 60)
