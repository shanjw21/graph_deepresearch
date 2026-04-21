from typing import TypedDict,List,Annotated
from pydantic import BaseModel,Field
from langgraph.graph import MessagesState
import operator



class Analyst(BaseModel):
    name: str = Field(description="分析师姓名")
    role: str = Field(description="角色定位")
    affiliation: str = Field(description="隶属机构")
    description: str = Field(description="分析师的关注焦点、关切点和动机的详细描述")

    @property
    def persona(self):
        return f"Name : {self.name}\n, Role: {self.role} \n, Affiliation : {self.affiliation}\n , Description: {self.description}\n"

# 生成分析师状态模型,每次都重新生成，是新的分析师
class GenerateAnalystsState(TypedDict):
    topic: str  # 研究主题
    max_analysts: int # 最大分析师上限
    human_analyst_feedback:str # 人类反馈
    analysts:List[Analyst] # 生成的分析师列表


# 自带 messages: Annotated[list, operator.add]，对话历史，每轮对话的消息自动追加
# context: 搜索结果：每轮搜索的文档自动追加
class InterviewState(MessagesState):
    max_num_turns:int # 最大访谈次数
    context:Annotated[list,operator.add] # 每轮搜索文档追加
    analyst: Analyst # 当前访谈师
    interview: str # 完整访谈记录文本
    sections: list # 报告小结


class ResearchGraphState(TypedDict):
    topic: str # 研究主题
    max_analysts: int # 最大分析师数量
    human_analyst_feedback: str  # 人类反馈
    analysts: List[Analyst] # 分析师列表
    sections: Annotated[list,operator.add] # 报告小结,累加所有访谈报告
    introduction:str # 报告引言
    content : str # 报告主体
    conclusion: str # 报告结论
    final_report: str # 最终报告


def simulate_state_update(result: dict, output:dict) -> dict:
    new_result = dict(result)
    for key, values in output.items():
        if key in ("context", "sections"):
            # values是一个list,应该使用[]处理
            new_result[key] = new_result.get(key,[]) + values
        else:
            new_result[key] = values
    return new_result

if __name__=="__main__":
    state = {"topic": "AI医疗", "sections": []}
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