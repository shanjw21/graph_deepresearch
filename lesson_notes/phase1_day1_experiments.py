"""
阶段一 · Day 1 动手实验：Pydantic 数据模型 + LLM 结构化输出

使用方法：
1. 确保已安装依赖：pip install pydantic langchain-openai
2. 设置环境变量：export OPENAI_API_KEY=your_key
3. 运行：python phase1_day1_experiments.py
"""

import os
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI


# ============================================
# 实验 1：最小结构化输出
# ============================================
print("=" * 60)
print("实验 1：最小结构化输出")
print("=" * 60)


class Analyst(BaseModel):
    name: str = Field(description="分析师姓名")
    role: str = Field(description="角色定位")
    affiliation: str = Field(description="隶属机构")


class Perspectives(BaseModel):
    analysts: List[Analyst]


llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(Perspectives)

result = structured_llm.invoke("为'AI在医疗中的应用'生成3个分析师")

for a in result.analysts:
    print(f"  姓名: {a.name}")
    print(f"  角色: {a.role}")
    print(f"  机构: {a.affiliation}")
    print("  ---")


# ============================================
# 实验 2：对比有无结构化输出
# ============================================
print("\n" + "=" * 60)
print("实验 2：对比有无结构化输出")
print("=" * 60)

# 无结构化输出
result1 = llm.invoke("为'AI在医疗中的应用'生成3个分析师")
print(f"  无结构化 → type: {type(result1.content)}")
print(f"  内容前200字: {result1.content[:200]}")
print()

# 有结构化输出
result2 = structured_llm.invoke("为'AI在医疗中的应用'生成3个分析师")
print(f"  有结构化 → type: {type(result2)}")
print(f"  第一个分析师: {result2.analysts[0]}")


# ============================================
# 实验 3：完整 Analyst 模型 + persona
# ============================================
print("\n" + "=" * 60)
print("实验 3：完整 Analyst 模型 + persona")
print("=" * 60)


class FullAnalyst(BaseModel):
    """尝试自己写这个类，然后和源码对比"""
    affiliation: str = Field(description="分析师的主要隶属机构或组织")
    name: str = Field(description="分析师姓名")
    role: str = Field(description="分析师在研究主题中的具体角色定位")
    description: str = Field(description="分析师的关注焦点、关切点和动机的详细描述")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class FullPerspectives(BaseModel):
    analysts: List[FullAnalyst]


structured_llm_full = llm.with_structured_output(FullPerspectives)
result3 = structured_llm_full.invoke("为'大语言模型在教育领域的应用'生成3个分析师")

for a in result3.analysts:
    print(f"  --- 分析师 ---")
    print(a.persona)


# ============================================
# 实验 4：SearchQuery 实验
# ============================================
print("\n" + "=" * 60)
print("实验 4：SearchQuery — 对话转搜索词")
print("=" * 60)


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="用于检索的搜索查询语句")


structured_llm_search = llm.with_structured_output(SearchQuery)

result4 = structured_llm_search.invoke(
    "以下是分析师和专家的对话：\n"
    "分析师问：目前AI在医疗影像诊断方面的准确率如何？有哪些最新的临床验证数据？\n"
    "请基于这段对话生成一条适合Web搜索的查询语句。"
)

print(f"  生成的搜索词: {result4.search_query}")


print("\n" + "=" * 60)
print("全部实验完成！")
print("=" * 60)
