from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model = os.getenv("MODEL")



# Step 1: 定义模型
class Analyst(BaseModel):
    name: str = Field(description="分析师姓名")
    role: str = Field(description="角色定位")
    affiliation: str = Field(description="隶属机构")
    description: str = Field(description="分析师的关注焦点、关切点和动机的详细描述")

    @property
    def persona(self):
        return f"Name : {self.name}\n, Role: {self.role} \n, Affiliation : {self.affiliation}\n , Description: {self.description}\n"

# pydantic类型的包装类,让LLM返回一个Analyst类型的列表
class Perspectives(BaseModel):
    analysts: List[Analyst]

# Step 2: 创建 LLM + 结构化输出
llm = ChatOpenAI(model=model, api_key=api_key,base_url=base_url,temperature=0)
structured_llm = llm.with_structured_output(Perspectives,method="function_calling")

result = structured_llm.invoke("为'大语言模型在教育领域的应用'生成3个分析师")

for res in result.analysts:
    print(f"LLM's Output is {res}")

class SearchQuery(BaseModel):
    search_query:str = Field(description="根据对话历史,提炼搜索关键词")

structured_llm_search = llm.with_structured_output(SearchQuery,method="function_calling")
search_result = structured_llm_search.invoke( "以下是分析师和专家的对话：\n"
    "分析师问：目前AI在医疗影像诊断方面的准确率如何？有哪些最新的临床验证数据？\n"
    "请基于这段对话生成一条适合Web搜索的查询语句。")
print(f"search_result : {search_result}")