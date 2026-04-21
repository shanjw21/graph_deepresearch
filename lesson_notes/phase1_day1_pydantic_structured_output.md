# 阶段一 · Day 1：Pydantic 数据模型 + LLM 结构化输出

## 今日目标

1. 理解 Pydantic 模型在 LLM 应用中的作用——为什么不让 LLM 直接返回 JSON？
2. 掌握 `with_structured_output()` 的原理和用法
3. 亲手重建项目中的 3 个 Pydantic 模型（Analyst / Perspectives / SearchQuery）
4. 跑通最小结构化输出 Demo，验证"给一个主题，返回结构化的分析师列表"

---

## 知识点一：为什么需要结构化输出？

### 问题场景

当你问 LLM："为'AI在医疗中的应用'生成 3 个分析师"，LLM 可能返回：

```
好的，以下是三个分析师：

1. **张伟博士** - 来自协和医院，角色是医疗AI专家，关注AI辅助诊断...
2. **Sarah Chen** - 来自MIT，角色是AI伦理顾问，关注...
3. ...
```

问题在于：这是**自由文本**。你无法用代码直接访问 `result[0].name`。

### 解决方案：Pydantic + with_structured_output

```python
from pydantic import BaseModel, Field
from typing import List

class Analyst(BaseModel):
    name: str = Field(description="分析师姓名")
    role: str = Field(description="角色定位")

class Perspectives(BaseModel):
    analysts: List[Analyst]

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(Perspectives)

result = structured_llm.invoke("为'AI在医疗中的应用'生成3个分析师")
print(result.analysts[0].name)   # 直接用属性访问 ✅
print(result.analysts[0].role)   # 类型安全，IDE 有提示 ✅
```

### 底层原理

```
你的代码                  OpenAI API                  LLM
   │                         │                         │
   │  Pydantic schema        │                         │
   │ ───────────────────→  function calling           │
   │                       的 tools 参数               │
   │                         │ ─────────────────────→  │
   │                         │                         │
   │                         │    LLM 生成符合 schema   │
   │                         │    的 JSON 输出          │
   │                         │ ←─────────────────────  │
   │                         │                         │
   │  Pydantic 自动解析      │                         │
   │  并校验类型             │                         │
   │ ←───────────────────   │                         │
```

**关键点：** `with_structured_output` 并不是让 LLM "返回一个 Pydantic 对象"，而是把 Pydantic 模型转成 function calling 的 tool schema，LLM 返回 JSON，LangChain 自动用 Pydantic 解析。

---

## 知识点二：Field 的 description 是写给谁看的？

```python
class Analyst(BaseModel):
    name: str = Field(description="分析师姓名")
    role: str = Field(description="分析师在研究主题中的具体角色定位")
```

**`description` 不是给程序员看的注释，是给 LLM 看的提示词。**

当 `with_structured_output` 把 Pydantic 模型转成 function calling schema 时，`description` 会变成每个字段的说明，指导 LLM 填充正确的内容。

**面试考点：**
- Q: "Field 的 description 有什么用？"
- A: "它是 LLM 的提示词，指导 LLM 理解每个字段应该填什么内容，提高生成质量"

---

## 知识点三：@property 的妙用

源码第 234-245 行：

```python
class Analyst(BaseModel):
    affiliation: str = Field(description="分析师的主要隶属机构或组织")
    name: str = Field(description="分析师姓名")
    role: str = Field(description="分析师在研究主题中的具体角色定位")
    description: str = Field(description="分析师的关注焦点、关切点和动机的详细描述")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"
```

**设计意图：** `persona` 不是一个独立字段（不需要 LLM 生成），而是把 4 个字段拼接成一段人设描述，直接嵌入到后续的提示词模板中。

```python
# 在 generate_question 中使用
system_message = question_instructions.format(goals=analyst.persona)
# persona 被插入到提示词中，告诉 LLM "你现在扮演这个人"
```

**面试考点：**
- Q: "为什么 persona 用 @property 而不是普通字段？"
- A: "persona 是其他 4 个字段的派生值，不需要 LLM 生成，用 @property 避免数据冗余，保证一致性"

---

## 知识点四：容器模式 — Perspectives

```python
class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="包含所有分析师角色和隶属机构的综合列表")
```

**为什么需要一个 Perspectives 包装类？**

因为 `with_structured_output` 要求返回**单个** Pydantic 对象。如果直接用 `List[Analyst]`，LLM 不知道要返回一个列表。用 Perspectives 包一层，LLM 就知道要返回一个包含 analysts 列表的对象。

```
❌ structured_llm = llm.with_structured_output(List[Analyst])   # 不确定的行为
✅ structured_llm = llm.with_structured_output(Perspectives)    # 明确的 schema
```

---

## 知识点五：SearchQuery — 最简单的结构化输出

```python
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="用于检索的搜索查询语句")
```

这个模型的用途：把一段对话历史转成一句搜索查询。

```python
# 源码第 681-682 行
structured_llm = llm.with_structured_output(SearchQuery)
search_query = structured_llm.invoke([search_instructions] + state['messages'])
# search_query.search_query 就是生成的搜索词
```

**面试考点：**
- Q: "为什么要把搜索查询也做成结构化输出？"
- A: "确保 LLM 只返回纯搜索词，不会带解释性文字，直接传给搜索引擎"

---

## 动手实验

### 实验 1：最小结构化输出（15 分钟）

```python
# 文件：experiment_1_structured_output.py

from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI

# Step 1: 定义模型
class Analyst(BaseModel):
    name: str = Field(description="分析师姓名")
    role: str = Field(description="角色定位")
    affiliation: str = Field(description="隶属机构")

class Perspectives(BaseModel):
    analysts: List[Analyst]

# Step 2: 创建 LLM + 结构化输出
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(Perspectives)

# Step 3: 调用
result = structured_llm.invoke("为'AI在医疗中的应用'生成3个分析师")

# Step 4: 检查结果
for a in result.analysts:
    print(f"姓名: {a.name}, 角色: {a.role}, 机构: {a.affiliation}")
```

**你的任务：** 运行以上代码，确认能正常输出。

---

### 实验 2：对比有无结构化输出（10 分钟）

```python
# 无结构化输出
normal_llm = ChatOpenAI(model="gpt-4o", temperature=0)
result1 = normal_llm.invoke("为'AI在医疗中的应用'生成3个分析师")
print("=== 无结构化输出 ===")
print(type(result1.content))  # str，纯文本
print(result1.content[:200])
print()

# 有结构化输出
structured_llm = llm.with_structured_output(Perspectives)
result2 = structured_llm.invoke("为'AI在医疗中的应用'生成3个分析师")
print("=== 有结构化输出 ===")
print(type(result2))           # Perspectives 对象
print(result2.analysts[0])     # Analyst 对象，可直接访问属性
```

**思考：** 两种方式返回的类型有什么不同？为什么结构化输出更适合生产环境？

---

### 实验 3：完整 Analyst 模型 + persona（20 分钟）

**你的任务：** 不看源码，根据下面的描述自己写代码。

描述：
1. 创建 `Analyst` 类，继承 `BaseModel`，包含 4 个字段：
   - `affiliation: str` — 隶属机构
   - `name: str` — 姓名
   - `role: str` — 角色定位
   - `description: str` — 关注焦点

2. 添加 `@property persona` 方法，返回格式化字符串：
   ```
   Name: {name}
   Role: {role}
   Affiliation: {affiliation}
   Description: {description}
   ```

3. 创建 `Perspectives` 容器类，包含 `analysts: List[Analyst]`

4. 调用 LLM，主题用 "大语言模型在教育领域的应用"

5. 遍历结果，打印每个分析师的 persona

**验证标准：**
- 能直接用 `result.analysts[0].name` 访问属性
- `persona` 输出格式正确
- 每个字段的 `description` 引导 LLM 生成了合理内容

---

### 实验 4：SearchQuery 实验（10 分钟）

```python
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="用于检索的搜索查询语句")

structured_llm = llm.with_structured_output(SearchQuery)

# 模拟：把一段对话转成搜索词
result = structured_llm.invoke(
    "以下是分析师和专家的对话："
    "分析师问：目前AI在医疗影像诊断方面的准确率如何？"
    "请生成一条Web搜索查询语句。"
)

print(result.search_query)
# 预期输出类似："AI medical imaging diagnosis accuracy 2024"
```

**思考：** 为什么输出是纯搜索词而不是带解释的段落？

---

## 今日面试题

### 基础题

**Q1: "Pydantic 在你的项目中起什么作用？"**
> 在 LLM 应用中，Pydantic 模型有双重角色：一是作为 function calling 的 schema 定义，指导 LLM 按照预定义的结构返回数据；二是作为数据校验层，确保 LLM 返回的数据类型正确。我项目中的 Analyst、Perspectives、SearchQuery 都是这样用的。

**Q2: "with_structured_output 的底层原理是什么？"**
> 它把 Pydantic 模型转换成 OpenAI function calling 的 tool schema，LLM 返回 JSON 格式的数据，LangChain 再用 Pydantic 自动解析和校验。本质上不是让 LLM "返回对象"，而是用 schema 约束 LLM 的输出格式。

**Q3: "Field 的 description 参数有什么用？"**
> 它是给 LLM 看的提示词。with_structured_output 会把 description 放进 tool schema 的字段描述中，指导 LLM 理解每个字段应该填什么内容。比如 `Field(description="分析师姓名")` 告诉 LLM 这个字段要填人名，不是职位。

### 进阶题

**Q4: "为什么 persona 用 @property 而不是做成 Pydantic 字段？"**
> persona 是其他 4 个字段（name/role/affiliation/description）的拼接结果，是派生值。如果做成独立字段，就需要 LLM 同时生成原始字段和 persona，造成数据冗余且可能不一致。用 @property 保证 persona 永远和其他字段同步。

**Q5: "为什么需要 Perspectives 包装类？"**
> with_structured_output 要求返回单个 Pydantic 对象。用 Perspectives 包装 List[Analyst]，LLM 能明确知道要返回一个包含 analysts 列表的结构，而不是其他格式。

---

## 今日总结

```
你今天学到的：

Pydantic 模型 ──── LLM 的"输出合同"
    │                用 Field(description=...) 告诉 LLM 每个字段填什么
    │                用 with_structured_output() 强制执行
    │
    ├── Analyst      4字段 + @property persona
    ├── Perspectives 容器类，包装 List[Analyst]
    └── SearchQuery  最简单的单字段模型，对话→搜索词

    底层原理：Pydantic schema → function calling tool → LLM 返回 JSON → Pydantic 解析

明天预告：
    - TypedDict 状态模型（ResearchGraphState / InterviewState）
    - operator.add 累加机制
    - MessagesState 继承
    - 这些状态模型如何驱动整个 LangGraph 工作流
```

---

## 参考源码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| Analyst 模型 | `research_assistant.py` | 215-246 |
| Perspectives 模型 | `research_assistant.py` | 248-255 |
| SearchQuery 模型 | `research_assistant.py` | 271-278 |
| with_structured_output 用法 | `research_assistant.py` | 582-598 |
| search_instructions 提示词 | `research_assistant.py` | 370-378 |
