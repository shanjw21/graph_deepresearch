# P3 优化：最终报告语言自适应

## 背景知识

### 为什么需要语言自适应？

官方教程的 `final_report_generation_prompt` 中有一段关键指令：

> CRITICAL: Make sure the answer is written in the same language as the human messages!
> For example, if the user's messages are in English, then MAKE SURE you write your response in English. If the user's messages are in Chinese, then MAKE SURE you write your entire response in Chinese.

这是一个**简单但重要**的改进。当前 `phase5_day9_skeleton.py` 中所有报告提示词都硬编码了"全部使用中文"，这意味着：
- 英文用户会收到中文报告
- 多语言团队无法使用
- 每次换语言都要改代码

### LLM 的语言检测能力

现代 LLM 天然具备语言检测能力——它们能在上下文中识别输入语言。但要让输出语言正确，需要在 prompt 中**显式指令**。

### 实现思路

最简单的方式：在每个报告相关 prompt 中替换硬编码语言要求为自适应指令。

---

## 实现步骤

### 步骤 1：定义语言检测提示词

```python
language_instruction = """
**重要：请检测用户输入的语言，并使用相同的语言输出报告。**
- 如果用户输入是中文，用中文写报告
- 如果用户输入是英文，用英文写报告
- 如果用户输入是其他语言，也用该语言写报告
"""
```

### 步骤 2：替换所有硬编码语言要求

将所有提示词中的：
```
**重要：全部使用中文**
```

替换为：
```python
{language_instruction}
```

### 步骤 3：在 prompt 格式化时传入

```python
report_writer_instructions = """...（报告指令）...

{language_instruction}

以下是分析师提供的备忘录：
{context}"""

system_messages = report_writer_instructions.format(
    topic=topic,
    context=context,
    language_instruction=language_instruction
)
```

### 影响的提示词

| 提示词 | 原硬编码 | 替换为 |
|--------|----------|--------|
| `section_writer_instructions` | 全部使用中文 | `{language_instruction}` |
| `report_writer_instructions` | 全部使用中文 | `{language_instruction}` |
| `intro_conclusion_instructions` | 全部使用中文 | `{language_instruction}` |

---

## 关键注意事项

1. **零成本**：这个改动不需要额外的 LLM 调用，只是 prompt 措辞的变化。

2. **可靠性**：LLM 对语言检测非常可靠。但可以在指令中加一个**示例**提高可靠性：
   ```
   Example: User says "What are the latest AI trends?" → Write in English.
   Example: User says "AI的最新趋势是什么？" → Write in Chinese.
   ```

3. **混合语言场景**：如果用户输入是混合语言（如中英文混用），LLM 通常会选择主要语言。如果需要在特定场景强制某种语言，可以加一个 `target_language` 参数。
