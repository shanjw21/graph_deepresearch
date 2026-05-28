"""
P3 优化：语言自适应 - 纯 prompt diff

这个文件只展示需要改动的提示词部分，不包含可运行的图代码。
要运行，请使用 optimization_p3_language.py。
"""

# ===== 原始硬编码（需要删除） =====
# section_writer_instructions 中的：
# - **重要：全部使用中文**
#
# report_writer_instructions 中的：
# - **重要要求：生成的报告必须全部使用中文。**
#
# intro_conclusion_instructions 中的：
# - **重要要求：全部使用中文。**

# ===== 替换为 =====
language_instruction = """
**重要：请检测用户输入的语言，并使用相同的语言输出报告。**
- 如果用户输入是中文，用中文写报告
- 如果用户输入是英文，用英文写报告
- 如果用户输入是其他语言，也用该语言写报告

示例：
- User says "What are the latest AI trends?" → Write in English.
- User says "AI的最新趋势是什么？" → Write in Chinese.
"""

# ===== 在所有提示词中，将硬编码语言要求替换为 =====
# {language_instruction}

# ===== 在节点中传入 =====
# section_writer_instructions.format(focus=..., language_instruction=language_instruction)
# report_writer_instructions.format(topic=..., context=..., language_instruction=language_instruction)
# intro_conclusion_instructions.format(topic=..., formatted_str_sections=..., language_instruction=language_instruction)
