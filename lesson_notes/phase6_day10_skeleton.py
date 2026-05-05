"""
阶段六 · Day 10：智能体在线评估 — Langfuse 追踪与评分

任务：
1. 构建邮件处理 Agent（LangGraph）
2. 接入 Langfuse 追踪
3. 用三种方案对执行结果评分
4. 去 Langfuse 控制台查看追踪和评分

参考源码：03_trace_and_evaluation_langgraph_agents.ipynb Cell 13-28
"""

import os
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL")

# Langfuse 配置（确保 .env 中有这些变量）
LANGFUSE_PUBLIC_KEY=os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY=os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST="https://cloud.langfuse.com"


# ============================================
# 一、数据模型
# ============================================

class EmailState(TypedDict):
    """邮件处理 Agent 的状态结构"""
    email: Dict[str, Any]
    is_spam: Optional[bool]
    draft_response: Optional[str]
    messages: List[str]


# ============================================
# 二、LLM 初始化
# ============================================

llm = ChatOpenAI(model=model_name, temperature=0, base_url=base_url, api_key=api_key)


# ============================================
# 三、节点函数（TODO：实现 5 个节点）
# ============================================

def read_email(state: EmailState):
    """
    TODO: 入口节点 — 打印邮件基础信息

    提示：
    1. 从 state["email"] 获取邮件信息
    2. 打印发件人和主题
    3. 返回 {}（只展示，不修改状态）

    参考源码：Cell 15 中的 read_email 函数
    """
    email = state["email"]
    print(f"sender is {email.get("sender")}")
    print(f"subject is {email.get("subject")}")
    return {}

check_spam_instructions = """
你是一名资深邮件处理专家，擅长对邮件内容做判断，判断邮件是"SPAM"还是"HAM".
读取邮件:{email},
根据邮件发件人,主题，和内容，对邮件做出判断。
输出: "SPAM" or "HAM"
根据邮件判断直接输出"SPAM" 或 "HAM".
"""

def classify_email(state: EmailState):
    """
    TODO: 垃圾邮件分类 — 用 LLM 判断 SPAM/HAM

    提示：
    1. 从 state 获取邮件信息
    2. 构造提示词，让 LLM 返回 "SPAM" 或 "HAM"
    3. 调用 llm.invoke([HumanMessage(content=prompt)])
    4. 解析响应：is_spam = "spam" in response.content.lower()
    5. 如果不是垃圾邮件，把问答追加到 messages
    6. 返回 {"is_spam": is_spam, "messages": new_messages}

    参考源码：Cell 15 中的 classify_email 函数
    """
    email = state["email"]
    messages = state["messages"]
    classify_prompt = check_spam_instructions.format(email=email)
    response = llm.invoke([HumanMessage(content=classify_prompt)])
    is_spam = "spam" in response.content.lower()
    new_messages = messages
    if not is_spam:
        new_messages = new_messages + [response.content]
    return {
        "is_spam":is_spam,
        "messages": new_messages
    }


def handle_spam(state: EmailState):
    """
    TODO: 垃圾邮件处理 — 标记并打印

    提示：
    1. 打印 "已标记为垃圾邮件"
    2. 返回 {}

    参考源码：Cell 15 中的 handle_spam 函数
    """
    print(f"已标记为垃圾邮件")
    return {}

drafting_response_instructions = """
你是一名邮件助手管家，擅长以管家的口吻写邮件回复内容。
邮件:{email}
请根据以上邮件内容，编写回复邮件。
"""

def drafting_response(state: EmailState):
    """
    TODO: 合法邮件回复 — 用 LLM 起草回复

    提示：
    1. 从 state 获取邮件信息和 messages
    2. 构造提示词，让 LLM 以管家的口吻写回复
    3. 调用 llm.invoke([HumanMessage(content=prompt)])
    4. 把问答追加到 messages
    5. 返回 {"draft_response": response.content, "messages": new_messages}

    参考源码：Cell 15 中的 drafting_response 函数
    """
    email = state["email"]
    messages = state["messages"]
    response_prompt = drafting_response_instructions.format(email=email)
    response = llm.invoke([HumanMessage(content=response_prompt)])
    new_messages = messages + [response.content]
    return {
        "draft_response": response.content,
        "messages": new_messages
    }



def notify_mr_wayne(state: EmailState):
    """
    TODO: 通知韦恩先生 — 格式化通知消息

    提示：
    1. 从 state 获取邮件和回复草稿
    2. 构造通知消息
    3. 打印通知
    4. 返回 {}

    参考源码：Cell 15 中的 notify_mr_wayne 函数
    """
    email = state["email"]
    drafting_response = state["draft_response"]
    notify_message = "\n\n" + "韦恩先生，您收到一封邮件: " + "\n\n" \
    + f"邮件的发件人是 :{email.get("sender")} \n\n" + f"邮件的主题是: {email.get("subject")} \n\n" \
    +  f"建议回复的内容是: {drafting_response} \n\n"
    print(notify_message)
    return {}


# ============================================
# 四、条件路由
# ============================================

def route_email(state: EmailState):
    """根据 is_spam 决定路由"""
    if state["is_spam"]:
        return "spam"
    return "legitimate"


# ============================================
# 五、构建 StateGraph
# ============================================

# TODO: 构建邮件处理图
#
# 提示：
# 1. 创建 StateGraph(EmailState)
# 2. 添加 5 个节点
# 3. 添加边：
#    START → read_email → classify_email
#    classify_email → 条件路由 → {spam: handle_spam, legitimate: drafting_response}
#    handle_spam → END
#    drafting_response → notify_mr_wayne → END
# 4. 编译图

email_graph = StateGraph(EmailState)

# 添加节点
email_graph.add_node("read_email",read_email)
email_graph.add_node("classify_email",classify_email)
email_graph.add_node("spam",handle_spam)
email_graph.add_node("legitimate",drafting_response)
email_graph.add_node("notify_mr_wayne",notify_mr_wayne)
# 添加边
email_graph.add_edge(START,"read_email")
email_graph.add_edge("read_email","classify_email")
email_graph.add_conditional_edges("classify_email",route_email,["spam","legitimate"])
email_graph.add_edge("spam",END)
email_graph.add_edge("legitimate","notify_mr_wayne")
email_graph.add_edge("notify_mr_wayne",END)


# 编译
compiled_graph = email_graph.compile()
print("✅ 邮件 Agent 编译完成")


# ============================================
# 六、Langfuse 追踪接入（v4 API）
# ============================================

from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler


@observe()
def process_email(email_data):
    """
    TODO: 用 Langfuse 追踪包裹 Agent 执行

    提示（v4 API）：
    1. 创建 CallbackHandler
    2. 调用 compiled_graph.invoke，传入 callbacks=[handler]
    3. 返回 result
    4. @observe() 装饰器会自动创建 Trace 并记录 input/output，无需手动设置

    参考源码：Cell 19
    """
    handler = CallbackHandler()
    result = compiled_graph.invoke(
        input={"email":email_data, "messages": []},
        config={"callbacks":[handler]}
    )
    return result


# ============================================
# 七、测试数据
# ============================================

legitimate_email = {
    "sender": "京东客服",
    "subject": "关于你近期订单的发票开具说明",
    "body": "尊敬的韦恩先生，你好！关于你在京东的近期订单，增值税电子普通发票已开具并推送至你的邮箱。如需纸质发票或抬头变更，请在7日内通过'我的订单-申请开票'发起，我们将尽快处理。给你带来不便，敬请谅解。"
}

spam_email = {
    "sender": "某数字货币项目方",
    "subject": "限时暴涨100倍，立即上车！",
    "body": "韦恩先生，我们新上线了一款数字货币，承诺稳稳赚、稳赚不赔！扫码加群，前100名赠送空投名额，错过今天再等一年！"
}


# ============================================
# 八、三种评分方案（v4 API）
# ============================================

def demo_scoring_methods():
    """
    TODO: 演示三种评分方案（v4 API）

    方案一：span.score_trace() — 即时评分
    提示：
    1. 用 langfuse.start_as_current_observation(as_type="span", name="demo-scoring-1") as span
    2. 在 with 块内调用 process_email(legitimate_email)
    3. 用 span.score_trace(name="quality", value=1, data_type="NUMERIC")

    方案二：score_current_trace() — 上下文评分
    提示：
    1. 用 langfuse.start_as_current_observation(as_type="span", name="demo-scoring-2")
    2. 在 with 块内调用 process_email(legitimate_email)
    3. 用 langfuse.score_current_trace(name="quality-score", value=0.9, data_type="NUMERIC")

    方案三：create_score(trace_id=...) — 异步评分
    提示：
    1. 用 langfuse.start_as_current_observation(as_type="span", name="demo-scoring-3") as obs
    2. 在 with 块内调用 process_email(spam_email)
    3. 保存 obs.trace_id 到变量
    4. 在 with 块外，用 langfuse.create_score(trace_id=saved_trace_id, name="user-feedback", value=0, data_type="NUMERIC")

    参考源码：Cell 26-28
    """
    langfuse = get_client()

    # print("\n=== 方案一：span.score_trace() ===")
    # TODO: 实现方案一
    # with langfuse.start_as_current_observation(as_type="span",name="demo-scoring-1") as span:
    #     result = process_email(legitimate_email)
    #     span.score_trace(name="quality-score", value=0.9, data_type="NUMERIC")

    # print("\n=== 方案二：score_current_trace() ===")
    # TODO: 实现方案二
    # with langfuse.start_as_current_observation(as_type="span",name="demo-scoring-2") as span:
    #     result = process_email(legitimate_email)
    #     langfuse.score_current_trace(name="quality-score",value=0.88,data_type="NUMERIC")
    print("\n=== 方案三：create_score(trace_id=...) ===")
    # TODO: 实现方案三
    with langfuse.start_as_current_observation(as_type="span",name="user_feed_back") as obs:
        result = process_email(spam_email)
        saved_trace_id = obs.trace_id
    langfuse.create_score(
        trace_id=saved_trace_id,
        name="user_feedback",
        value=1,
        data_type="NUMERIC",
        comment="用户点击了有帮助"
    )



    # 刷新缓冲区，确保数据发送到 Langfuse
    langfuse.flush()
    print("\n✅ 评分数据已发送到 Langfuse")


# ============================================
# 九、端到端测试
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 10：智能体在线评估 — Langfuse 追踪与评分")
    print("=" * 60)

    # 第1步：测试合法邮件
    # print("\n--- 测试合法邮件 ---")
    # result_legit = process_email(legitimate_email)
    # print(f"\n结果: is_spam={result_legit.get('is_spam')}")
    # if result_legit.get('draft_response'):
    #     print(f"回复草稿: {result_legit['draft_response'][:100]}...")

    # # 第2步：测试垃圾邮件
    # print("\n--- 测试垃圾邮件 ---")
    # result_spam = process_email(spam_email)
    # print(f"\n结果: is_spam={result_spam.get('is_spam')}")

    # 第3步：演示三种评分方案
    print("\n--- 演示三种评分方案 ---")
    demo_scoring_methods()

    print("\n" + "=" * 60)
    print("✅ Day 10 完成！去 Langfuse 控制台查看追踪和评分")
    print("=" * 60)