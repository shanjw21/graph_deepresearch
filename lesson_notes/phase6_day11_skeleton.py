"""
阶段六 · Day 11：离线评估 + LLM-as-a-Judge

任务：
1. 复用 Day 10 邮件 Agent
2. 构造测试数据集并上传到 Langfuse
3. 用 run_experiment 跑批量实验
4. 实现 LLM-as-a-Judge 自动评分
5. 去控制台查看实验结果

参考源码：03_trace_and_evaluation_langgraph_agents.ipynb Cell 32-50
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

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = "https://cloud.langfuse.com"


# ============================================
# 一、复用 Day 10 邮件 Agent（已实现）
# ============================================

class EmailState(TypedDict):
    email: Dict[str, Any]
    is_spam: Optional[bool]
    draft_response: Optional[str]
    messages: List[str]

llm = ChatOpenAI(model=model_name, temperature=0, base_url=base_url, api_key=api_key)
judge_llm = ChatOpenAI(model=model_name, temperature=0, base_url=base_url, api_key=api_key)

check_spam_instructions = """
你是一名资深邮件处理专家，擅长对邮件内容做判断，判断邮件是"SPAM"还是"HAM".
读取邮件:{email},
根据邮件发件人,主题，和内容，对邮件做出判断。
输出: "SPAM" or "HAM"
"""

def read_email(state: EmailState):
    email = state["email"]
    print(f"sender: {email.get('sender')}, subject: {email.get('subject')}")
    return {}

def classify_email(state: EmailState):
    email = state["email"]
    messages = state["messages"]
    prompt = check_spam_instructions.format(email=email)
    response = llm.invoke([HumanMessage(content=prompt)])
    is_spam = "spam" in response.content.lower()
    new_messages = messages
    if not is_spam:
        new_messages = new_messages + [response.content]
    return {"is_spam": is_spam, "messages": new_messages}

def handle_spam(state: EmailState):
    print("已标记为垃圾邮件")
    return {}

drafting_response_instructions = """
你是一名邮件助手管家，擅长以管家的口吻写邮件回复内容。
邮件:{email}
请根据以上邮件内容，编写回复邮件。
"""

def drafting_response(state: EmailState):
    email = state["email"]
    messages = state["messages"]
    prompt = drafting_response_instructions.format(email=email)
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "draft_response": response.content,
        "messages": messages + [response.content]
    }

def notify_mr_wayne(state: EmailState):
    email = state["email"]
    draft = state["draft_response"]
    print(f"韦恩先生，{email.get('sender')}来信: {email.get('subject')}\n建议回复: {draft[:80]}...")
    return {}

def route_email(state: EmailState):
    if state["is_spam"]:
        return "spam"
    return "legitimate"

# 构建图
email_graph = StateGraph(EmailState)
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("spam", handle_spam)
email_graph.add_node("legitimate", drafting_response)
email_graph.add_node("notify_mr_wayne", notify_mr_wayne)
email_graph.add_edge(START, "read_email")
email_graph.add_edge("read_email", "classify_email")
email_graph.add_conditional_edges("classify_email", route_email, ["spam", "legitimate"])
email_graph.add_edge("spam", END)
email_graph.add_edge("legitimate", "notify_mr_wayne")
email_graph.add_edge("notify_mr_wayne", END)
compiled_graph = email_graph.compile()
print("✅ 邮件 Agent 编译完成")


# ============================================
# 二、Langfuse 初始化
# ============================================

from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler


@observe()
def process_email(email_data):
    handler = CallbackHandler()
    result = compiled_graph.invoke(
        input={"email": email_data, "messages": []},
        config={"callbacks": [handler]}
    )
    return result


# ============================================
# 三、测试数据集（TODO：构造测试用例）
# ============================================

# TODO: 构造至少 6 条测试数据（3 条合法 + 3 条垃圾）
# 每条包含 input（邮件数据）和 expected_output（期望判定）
#
# 提示：
# input 格式：{"email": {"sender": "...", "subject": "...", "body": "..."}}
# expected_output 格式：{"is_spam": True/False}
#
# 参考源码：Cell 34-36

test_cases = [
    # === 3 条合法邮件 ===                                                                               
    {                                                                                                    
        "input": {"email": {                                                                             
            "sender": "京东客服",                                                                        
            "subject": "关于你近期订单的发票开具说明",                                                   
            "body": "尊敬的韦恩先生，你好！关于你在京东的近期订单，增值税电子普通发票已开具并推送至你的邮箱。如需纸质发票或抬头变更，请在7日内通过'我的订单-申请开票'发起，我们将尽快处理。"                      
        }},                                                                                              
        "expected_output": {"is_spam": False}                                                            
    },                                                    
    {
        "input": {"email": {
            "sender": "招商银行",
            "subject": "您的信用卡账单已出",                                                             
            "body": "韦恩先生您好，您尾号8866的信用卡2026年4月账单已生成，本期应还金额为3,256.80元，最后还款日为5月25日。请登录招商银行App查看详情。"                                                            
        }},                                               
        "expected_output": {"is_spam": False}                                                            
    },                                                                                                   
    {                                                                                                    
        "input": {"email": {                                                                             
            "sender": "行政部-李晓明",                                                                   
            "subject": "关于下周一全体员工会议的通知",                                                   
            "body": "韦恩先生你好，下周一（5月11日）上午10点将在3楼大会议室召开全体员工季度总结会，请提前准备好本部门的工作汇报材料。如有冲突请提前告知。"                                                        
        }},                                                                                              
        "expected_output": {"is_spam": False}                                                            
    },                                                                                                   
                                                                                                           
    # === 3 条垃圾邮件 ===                                                                               
    {                                                                                                    
        "input": {"email": {                                                                             
            "sender": "某数字货币项目方",                                                                
            "subject": "限时暴涨100倍，立即上车！",                                                      
            "body": "韦恩先生，我们新上线了一款数字货币，承诺稳稳赚、稳赚不赔！扫码加群，前100名赠送空投 名额，错过今天再等一年！"                                                                                
        }},                                                                                              
        "expected_output": {"is_spam": True}                                                             
    },                                                                                                   
    {                                                                                                    
        "input": {"email": {                                                                             
            "sender": "system-alert@bank-security.cn",                                                   
            "subject": "紧急：您的账户存在安全风险，请立即验证",                                         
            "body": "尊敬的用户，我们检测到您的银行账户在异地登录，为保障资金安全，请立即点击以下链接验证身份：http://bank-verify.xyz/auth。如12小时内未验证，账户将被冻结。"                                     
        }},                                                                                              
        "expected_output": {"is_spam": True}                                                             
    },                                                                                                   
    {
        "input": {"email": {                                                                             
            "sender": "幸运抽奖中心",                                                                    
            "subject": "恭喜您中奖100万！",                                                              
            "body": "韦恩先生恭喜！您的手机号在年度抽奖中被抽中一等奖100万元！请在48小时内添加客服微信luck100w 领取奖金，逾期作废。本次活动由国家公证处公证，真实有效！"                                       
        }},                                                                                              
        "expected_output": {"is_spam": True}                                                             
    }                  
]


# ============================================
# 四、上传数据集到 Langfuse（TODO）
# ============================================

def create_dataset():
    """
    TODO: 创建数据集并上传测试用例

    提示：
    1. langfuse = get_client()
    2. langfuse.create_dataset(name="email-agent-test", description="...")
    3. 遍历 test_cases，对每条用 langfuse.create_dataset_item(
           dataset_name="email-agent-test",
           input=item["input"],
           expected_output=item["expected_output"]
       )
    4. 打印数据集信息

    参考源码：Cell 34-36
    """
    # TODO: 实现该函数
    langfuse = get_client()
    try:                                                                                                     
        dataset = langfuse.get_dataset("email-agent-test")                                                   
        print(f"数据集已存在，包含 {len(dataset.items)} 个测试项，跳过创建")                                 
        return                                                                                               
    except Exception:                                                                                        
        pass  # 不存在，继续创建   
    langfuse.create_dataset(
        name="email-agent-test",
        description="邮件处理 Agent 测试数据集"
    )
    for case in test_cases:
        langfuse.create_dataset_item(
            dataset_name="email-agent-test",
            input=case["input"],
            expected_output=case["expected_output"]
        )
    dataset = langfuse.get_dataset("email-agent-test")
    print(f"数据集 'email-agent-test'包含 {len(dataset.items)} 个测试项")                              


# ============================================
# 五、LLM-as-a-Judge 评分函数（TODO）
# ============================================

judge_prompt = """
你是一名评估专家。请评估邮件分类 Agent 的输出是否正确。

邮件信息：{email}
期望分类：{expected}
实际分类：{actual}

评估标准：
- 如果实际分类与期望分类一致，输出 1
- 如果不一致，输出 0

只输出 1 或 0，不要其他内容。
"""

def judge_classification(email_data, expected_is_spam, actual_is_spam):
    """
    TODO: 用 LLM 判断分类是否正确

    提示：
    1. 用 judge_prompt.format(...) 构造提示词
    2. 用 judge_llm.invoke([HumanMessage(content=prompt)]) 调用 LLM
    3. 解析响应，返回 float 分数（0 或 1）

    参考源码：Cell 49-50
    """
    # TODO: 实现该函数
    prompt = judge_prompt.format(email=email_data,
                                 expected=expected_is_spam,
                                 actual=actual_is_spam)
    result = judge_llm.invoke([HumanMessage(content=prompt)])
    score = float(result.content.strip())
    return score
    


# ============================================
# 六、运行实验（TODO）
# ============================================




def run_evaluation():
    """
    TODO: 用 Langfuse Experiment SDK 跑批量评估

    提示（v4 API）：
    1. langfuse = get_client()
    2. dataset = langfuse.get_dataset("email-agent-test")
    3. 定义 task 函数：
       def my_task(*, item, **kwargs):
           email_data = item.input["email"]
           result = process_email(email_data)
           return result
    4. 用 dataset.run_experiment(name="exp-email-agent", task=my_task) 跑实验
    5. 打印完成信息

    参考源码：Cell 41-48
    """
    # TODO: 实现该函数
    langfuse = get_client()
    dataset = langfuse.get_dataset("email-agent-test")
    def my_task(*,item,**kwargs):
        email_data = item.input["email"]
        result = process_email(email_data=email_data)
        return result
    dataset.run_experiment(name="exp-email-agent",task=my_task)
    print("测试实验执行结束")
    


# ============================================
# 七、端到端测试
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 11：离线评估 + LLM-as-a-Judge")
    print("=" * 60)

    # 第1步：上传数据集
    print("\n--- 上传数据集 ---")
    create_dataset()

    # 第2步：跑实验
    print("\n--- 运行评估实验 ---")
    run_evaluation()

    # 第3步：手动测试 Judge 函数
    print("\n--- 测试 LLM-as-a-Judge ---")
    test_email = {"sender": "测试", "subject": "测试主题", "body": "测试内容"}
    score = judge_classification(test_email, expected_is_spam=False, actual_is_spam=False)
    print(f"Judge 评分: {score}（期望 1.0）")

    langfuse = get_client()
    langfuse.flush()

    print("\n" + "=" * 60)
    print("✅ Day 11 完成！去 Langfuse 控制台查看实验结果")
    print("=" * 60)
