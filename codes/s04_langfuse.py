from langfuse import get_client 
from dotenv import load_dotenv

load_dotenv()                                                                                                                                                                                                
langfuse = get_client()                                                                                                                                                                                                         
print("连接成功" if langfuse.auth_check() else "连接失败")     