from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_classic import hub
from langchain_classic.agents import create_react_agent
from langchain_classic.agents import AgentExecutor   # ✔ 正确位置

import re
from dotenv import load_dotenv
import os

load_dotenv()

# =========================
#  Tool：写文件
# =========================
def extract_python_code(text: str) -> str:
    """
    从包含 ```python ... ``` 的文本中提取纯代码内容
    """
    pattern = r"```python\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return "未找到Python代码块"
    
    print("提取到的代码块数量:", len(matches))

    # 如果有多个代码块，拼接
    return "\n\n".join(code.strip() for code in matches)

@tool
def write_file(content: str) -> str:
    """将生成的Python代码写入book_system.py"""
    with open("book_system.py", "w", encoding="utf-8") as f:
        content = extract_python_code(content)  # 提取纯代码内容
        f.write(content)
    return "✔ 已生成 book_system.py"



tools = [
    write_file,
]

# =========================
# LLM
# =========================
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("OPENAI_MODEL"),
    temperature=0.2
)


# =========================
#  Prompt（ReAct模板）
# =========================
prompt = hub.pull("hwchase17/react")

# =========================
#  Agent（新版）
# =========================
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# =========================
#  运行
# =========================
if __name__ == "__main__":
    task = """
请生成一个完整Python图书管理系统：
要求：
1. 类封装BookManager
2. 增删查改
3. 字典存储
4. 命令行交互
并写入book_system.py
"""

    result = agent_executor.invoke({"input": task})
    print(result)
