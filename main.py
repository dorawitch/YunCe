from dotenv import load_dotenv
from llm import MyLLM
from simple_agent import MySimpleAgent
from react_agent import MyReActAgent
from hello_agents import ToolRegistry


load_dotenv()
llm = MyLLM()
task = "完成图书管理系统设计"

# -----------------------
# Planner（项目经理）
# -----------------------
planner = MySimpleAgent(
    name="项目经理",
    llm=llm,
    system_prompt=(
        "你是一个项目经理，负责拆解任务提供代码需求。"
        "输出必须是结构化要点，，一定要十分精简不要写长文本。"
        "在回答问题的最后，输出'python .\test.py'"
    )
)

# -----------------------
# Executor（代码工程师 - ReAct）
# -----------------------
executor = MySimpleAgent(
    name="代码工程师",
    llm=llm,
    system_prompt="你是代码工程师，负责根据需求完成代码",

)

# Step 1: 规划
plan = planner.run(task)
print("\n📌 规划结果：\n", plan)

# Step 2: 执行
execution = executor.run(f"根据此计划完成代码：\n{plan}")
print("\n⚙️ 执行结果：\n", execution)

# Step 3: 总结

print("\n✅ 最终结果：\n", execution)