from dotenv import load_dotenv
from llm import MyLLM
# from simple_agent import MySimpleAgent
from react_agent import MyReActAgent
from hello_agents import ToolRegistry, ReActAgent, SimpleAgent

import asyncio

# 这里导入你的工具函数
from tools import write_file, run_training, read_file, run_shell   # 示例：你自己的工具

load_dotenv()
llm = MyLLM()
task = "使用工具read_file来读D:\benchmark\audio\acm_mm_competition_2026下的文件，搞清楚内容和任务，并为完成A1目标做规划。"

# # -----------------------
# # Planner（项目经理）
# # -----------------------
# planner = SimpleAgent(
#     name="数据科学竞赛解读专家",
#     llm=llm,
#     system_prompt=(
#         "你是一个数据科学竞赛解读专家，负责读取项目文档并拆解任务需求，读文档时要使用read_file工具，"
#         "目标代码文件的绝对路径为: C:\acm\AdoDAS2026-main，目标数据集的绝对路径为: D:\benchmark\audio\acm_mm_competition_2026\Train"
#         "训练集和测试集的路径分别为D:\benchmark\audio\acm_mm_competition_2026\Train\train和D:\benchmark\audio\acm_mm_competition_2026\Train\val"
#         "数据集的.csv文件路径为D:\benchmark\audio\acm_mm_competition_2026\Train\manifests_sch002_sch003"
#         "输出必须是结构化要点，一定要十分精简，不要写长文本。"
#     )
# )
# planner.add_tool(read_file) # 规划阶段只需要读取工具，执行阶段再使用写入和训练工具

# -----------------------
# Logger（史官）
# -----------------------
logger = SimpleAgent(
    name="史官",
    llm=llm,
    system_prompt=(
        "你是一个史官（实验记录员），负责记录数据科学代码工程师在算法调优过程中的关键信息，"
        "用于后续复盘、问题定位和持续优化。\n\n"

        "你的职责：\n"
        "1. 对当前阶段的实验过程进行分析总结；\n"
        "2. 提炼出后续算法优化最有价值的关键信息；\n"
        "3. 只保留真正重要的信息，删除无意义过程性描述；\n"
        "4. 必须输出结构化、精简、高价值日志，禁止长篇叙述。\n\n"

        "你重点记录以下内容：\n"
        "- 当前实验目标\n"
        "- 修改了哪些核心代码\n"
        "- 使用了哪些关键参数\n"
        "- 训练结果（如 Accuracy、Loss、F1 等）\n"
        "- 当前最主要的问题\n"
        "- 下一步最值得尝试的优化方向\n"
        "- 已验证无效的方法（避免重复踩坑）\n\n"

        "日志输出要求：\n"
        "1. 必须先完成分析总结，再调用 write_file 工具；\n"
        "2. 日志必须使用 Markdown 格式；\n"
        "3. 文件名必须严格使用：\n"
        "   C:\\acm\\AdoDAS2026-main\\log_{时间戳}.md\n"
        "4. 内容必须是结构化要点（bullet points），不能写成长段文字；\n"
        "5. 必须极度精简，强调可复用信息，而不是过程流水账；\n"
        "6. 日志的目标是：让未来的自己快速理解当前进展。\n"
    ),
)

logger.add_tool(write_file)


# -----------------------
# Executor（代码工程师 - ReAct）
# -----------------------
executor = ReActAgent(
    name="数据科学代码工程师",
    llm=llm,
    system_prompt=(
        "你是一个专业的数据科学代码工程师，负责读取项目文档、理解任务目标、修改代码并完成模型训练。\n\n"

        "你的核心目标：\n"
        "暂时不需要关注最终结果，因为我只是想先在小样本数据集上调试通baseline的训练流程，确保代码能够成功运行并产出结果。"
        "必须使用 GPU 成功跑通 baseline，完成 A1 目标。\n\n"
        # "并持续优化结果"

        "项目路径信息（必须牢记）：\n"
        "1. 项目代码根目录：\n"
        "   C:\\acm\\AdoDAS2026-main\n\n"

        "2. 数据集根目录：\n"
        "   D:\\benchmark\\audio\\acm_mm_competition_2026\\Train\n\n"

        "3. 训练集路径：\n"
        "   D:\\benchmark\\audio\\acm_mm_competition_2026\\Train\\train\n\n"

        "4. 验证集路径：\n"
        "   D:\\benchmark\\audio\\acm_mm_competition_2026\\Train\\val\n\n"

        "5. CSV 文件路径：\n"
        "   D:\\benchmark\\audio\\acm_mm_competition_2026\\Train\\manifests_sch002_sch003\n\n"

        "工具使用规则（必须严格遵守）：\n"

        "【读取目录结构 / 查看文件内容】\n"
        "必须使用 smart_read_file 工具。\n"
        "严禁使用 shell 命令查看文件（如 cat/type/more/less/head/tail）。\n\n"

        "【写代码 / 保存代码】\n"
        "必须使用 write_file 工具。\n\n"

        "【运行 Python 文件 / 模型训练】\n"
        "优先使用 run_training 工具。\n"
        "例如：run_training({'script_name': 'train.py'})\n\n"

        "【执行完整 shell 命令】\n"
        "仅在以下情况使用 run_shell：\n"
        "- 使用 uv 进行库的下载和管理\n"
        "- python xxx.py（完整 shell 命令）\n"
        "- cd xxx && python xxx.py\n"
        "- dir 等系统命令\n\n"

        "禁止错误用法：\n"
        "不要把读取文件内容交给 run_shell。\n"
        "不要使用 conda 或者 pip 进行库的安装或下载。\n"
        "不要把 shell 命令交给 run_training。\n\n"

        "执行原则：\n"
        "1. 优先先读项目结构，再读关键文件；\n"
        "2. 不清楚时必须主动查看文件，不要猜；\n"
        "3. 遇到报错必须定位根因，不允许盲目重复尝试；\n"
        "4. 优先保证 baseline 跑通，再考虑进一步优化；\n"
        "5. 每次修改代码都必须具有明确目的；\n"
        "6. 所有训练必须使用 GPU；\n"
        "7. 必须向用户持续展示训练进度（包括进度条 / epoch 输出 / accuracy 变化）；\n"
        "8. 不需要理解数据集业务含义，只需要围绕任务目标完成训练与优化。\n\n"

        "你的工作风格：\n"
        "像高级算法工程师一样行动：先判断、再读取、再修改、再验证，严禁无目的试错。"
    ),
    max_steps=80
)

executor.add_tool(write_file)
executor.add_tool(read_file)
executor.add_tool(run_training)
executor.add_tool(run_shell)


# # -----------------------
# # Step 1: 规划
# # -----------------------
# total_plan = ""
# async def agent_plan():
#     plan = planner.arun_stream(task)

#     print("\n📌 规划结果：\n")
    
#     async for event in plan:
#         # 安全地尝试获取 chunk 文本
#         try:
#             # 尝试访问 event.data['chunk']
#             if hasattr(event, 'data') and 'chunk' in event.data:
#                 text = event.data['chunk']
#                 if text: # 确保不是空内容
#                     print(text, end='', flush=True)
#                     total_plan += text # 将分块流式的规划拼接成一个完整的规划
#         except Exception:
#             # 如果解析失败，忽略该事件即可
#             pass

# asyncio.run(agent_plan())

# -----------------------
# Step 2: 执行（ReAct）
# -----------------------
single_execution = ""
async def agent_execution():
    execution = executor.arun_stream(task)

    print("\n⚙️ 执行结果：\n")
    
    async for event in execution:
        # 安全地尝试获取 chunk 文本
        try:
            # 尝试访问 event.data['chunk']
            if hasattr(event, 'data') and 'chunk' in event.data:
                text = event.data['chunk']
                if text: # 确保不是空内容
                    print(text, end='', flush=True)
                    single_execution += text
        except Exception:
            # 如果解析失败，忽略该事件即可
            pass

asyncio.run(agent_execution())

# -----------------------
# Step 2: 执行（ReAct）
# -----------------------
async def agent_log():
    log_information = logger.arun_stream(single_execution)

    print("\n⚙️ 记录日志信息：\n")
    
    async for event in log_information:
        # 安全地尝试获取 chunk 文本
        try:
            # 尝试访问 event.data['chunk']
            if hasattr(event, 'data') and 'chunk' in event.data:
                text = event.data['chunk']
                if text: # 确保不是空内容
                    print(text, end='', flush=True)
        except Exception:
            # 如果解析失败，忽略该事件即可
            pass

asyncio.run(agent_log())

# -----------------------
# Step 3: 总结
# -----------------------
# print("\n✅ 最终结果：\n", execution)