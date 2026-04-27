import re
from typing import Optional, List, Iterator
from hello_agents import ReActAgent, HelloAgentsLLM, Config, Message, ToolRegistry


MY_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。

## 可用工具
{tools}

## 工作流程
Thought: 分析问题
Action: tool_name[参数] 或 Finish[答案]

## 当前任务
Question: {question}

## 历史
{history}

开始：
"""


class MyReActAgent(ReActAgent):
    """
    稳定版 ReAct Agent（防崩版本）
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):

        super().__init__(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            config=config
        )

        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else MY_REACT_PROMPT

        print(f"✅ {name} 初始化完成（稳定版ReAct）")

    # =========================
    # 安全解析（不会崩）
    # =========================
    def _parse_output(self, text: str):
        if not isinstance(text, str):
            text = getattr(text, "content", str(text))

        thought, action = "", ""

        try:
            t = re.search(r"Thought[:：]\s*(.*?)(?=Action[:：])", text, re.S)
            if t:
                thought = t.group(1).strip()

            a = re.search(r"Action[:：]\s*(.*)", text, re.S)
            if a:
                action = a.group(1).strip()

        except:
            return "", ""

        return thought, action

    # =========================
    # 安全 Action 解析
    # =========================
    def _parse_action(self, action: str):
        if not action:
            return None, None

        m = re.match(r"^\s*([A-Za-z_]\w*)\[(.*)\]\s*$", action, re.S)  # 支持多行
        if m:
            return m.group(1), m.group(2).strip()

        m = re.match(r"^\s*([A-Za-z_]\w*)\((.*)\)\s*$", action, re.S)  # 兼容括号
        if m:
            return m.group(1), m.group(2).strip()

        return None, None


    # =========================
    # 主运行逻辑（稳定版）
    # =========================
    def run(self, input_text: str, **kwargs) -> str:

        self.current_history = []
        last_action = None

        print(f"\n🤖 {self.name} 开始任务: {input_text}")

        for step in range(self.max_steps):

            print(f"\n--- 第 {step + 1} 步 ---")

            response = None

            # =========================
            # 1️⃣ LLM调用（强容错 + 重试）
            # =========================
            for i in range(3):
                try:
                    tools_desc = self.tool_registry.get_tools_description()
                    history_str = "\n".join(self.current_history[-6:])  # ⭐只保留最近历史

                    prompt = self.prompt_template.format(
                        tools=tools_desc,
                        question=input_text,
                        history=history_str
                    )

                    messages = [{"role": "user", "content": prompt}]

                    response = self.llm.invoke(
                        messages,
                        timeout=120,   # ⭐关键：防超时
                        **kwargs
                    )


                    break

                except Exception as e:
                    print(f"⚠️ LLM失败重试 {i+1}/3: {e}")
                    response = None

            # =========================
            # 2️⃣ LLM完全失败 → 不终止
            # =========================
            if response is None:
                print("❌ LLM完全失败，本轮跳过")

                self.current_history.append(
                    "Observation: LLM failed this step"
                )
                continue

            # =========================
            # 3️⃣ 统一格式
            # =========================
            if hasattr(response, "content"):
                response = response.content

            response = str(response).strip()

            if not response:
                print("⚠️ 空响应，跳过")
                continue

            # =========================
            # 4️⃣ 解析 Thought / Action
            # =========================
            thought, action = self._parse_output(response)

            print(f"\n🧠 Thought:\n{thought}")
            print(f"\n⚡ Action:\n{action}")

            # =========================
            # 5️⃣ Action为空 → 不终止
            # =========================
            if not action:
                print("⚠️ 无Action，继续下一步")
                continue

            # =========================
            # 6️⃣ 防死循环
            # =========================
            if action == last_action:
                print("⚠️ 重复Action，终止循环")
                break

            last_action = action

            # =========================
            # 7️⃣ Finish 处理
            # =========================
            if action.startswith("Finish["):
                result = re.sub(r"Finish\[(.*)\]", r"\1", action, flags=re.S).strip()


                self.add_message(Message(input_text, "user"))
                self.add_message(Message(result, "assistant"))
                return result

            # =========================
            # 8️⃣ Tool 执行（完全防崩）
            # =========================
            try:
                tool_name, tool_input = self._parse_action(action)

                if tool_name is None:
                    observation = "❌ 无效工具调用格式"
                else:
                    tool_response = self.tool_registry.execute_tool(
                        tool_name,
                        tool_input
                    )
                    observation = getattr(tool_response, "text", str(tool_response))

            except Exception as e:
                observation = f"❌ Tool执行失败: {str(e)}"

            # =========================
            # 9️⃣ 记录历史
            # =========================
            self.current_history.append(f"Thought: {thought}")
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        # =========================
        # 🔚 fallback（最终兜底）
        # =========================
        fallback = "⚠️ 达到最大步数但任务未完成（可能LLM不稳定）"

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(fallback, "assistant"))

        return fallback
    
    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        自定义的流式运行方法
        """
        print(f"🌊 {self.name} 开始流式处理: {input_text}")

        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        # 流式调用LLM
        full_response = ""
        print("📝 实时响应: ", end="")
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            print(chunk, end="", flush=True)
            yield chunk

        print()  # 换行

        # 保存完整对话到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
        print(f"✅ {self.name} 流式响应完成")