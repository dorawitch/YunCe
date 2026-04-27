import os
from typing import Any, Dict, List
import ast
import re
import subprocess
import time

from hello_agents import ToolRegistry
from hello_agents.tools import Tool, ToolErrorCode, ToolParameter, ToolResponse
from policy import policy


# =========================
# 1. 提取 Python 代码块（安全版）
# =========================
def extract_python_code(text: str) -> str:
    """
    支持三种格式：
    1) ```python ... ```
    2) \"\"\" ... \"\"\" 或 ''' ... '''
    3) 纯代码
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("❌ 输入为空，无法提取代码")

    def normalize(code: str) -> str:
        code = code.strip()
        code = re.sub(r"^\s*(python|py)\s*\n", "", code, flags=re.I)
        return code.strip()

    candidates: List[str] = []

    # markdown 代码块
    md_blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, re.S | re.I)
    if md_blocks:
        candidates.append(
            "\n\n".join(normalize(b) for b in md_blocks if normalize(b))
        )

    # 三引号代码块
    triple_blocks: List[str] = []
    triple_blocks += re.findall(r'"""(.*?)"""', text, re.S)
    triple_blocks += re.findall(r"'''(.*?)'''", text, re.S)
    if triple_blocks:
        candidates.append(
            "\n\n".join(normalize(b) for b in triple_blocks if normalize(b))
        )

    # 整段文本作为候选
    candidates.append(normalize(text))

    uniq: List[str] = []
    for c in candidates:
        if c and c not in uniq:
            uniq.append(c)

    # 优先返回可通过 AST 校验的代码
    for code in uniq:
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            continue

    # 都不合法时返回最长候选，让 validate_python 报错
    if uniq:
        return max(uniq, key=len)

    raise ValueError("❌ 未找到可用代码")


# =========================
# 2. Python 语法校验
# =========================
def validate_python(code: str) -> None:
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"❌ Python语法错误: {e}") from e


# =========================
# 3. 内部实现
# =========================
def _write_file_impl(content: str, output_file: str = "result.py") -> str:
    code = extract_python_code(content)
    validate_python(code)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(code)

    return f"✔ 已成功生成 {output_file}"


def _run_training_impl(script_name: str, timeout: int = 600) -> Dict[str, Any]:
    try:
        process = subprocess.Popen(
            ["python", "-u", script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is None:
            return {
                "success": False,
                "output": "执行异常: 无法读取训练输出流",
                "accuracy": 0.0,
            }

        all_output: List[str] = []
        start = time.monotonic()

        while True:
            # 超时控制
            if time.monotonic() - start > timeout:
                process.kill()
                process.wait()
                return {
                    "success": False,
                    "output": "训练超时（timeout）",
                    "accuracy": 0.0,
                }

            line = process.stdout.readline()

            if line:
                print(line, end="", flush=True)
                all_output.append(line)
            elif process.poll() is not None:
                break

        return_code = process.wait()
        output = "".join(all_output)

        match = re.search(
            r"Final\s*Accuracy[:=]?\s*(\d+\.?\d*)",
            output
        )
        accuracy = float(match.group(1)) if match else 0.0

        return {
            "success": return_code == 0,
            "output": output[-3000:],  # 防止输出过长
            "accuracy": accuracy,
        }

    except Exception as e:
        return {
            "success": False,
            "output": f"执行异常: {str(e)}",
            "accuracy": 0.0,
        }


# =========================
# 4. Tool：写文件
# =========================
class WriteFileTool(Tool):
    def __init__(self, output_file: str = "result.py"):
        super().__init__(
            name="write_file",
            description="提取输入中的 Python 代码并写入文件，自动进行语法校验。",
        )
        self.output_file = output_file

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="content",
                type="string",
                description="要写入文件的代码内容",
                required=True,
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        content = parameters.get("content")

        if content is None:
            content = parameters.get("input")

        if not isinstance(content, str) or not content.strip():
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少参数 content（不能为空）",
            )

        # 白名单控制
        if not policy.is_allowed_write(self.output_file):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    f"禁止写入文件：{self.output_file}\n"
                    f"该路径不在写入白名单中。"
                ),
            )

        try:
            message = _write_file_impl(content, self.output_file)

            return ToolResponse.success(
                text=message,
                data={
                    "output_file": self.output_file,
                },
            )

        except ValueError as e:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_FORMAT,
                message=str(e),
            )

        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"写文件失败: {e}",
            )


# =========================
# 5. Tool：运行训练
# =========================
class RunTrainingTool(Tool):
    def __init__(self, default_timeout: int = 600):
        super().__init__(
            name="run_training",
            description="执行 Python 训练脚本并返回结果。",
        )
        self.default_timeout = default_timeout

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="script_name",
                type="string",
                description="脚本路径",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="超时时间（秒）",
                required=False,
                default=self.default_timeout,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        script_name = parameters.get("script_name")

        if script_name is None:
            script_name = parameters.get("input")

        if not isinstance(script_name, str) or not script_name.strip():
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少参数 script_name",
            )

        try:
            timeout = int(
                parameters.get("timeout", self.default_timeout)
            )
        except Exception:
            timeout = self.default_timeout

        result = _run_training_impl(
            script_name.strip(),
            timeout
        )

        if result.get("success"):
            return ToolResponse.success(
                text=f"训练完成",
                data=result,
            )

        return ToolResponse.error(
            code=ToolErrorCode.EXECUTION_ERROR,
            message=result.get("output", "训练失败"),
        )


# =========================
# 6. Tool：智能读文件（目录 + 文件）
# =========================
class SmartReadFileTool(Tool):
    """
    智能读取项目文件：

    情况1：
    只给 base_path
    → 返回目录结构

    情况2：
    给了 base_path + file_name
    → 读取文件内容

    情况3：
    文件不存在
    → 返回目录结构 + 提示
    """

    def __init__(self):
        super().__init__(
            name="smart_read_file",
            description=(
                "智能读取项目文件。"
                "若未提供 file_name，则返回目录结构；"
                "若提供 file_name，则读取对应文件内容。"
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="base_path",
                type="string",
                description="项目目录路径，例如 D:\\project",
                required=True,
            ),
            ToolParameter(
                name="file_name",
                type="string",
                description="要读取的文件名，可选，例如 train.py",
                required=False,
            ),
            ToolParameter(
                name="max_chars",
                type="integer",
                description="最大读取字符数，默认 8000",
                required=False,
                default=8000,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        base_path = parameters.get("base_path")
        file_name = parameters.get("file_name")
        max_chars = parameters.get("max_chars", 8000)

        if base_path is None:
            base_path = parameters.get("input")

        if not isinstance(base_path, str) or not base_path.strip():
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少参数 base_path（不能为空）",
            )

        if not os.path.exists(base_path):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"目录不存在：{base_path}",
            )

        if not os.path.isdir(base_path):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"不是目录：{base_path}",
            )

        # 白名单控制
        if not policy.is_allowed_read(base_path):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    f"禁止访问目录：{base_path}\n"
                    f"该路径不在读取白名单中。"
                ),
            )

        try:
            items = os.listdir(base_path)

            # 情况1：没给 file_name，只返回目录结构
            if not file_name:
                return ToolResponse.success(
                    text=(
                        f"目录内容如下：\n"
                        + "\n".join(items)
                    ),
                    data={
                        "base_path": base_path,
                        "items": items,
                        "count": len(items),
                    },
                )

            # 情况2：文件不存在
            if file_name not in items:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=(
                        f"文件不存在：{file_name}\n\n"
                        f"当前目录内容如下：\n"
                        + "\n".join(items)
                    ),
                )

            full_path = os.path.join(base_path, file_name)

            if not os.path.isfile(full_path):
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=f"{file_name} 不是文件，可能是文件夹",
                )

            if not policy.is_allowed_read(full_path):
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=(
                        f"禁止读取文件：{full_path}\n"
                        f"该路径不在读取白名单中。"
                    ),
                )

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_length = len(content)

            if original_length > max_chars:
                content = (
                    content[:max_chars]
                    + "\n\n[文件内容过长，已自动截断]"
                )

            return ToolResponse.success(
                text=(
                    f"成功读取文件：{full_path}\n\n"
                    f"目录内容：\n"
                    + "\n".join(items)
                    + "\n\n文件内容如下：\n"
                    + content
                ),
                data={
                    "base_path": base_path,
                    "file_name": file_name,
                    "full_path": full_path,
                    "content_length": original_length,
                    "directory_items": items,
                },
            )

        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"读取失败：{str(e)}",
            )

# =========================
# 5.5 Tool：运行 Shell 命令
# =========================
class RunShellTool(Tool):
    """
    专门执行 shell 命令，例如：

    python result.py
    pip install -r requirements.txt
    dir
    cd xxx && python train.py

    避免 LLM 把 shell 命令错误传给 run_training
    """

    def __init__(self, default_timeout: int = 600):
        super().__init__(
            name="run_shell",
            description=(
                "执行 Shell 命令（如 python result.py、pip install 等）并返回结果。"
                "注意："
                "1. shell 命令只能用于运行程序、安装依赖、执行系统命令；"
                "2. 不允许使用 shell 查看文件内容（如 type、cat、more、less 等）；"
                "3. 查看文件内容必须使用 smart_read_file 工具；"
                "4. 若只是执行 Python 文件，请优先使用 run_training；"
                "5. 若是完整 shell 命令（如 python result.py、pip install -r requirements.txt），才使用 run_shell。"
            ),
        )
        self.default_timeout = default_timeout

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="要执行的 shell 命令，例如 python result.py",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="超时时间（秒）",
                required=False,
                default=self.default_timeout,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        command = parameters.get("command")

        if command is None:
            command = parameters.get("input")

        if not isinstance(command, str) or not command.strip():
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    "❌ 缺少参数 command（不能为空）\n\n"
                    "示例：\n"
                    "run_shell({'command': 'python result.py'})"
                ),
            )

        command = command.strip()

        # =========================
        # 安全兜底：禁止用 shell 查看文件内容
        # =========================
        forbidden_patterns = [
            "cat ",
            "type ",
            "more ",
            "less ",
            "tail ",
            "head ",
        ]

        lower_command = command.lower()

        for pattern in forbidden_patterns:
            if pattern in lower_command:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=(
                        "❌ 检测到你正在尝试使用 Shell 查看文件内容，这是被禁止的。\n\n"
                        "请使用 smart_read_file 工具读取文件，而不是 run_shell。\n\n"
                        "错误命令：\n"
                        f"{command}\n\n"
                        "正确示例：\n"
                        "smart_read_file({\n"
                        "    'base_path': 'D:\\\\project',\n"
                        "    'file_name': 'train.py'\n"
                        "})"
                    ),
                )

        try:
            timeout = int(
                parameters.get("timeout", self.default_timeout)
            )
        except Exception:
            timeout = self.default_timeout

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if process.stdout is None:
                return ToolResponse.error(
                    code=ToolErrorCode.EXECUTION_ERROR,
                    message="❌ 执行异常：无法读取输出流",
                )

            all_output: List[str] = []
            start = time.monotonic()

            while True:
                if time.monotonic() - start > timeout:
                    process.kill()
                    process.wait()

                    return ToolResponse.error(
                        code=ToolErrorCode.EXECUTION_ERROR,
                        message=(
                            "❌ Shell 命令执行超时（timeout）\n\n"
                            f"命令：{command}\n"
                            f"超时时间：{timeout} 秒"
                        ),
                    )

                line = process.stdout.readline()

                if line:
                    print(line, end="", flush=True)
                    all_output.append(line)

                elif process.poll() is not None:
                    break

            return_code = process.wait()
            output = "".join(all_output).strip()

            # 输出兜底（防止空输出）
            if not output:
                output = "[命令执行完成，但没有输出内容]"

            output = output[-3000:]  # 防止过长

            # =========================
            # 成功格式化输出
            # =========================
            if return_code == 0:
                formatted_text = (
                    "✅ Shell 命令执行成功\n\n"
                    f"【执行命令】\n{command}\n\n"
                    f"【返回码】\n{return_code}\n\n"
                    f"【执行结果】\n{output}"
                )

                return ToolResponse.success(
                    text=formatted_text,
                    data={
                        "command": command,
                        "output": output,
                        "return_code": return_code,
                    },
                )

            # =========================
            # 失败格式化输出
            # =========================
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=(
                    "❌ Shell 命令执行失败\n\n"
                    f"【执行命令】\n{command}\n\n"
                    f"【返回码】\n{return_code}\n\n"
                    f"【错误输出】\n{output}"
                ),
            )

        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=(
                    "❌ 执行 Shell 命令失败\n\n"
                    f"【执行命令】\n{command}\n\n"
                    f"【异常信息】\n{str(e)}"
                ),
            )


# =========================
# 7. 对外导出
# =========================
write_file = WriteFileTool()
run_training = RunTrainingTool()
read_file = SmartReadFileTool()
run_shell = RunShellTool()


def register_tools(tool_registry: ToolRegistry) -> ToolRegistry:
    tool_registry.register_tool(write_file)
    tool_registry.register_tool(run_training)
    tool_registry.register_tool(read_file)
    tool_registry.register_tool(run_shell)
    return tool_registry