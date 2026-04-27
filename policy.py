import os
import yaml


class PathPolicy:
    """
    统一管理 Agent 的读写权限策略：
    - 读取白名单
    - 写入白名单
    - 黑名单
    """

    def __init__(self):
        self.project_root = ""
        self.read_whitelist = []
        self.write_whitelist = []
        self.blacklist = []

        self.load_policy()

    def load_policy(self):
        """
        从 config/policy.yaml 加载策略配置
        """
        config_path = os.path.join("config", "policy.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"策略文件不存在：{config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        self.project_root = config.get("project_root", "")
        self.read_whitelist = config.get("read_whitelist", [])
        self.write_whitelist = config.get("write_whitelist", [])
        self.blacklist = config.get("blacklist", [])

    def is_allowed_read(self, file_path: str) -> bool:
        """
        判断是否允许读取文件
        """
        return self._check(file_path, self.read_whitelist)

    def is_allowed_write(self, file_path: str) -> bool:
        """
        判断是否允许写入文件
        """
        return self._check(file_path, self.write_whitelist)

    def _check(self, file_path: str, whitelist: list) -> bool:
        """
        通用路径检查逻辑：
        1. 黑名单优先拦截
        2. 白名单决定是否放行
        """
        full_path = os.path.abspath(file_path)

        # 黑名单优先
        for blocked in self.blacklist:
            if blocked and blocked in full_path:
                return False

        # 白名单判断
        for allowed in whitelist:
            if allowed and allowed in full_path:
                return True

        return False


# 全局单例
policy = PathPolicy()