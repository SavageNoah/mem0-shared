"""
长期记忆层封装 (mem0)
为 second-brain 提供跨会话的用户记忆能力
"""
import os
import sys
import yaml

# 确保能找到 config.yaml - 先找同目录，再找上一级
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
# 用户的 .env 文件位置（回退）
ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")


def _load_dotenv():
    """从 .env 文件读取环境变量"""
    if not os.path.exists(ENV_PATH):
        return {}
    env = {}
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            env[key.strip()] = val.strip()
    return env


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_mem0_config(cfg: dict) -> dict:
    """根据 config.yaml 生成 mem0 Memory 配置字典"""
    mem_cfg = cfg.get("memory", {})
    if not mem_cfg.get("enabled", False):
        return None

    # 复用顶层 llm 配置作为回退
    top_llm = cfg.get("llm", {})
    mem_llm = mem_cfg.get("llm", {})

    api_key = mem_llm.get("api_key") or top_llm.get("api_key", "")
    base_url = mem_llm.get("base_url") or top_llm.get("base_url", "")
    model = mem_llm.get("model") or top_llm.get("model", "gpt-4o-mini")

    # 如果 config 里没有 key，尝试从 .env 回退
    if not api_key:
        env_vars = _load_dotenv()
        api_key = env_vars.get("OPENAI_API_KEY", "")
    if not base_url:
        env_vars = env_vars if 'env_vars' in dir() else _load_dotenv()
        base_url = env_vars.get("OPENAI_BASE_URL", "")

    vector_cfg = mem_cfg.get("vector_store", {})
    vs_provider = vector_cfg.get("provider", "chroma")
    vs_conf = vector_cfg.get("config", {})
    # 如果是 chroma，也转换路径
    if vs_provider == "chroma" and "path" in vs_conf:
        vs_conf = dict(vs_conf)
        vs_conf["path"] = _normalize_path(vs_conf["path"])

    # 读取 embedding 配置
    emb_cfg = cfg.get("embedding", {})
    emb_provider = emb_cfg.get("provider", "huggingface")
    emb_model = emb_cfg.get("model", "BAAI/bge-small-zh-v1.5")
    emb_base_url = emb_cfg.get("base_url", "")
    emb_local_path = emb_cfg.get("local_model_path", "")

    if emb_provider == "openai":
        # 使用云端 Embedding API（如火山方舟、OpenAI 等 OpenAI 兼容接口）
        embedder = {
            "provider": "openai",
            "config": {
                "model": emb_model,
                "api_key": api_key,
                "openai_base_url": emb_base_url or base_url or None,
            },
        }
    elif emb_provider == "local" or emb_provider == "huggingface":
        # 使用本地 HuggingFace 模型
        model_path = emb_local_path or emb_model or "BAAI/bge-small-zh-v1.5"
        embedder = {
            "provider": "huggingface",
            "config": {
                "model": model_path,
            },
        }
    else:
        # 默认回退到本地模型
        embedder = {
            "provider": "huggingface",
            "config": {
                "model": "BAAI/bge-small-zh-v1.5",
            },
        }

    config = {
        "vector_store": {
            "provider": vs_provider,
            "config": vs_conf,
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": model,
                "api_key": api_key,
                "openai_base_url": base_url or None,
            },
        },
        "embedder": embedder,
        # 暂时禁用 history_db，避免 SQLite 路径问题导致无法启动
    }
    return config


def _normalize_path(path: str) -> str:
    """标准化路径格式，确保返回绝对路径，防止创建非法文件夹"""
    if not path:
        return path

    import os
    import re

    # 先替换反斜杠
    normalized = path.replace("\\", "/")

    # 检测并拒绝看起来像驱动器号的相对路径（如 "D:", "C:" 作为文件夹名）
    # 这是防止创建非法文件夹的关键保护！
    parts = normalized.split("/")
    for i, part in enumerate(parts):
        # 如果部分看起来像驱动器号（只有 1-2 个字符且以冒号结尾）
        if len(part) <= 3 and ":" in part and re.match(r'^[a-zA-Z]:$', part):
            # 如果这不是第一个部分（说明是作为文件夹名），拒绝它！
            if i > 0:
                # 移除这个非法文件夹名，或者报错
                raise ValueError(f"Illegal folder name '{part}' in path: {path}")

    # 转换为绝对路径
    abs_path = os.path.abspath(normalized)

    # 确保不会返回只包含驱动器号的路径（如 "C:" 或 "D:"）
    if len(abs_path) == 2 and abs_path[1] == ":":
        abs_path += "/"

    return abs_path


# 全局缓存 Memory 实例
_memory_instance = None


def get_memory():
    """获取或初始化 mem0 Memory 实例"""
    global _memory_instance
    if _memory_instance is not None:
        return _memory_instance

    cfg = _load_config()
    mem0_cfg = _build_mem0_config(cfg)
    if mem0_cfg is None:
        return None

    try:
        from mem0 import Memory
        _memory_instance = Memory.from_config(config_dict=mem0_cfg)
        return _memory_instance
    except ImportError as e:
        print(f"[memory] mem0 import failed: {e}")
        return None
    except Exception as e:
        print(f"[memory] 初始化失败: {e}")
        return None


def add_memory(content: str, category: str = "general", metadata: dict = None, user_id: str = None) -> bool:
    """
    添加一条记忆
    :param content: 记忆内容
    :param category: 分类 (preference/fact/habit/project/general)
    :param metadata: 附加元数据
    :param user_id: 自定义 user_id（claude, gsd, hermes），None 则使用配置默认值
    """
    m = get_memory()
    if m is None:
        return False

    if user_id is None:
        cfg = _load_config()
        user_id = cfg.get("memory", {}).get("defaults", {}).get("user_id", "default_user")

    meta = {"category": category}
    if metadata:
        meta.update(metadata)

    try:
        m.add(content, user_id=user_id, metadata=meta)
        return True
    except Exception as e:
        print(f"[memory] add 失败: {e}")
        return False


def search_memory(query: str, limit: int = 5, user_id: str = None) -> list:
    """
    搜索相关记忆
    :param user_id: 可选，指定搜索哪个用户的记忆，None 则使用配置默认
    :return: 列表，每项包含 memory, score, metadata
    """
    m = get_memory()
    if m is None:
        return []

    if user_id is None:
        cfg = _load_config()
        user_id = cfg.get("memory", {}).get("defaults", {}).get("user_id", "default_user")

    try:
        search_query = query if query.strip() else "the"
        results = m.search(search_query, filters={"user_id": user_id}, limit=limit)
        if isinstance(results, list):
            return results
        if isinstance(results, dict) and "results" in results:
            return results["results"]
        return []
    except Exception as e:
        print(f"[memory] search failed: {e}")
        return []


def get_related_memories(content: str, limit: int = 3) -> str:
    """
    为给定内容获取相关记忆，返回简单文本摘要
    """
    results = search_memory(content, limit=limit)
    if not results:
        return ""

    parts = []
    for r in results:
        # 兼容不同版本的返回结构
        mem_text = ""
        if isinstance(r, dict):
            mem_text = r.get("memory", r.get("text", ""))
        elif isinstance(r, str):
            mem_text = r
        if mem_text:
            parts.append(f"- {mem_text}")

    return "\n".join(parts)


def remember_user_fact(fact: str, category: str = "fact") -> bool:
    """快捷函数：记住一条用户事实"""
    return add_memory(fact, category=category)


def remember_preference(preference: str) -> bool:
    """快捷函数：记住用户偏好"""
    return add_memory(preference, category="preference")


if __name__ == "__main__":
    # 简单测试
    print("[memory] 测试添加记忆...")
    ok = add_memory("用户喜欢投资科技股，特别关注 AI 和芯片行业", category="preference")
    print(f"添加结果: {ok}")

    print("\n[memory] 测试搜索...")
    related = get_related_memories("投资风格")
    print(f"相关记忆:\n{related}")
