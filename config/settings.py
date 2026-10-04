"""全局配置中心。

- 运行参数从 `.env` 读取（字段名小写 → 环境变量大写）
- 业务配置从 `config/knowledge_base.yaml` 读取（学校信息与知识库分类）

设计依据：docs/02-架构设计.md §10 配置项清单、§10.3 内置常量
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.errors import ConfigError

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_CONFIG_PATH = PROJECT_ROOT / "config" / "knowledge_base.yaml"

# ---- 内置常量（docs/02 §10.3，代码内定义，不可通过配置覆盖）----
UNCATEGORIZED_KEY = "uncategorized"      # 个人资料未选分类时的归属
PUBLIC_USER_ID = -1                      # Chroma 元数据中公共文档的 user_id 占位
COLLECTION_NAME = "campus_kb"            # Chroma 集合名
CHROMA_DISTANCE = "cosine"               # 集合距离度量，创建后不可改
CHUNK_SIZE = 500                         # 中文切片默认大小
CHUNK_OVERLAP = 80                       # 中文切片默认重叠
EMBED_BATCH_SIZE = 32                    # 入库时每批向量化与更新进度的条数
PBKDF2_ITERATIONS = 200_000              # 密码哈希迭代次数
METRICS_RETENTION_DAYS = 90              # 问答日志保留天数

# 上传白名单（FR-05）。图片后缀为二期 OCR 扩展（M6），是否放行还取决于
# `Settings.ocr_enabled`——关掉 OCR 时图片会被明确拒绝，而不是入库后无内容。
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"} | IMAGE_SUFFIXES
# 疑为扫描件的判定阈值：平均每页可提取字符数低于该值（docs/02 §4.2）
OCR_CHAR_THRESHOLD_PER_PAGE = 20

# knowledge_base.yaml 的必填结构
_REQUIRED_TOP_KEYS = ("school", "categories")
_REQUIRED_SCHOOL_KEYS = ("name", "contact")
_REQUIRED_CATEGORY_KEYS = ("key", "name")


class Settings(BaseSettings):
    """运行期配置。默认值即 docs/02 §10.1 中的默认值。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- 大模型 ----
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_temperature: float = 0.2

    # ---- 向量化 ----
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_api_key: str = ""
    embedding_base_url: str = ""

    # ---- 检索 ----
    retrieve_top_k: int = 5
    # 只作用于向量路的相似度阈值（DR-01）。CR-02：由 0.35 上调至 0.6——
    # M1 实测库外问题 top-1 相似度已达 0.42~0.60，0.35 完全不起过滤作用
    retrieve_score_threshold: float = 0.6
    vector_weight: float = 0.6
    bm25_weight: float = 0.4

    # ---- 对话 ----
    history_rounds: int = 5

    # ---- 服务 ----
    host: str = "0.0.0.0"
    port: int = 8501

    # ---- 上传 ----
    max_upload_mb: int = 20
    page_size: int = 10

    # ---- OCR（二期 M6：图片与扫描件入库）----
    # 关掉后图片后缀会被拒绝、扫描件 PDF 只保留「疑似扫描件」告警不入内容
    ocr_enabled: bool = True

    # ---- Agent 工具调度（二期 2.2）----
    # 关掉后问答一律走知识库检索（一键回退到二期前的行为）
    agent_enabled: bool = True

    # ---- 路径 ----
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path | None = None
    upload_dir: Path | None = None
    db_path: Path | None = None

    # ---- 管理员 ----
    admin_username: str = "admin"
    admin_password: str = "change-me-please"

    # ================= 派生路径 =================

    @property
    def chroma_path(self) -> Path:
        return self.chroma_dir or (self.data_dir / "chroma")

    @property
    def uploads_path(self) -> Path:
        return self.upload_dir or (self.data_dir / "uploads")

    @property
    def database_path(self) -> Path:
        return self.db_path or (self.data_dir / "app.db")

    @property
    def samples_path(self) -> Path:
        return self.data_dir / "samples"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        """创建运行期需要的目录。"""
        for path in (self.data_dir, self.chroma_path, self.uploads_path, self.samples_path):
            path.mkdir(parents=True, exist_ok=True)


def load_kb_config(path: Path | None = None) -> dict[str, Any]:
    """读取并校验 knowledge_base.yaml。

    配置驱动是 FR-27 的核心：换学校只改这个文件与资料，不改代码。
    配置缺失或结构错误时抛出 ConfigError，并附带正确示例。
    """
    config_path = path or KB_CONFIG_PATH

    if not config_path.exists():
        raise ConfigError(
            f"知识库配置文件不存在：{config_path}\n"
            "请创建该文件，示例：\n"
            "school:\n"
            '  name: "示例大学"\n'
            '  contact: "学生工作处 0597-0000000"\n'
            "categories:\n"
            '  - { key: freshman, name: 新生事务, desc: 报到、宿舍 }\n'
        )

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"知识库配置文件格式错误：{config_path}\n{exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"知识库配置根节点必须是键值结构：{config_path}")

    for key in _REQUIRED_TOP_KEYS:
        if key not in data:
            raise ConfigError(f"知识库配置缺少必填项 `{key}`：{config_path}")

    school = data["school"]
    if not isinstance(school, dict):
        raise ConfigError("知识库配置的 `school` 必须是键值结构")
    for key in _REQUIRED_SCHOOL_KEYS:
        if not school.get(key):
            raise ConfigError(f"知识库配置的 `school.{key}` 不能为空")

    categories = data["categories"]
    if not isinstance(categories, list) or not categories:
        raise ConfigError("知识库配置的 `categories` 必须是非空列表")

    seen: set[str] = set()
    for index, item in enumerate(categories):
        if not isinstance(item, dict):
            raise ConfigError(f"`categories[{index}]` 必须是键值结构")
        for key in _REQUIRED_CATEGORY_KEYS:
            if not item.get(key):
                raise ConfigError(f"`categories[{index}].{key}` 不能为空")
        category_key = str(item["key"])
        if category_key == UNCATEGORIZED_KEY:
            raise ConfigError(
                f"分类 key `{UNCATEGORIZED_KEY}` 为程序内置保留值，不能在配置中占用"
            )
        if category_key in seen:
            raise ConfigError(f"分类 key 重复：{category_key}")
        seen.add(category_key)

    return data


def category_options() -> list[dict[str, str]]:
    """返回用于下拉框的分类选项（含内置的「未分类」）。"""
    options = [
        {"key": str(item["key"]), "name": str(item["name"])}
        for item in load_kb_config()["categories"]
    ]
    options.append({"key": UNCATEGORIZED_KEY, "name": "未分类"})
    return options


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例。"""
    return Settings()


def reset_settings_cache() -> None:
    """清空配置缓存。仅供测试使用。"""
    get_settings.cache_clear()
