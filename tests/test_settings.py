"""配置中心单元测试。

覆盖 docs/03 §6.1 的 TC-U05 / TC-U06（knowledge_base.yaml 读取与异常）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import (
    UNCATEGORIZED_KEY,
    Settings,
    category_options,
    load_kb_config,
)
from src.errors import ConfigError

VALID_YAML = """
school:
  name: "测试大学"
  contact: "学生工作处 0000-0000000"
  notice: "以官方为准"
categories:
  - { key: freshman, name: 新生事务, desc: 报到 }
  - { key: admin, name: 行政事务, desc: 请假 }
"""


def write_yaml(tmp_path: Path, content: str, name: str = "kb.yaml") -> Path:
    target = tmp_path / name
    target.write_text(content, encoding="utf-8")
    return target


# ---------------- TC-U05：正常读取 ----------------


def test_load_kb_config_reads_school_and_categories(tmp_path: Path) -> None:
    config = load_kb_config(write_yaml(tmp_path, VALID_YAML))

    assert config["school"]["name"] == "测试大学"
    assert config["school"]["contact"] == "学生工作处 0000-0000000"
    assert [c["key"] for c in config["categories"]] == ["freshman", "admin"]


def test_repo_knowledge_base_yaml_is_valid() -> None:
    """仓库自带的配置文件必须能通过校验。"""
    config = load_kb_config()
    assert config["school"]["name"]
    assert len(config["categories"]) == 3


def test_category_options_appends_uncategorized_last(tmp_path: Path, monkeypatch) -> None:
    """分类下拉必须包含内置的「未分类」，且排在最后。"""
    import config.settings as settings_module

    monkeypatch.setattr(
        settings_module, "load_kb_config", lambda path=None: load_kb_config(write_yaml(tmp_path, VALID_YAML))
    )

    options = category_options()
    assert [o["key"] for o in options] == ["freshman", "admin", UNCATEGORIZED_KEY]
    assert options[-1]["name"] == "未分类"


# ---------------- TC-U06：异常分支 ----------------


def test_missing_file_raises_with_example(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_kb_config(tmp_path / "not-exist.yaml")
    # 异常信息必须附带正确示例，便于排错
    assert "categories" in str(excinfo.value)


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="格式错误"):
        load_kb_config(write_yaml(tmp_path, "school: [unclosed\n"))


def test_missing_top_level_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="categories"):
        load_kb_config(write_yaml(tmp_path, 'school:\n  name: "x"\n  contact: "y"\n'))


def test_missing_school_field_raises(tmp_path: Path) -> None:
    content = """
school:
  name: "测试大学"
categories:
  - { key: freshman, name: 新生事务 }
"""
    with pytest.raises(ConfigError, match="school.contact"):
        load_kb_config(write_yaml(tmp_path, content))


def test_empty_categories_raises(tmp_path: Path) -> None:
    content = """
school:
  name: "测试大学"
  contact: "0000"
categories: []
"""
    with pytest.raises(ConfigError, match="非空列表"):
        load_kb_config(write_yaml(tmp_path, content))


def test_duplicate_category_key_raises(tmp_path: Path) -> None:
    content = """
school:
  name: "测试大学"
  contact: "0000"
categories:
  - { key: freshman, name: 新生事务 }
  - { key: freshman, name: 重复 }
"""
    with pytest.raises(ConfigError, match="重复"):
        load_kb_config(write_yaml(tmp_path, content))


def test_reserved_uncategorized_key_raises(tmp_path: Path) -> None:
    content = """
school:
  name: "测试大学"
  contact: "0000"
categories:
  - { key: uncategorized, name: 未分类 }
"""
    with pytest.raises(ConfigError, match="保留值"):
        load_kb_config(write_yaml(tmp_path, content))


# ---------------- Settings 派生路径 ----------------


def test_settings_default_paths_are_derived_from_data_dir(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)

    assert settings.chroma_path == tmp_path / "chroma"
    assert settings.uploads_path == tmp_path / "uploads"
    assert settings.database_path == tmp_path / "app.db"
    assert settings.samples_path == tmp_path / "samples"


def test_settings_explicit_paths_win(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        chroma_dir=tmp_path / "custom-chroma",
        db_path=tmp_path / "custom.db",
    )

    assert settings.chroma_path == tmp_path / "custom-chroma"
    assert settings.database_path == tmp_path / "custom.db"


def test_max_upload_bytes_matches_mb(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, max_upload_mb=20)
    assert settings.max_upload_bytes == 20 * 1024 * 1024


def test_ensure_dirs_creates_all_paths(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    settings.ensure_dirs()

    assert settings.data_dir.is_dir()
    assert settings.chroma_path.is_dir()
    assert settings.uploads_path.is_dir()
    assert settings.samples_path.is_dir()


def test_defaults_match_design_document(tmp_path: Path) -> None:
    """默认值必须与 docs/02 §10.1 一致，防止实现与设计漂移。"""
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "deepseek"
    assert settings.llm_model == "deepseek-chat"
    assert settings.llm_temperature == 0.2
    assert settings.embedding_provider == "local"
    assert settings.embedding_model == "BAAI/bge-small-zh-v1.5"
    assert settings.retrieve_top_k == 5
    assert settings.retrieve_score_threshold == 0.6
    assert settings.vector_weight == 0.6
    assert settings.bm25_weight == 0.4
    assert settings.history_rounds == 5
    assert settings.host == "0.0.0.0"
    assert settings.port == 8501
    assert settings.max_upload_mb == 20
    assert settings.page_size == 10
    assert settings.admin_username == "admin"
