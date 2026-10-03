"""原始文件的落盘位置约定。

设计依据：docs/02-架构设计.md §4.7 / §4.9。

约定：`data/uploads/<归属者>/<doc_id>_<清洗后的文件名>`。
- 归属者：个人资料用其 `user_id`；公共文档用占位值 `-1`（与 Chroma 元数据一致）。
- 带 `doc_id` 前缀：保证同名文件互不覆盖，且可由文档记录反推出落盘路径。

这些函数原先在 `src/ui/documents.py`。备份恢复（FR-24）需要在非 UI 场景
（批量入库、恢复重向量化）里复用同一套命名规则，故抽到本模块，避免两处实现漂移。
"""

from __future__ import annotations

import re
from pathlib import Path

from config.settings import PUBLIC_USER_ID, Settings

# Windows 与 POSIX 的非法文件名字符，统一替换为下划线
_ILLEGAL_NAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def sanitize_filename(name: str) -> str:
    """清洗文件名：去掉目录部分与非法字符，避免目录穿越与路径注入。

    Windows 风格的反斜杠在 POSIX 上不是分隔符，但上传者可能来自 Windows，
    因此先把两种分隔符统一，再取最后一段。
    """
    normalized = (name or "").strip().replace("\\", "/")
    base = _ILLEGAL_NAME_CHARS.sub("_", Path(normalized).name).strip()
    # 去掉开头的点，避免落成隐藏文件或 "." / ".."
    base = base.lstrip(".")
    return base or "未命名文件"


def uploads_dir(owner_id: int | None, *, settings: Settings) -> Path:
    """该归属者的上传目录：`data/uploads/<user_id>/`。

    公共文档的 user_id 用占位值 -1，与 Chroma 元数据里的约定保持一致。
    """
    directory = settings.uploads_path / str(PUBLIC_USER_ID if owner_id is None else owner_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def stored_path(
    *, owner_id: int | None, doc_id: int, filename: str, settings: Settings
) -> Path:
    """落盘路径。带 doc_id 前缀，保证同名文件互不覆盖，且可由文档记录反推出来。"""
    return uploads_dir(owner_id, settings=settings) / f"{doc_id}_{sanitize_filename(filename)}"
