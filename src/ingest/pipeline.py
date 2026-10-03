"""入库主流程：解析 → 切片 → 分批向量化 → 写库。

设计依据：docs/02-架构设计.md §4.7 / §6.1、docs/06-接口文档.md §1.3

分批的意义（NFR 性能 + 前端进度）：按 `EMBED_BATCH_SIZE`（默认 32）一批写入，
每批结束更新一次 `ingest_tasks.done_chunks`，前端轮询即可看到进度推进。

异常约定（对照 docs/02 §11）：
- 文件为空 / 未能提取到文本 → `IngestError`，任务置 failed
- 向量化失败 → `IngestError`，**已写入的批次保留**，可重试
- 解析类错误（`ParseError` / `UnsupportedFileType`）原样上抛，
  由任务层写入 `ingest_tasks.error`，避免此处丢失异常类型
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_BATCH_SIZE,
    UNCATEGORIZED_KEY,
    Settings,
    get_settings,
)
from src.errors import IngestError
from src.files import stored_path
from src.ingest.loader import load_document
from src.ingest.splitter import split_text
from src.repository import create_document, update_chunk_count, update_task
from src.store.chroma import VectorStore, build_metadata

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """入库结果。字段与 docs/06 §3.1 一致。"""

    doc_id: int
    chunk_count: int
    needs_ocr: bool = False
    warning: str = ""
    elapsed_ms: int = 0


def _build_pairs(
    chunks,
    *,
    doc_id: int,
    user_id: int | None,
    is_public: bool,
    category: str,
    filename: str,
) -> list[tuple[str, dict]]:
    """把切片转成 `(文本, 元数据)` 列表，元数据结构见 docs/06 §3.2。"""
    return [
        (
            chunk.text,
            build_metadata(
                doc_id=doc_id,
                user_id=user_id,
                is_public=is_public,
                category=category,
                filename=filename,
                chunk_index=chunk.chunk_index,
            ),
        )
        for chunk in chunks
    ]


def ingest_file(
    path: str | Path,
    *,
    category: str = UNCATEGORIZED_KEY,
    is_public: bool = False,
    user_id: int | None = None,
    doc_id: int | None = None,
    task_id: int | None = None,
    store: VectorStore | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
    db_path: Path | str | None = None,
    settings: Settings | None = None,
) -> IngestResult:
    """把一个文件完整入库。

    Args:
        path: 待入库文件路径。
        category: 分类 key；留空归入内置的「未分类」。
        is_public: 是否公共文档（公共文档的 Chroma user_id 固定为 -1）。
        user_id: 上传者；个人资料必填。
        doc_id: 已存在的文档记录 id。为空则本函数自己创建 `documents` 记录。
        task_id: 关联的 `ingest_tasks` id。为空则不写进度（如命令行同步入库）。
        store: 向量库实例。为空则现场构造（会加载 Embedding 模型，批量场景应外部复用）。
        batch_size: 每批向量化条数。
        db_path: 测试用；默认走配置中的 SQLite 路径。

    Returns:
        IngestResult：切片数、是否疑似扫描件、告警、耗时。

    Raises:
        IngestError: 文件为空、未能提取到文本或向量化失败。
        ParseError / UnsupportedFileType: 由 `load_document` 原样抛出。
    """
    target = Path(path)
    started = time.perf_counter()
    resolved_settings = settings or get_settings()

    # 个人资料必须有归属，否则 Chroma 元数据会落成 user_id=-1 且 is_public=0，
    # 任何查询都过滤不到，等于静默丢数据
    if not is_public and user_id is None:
        raise IngestError("个人资料必须指定 user_id，否则入库后任何人都检索不到")

    load_result = load_document(target)
    if load_result.is_empty:
        raise IngestError(load_result.warning or "未能从文件中提取到文本")

    text = "\n\n".join(section.text for section in load_result.sections)
    chunks = split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not chunks:
        raise IngestError(load_result.warning or "未能从文件中提取到文本")

    resolved_category = (category or "").strip() or UNCATEGORIZED_KEY
    total = len(chunks)

    if doc_id is None:
        doc_id = create_document(
            filename=target.name,
            filetype=target.suffix.lower().lstrip("."),
            category=resolved_category,
            size_bytes=target.stat().st_size if target.is_file() else 0,
            user_id=user_id,
            is_public=is_public,
            db_path=db_path,
        )

    # 原始文件必须落盘一份：删除文档要删它，备份恢复要靠它重建向量。
    # 网页上传已在 stage_upload 落盘（路径相同，这里按路径相等跳过），
    # 批量入库（CLI）此前不落盘，导致备份恢复时「缺原始文件」无法重建（FB-4.x 实测）。
    _persist_original(
        target,
        doc_id=doc_id,
        user_id=user_id,
        is_public=is_public,
        settings=resolved_settings,
    )

    # 总切片数先落库，前端进度分母才有意义；done_chunks 归零重新计数
    if task_id is not None:
        update_task(task_id, db_path=db_path, total_chunks=total, done_chunks=0)

    vector_store = store if store is not None else VectorStore()
    pairs = _build_pairs(
        chunks,
        doc_id=doc_id,
        user_id=user_id,
        is_public=is_public,
        category=resolved_category,
        filename=target.name,
    )

    done = 0
    for start in range(0, total, batch_size):
        batch = pairs[start : start + batch_size]
        try:
            vector_store.add_chunks(batch)
        except Exception as exc:
            raise IngestError(
                f"向量化失败（已写入 {done}/{total} 个切片）：{exc}"
            ) from exc
        done += len(batch)
        if task_id is not None:
            update_task(task_id, db_path=db_path, done_chunks=done)

    update_chunk_count(doc_id, total, db_path=db_path)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "入库完成 doc_id=%s 切片=%d 耗时=%dms%s",
        doc_id,
        total,
        elapsed_ms,
        f" 告警：{load_result.warning}" if load_result.warning else "",
    )

    return IngestResult(
        doc_id=doc_id,
        chunk_count=total,
        needs_ocr=load_result.needs_ocr,
        warning=load_result.warning,
        elapsed_ms=elapsed_ms,
    )


def _persist_original(
    source: Path, *, doc_id: int, user_id: int | None, is_public: bool, settings: Settings
) -> None:
    """把原始文件复制到 uploads 的标准位置（若已在同一路径则跳过）。

    路径规则见 `src/files.py`；网页上传的 `stage_upload` 已按同一规则落盘，
    因此走异步任务时这里会因路径相同而跳过，不会重复复制。
    """
    destination = stored_path(
        owner_id=None if is_public else user_id,
        doc_id=doc_id,
        filename=source.name,
        settings=settings,
    )
    if source.resolve() == destination.resolve():
        return

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    except OSError as exc:
        raise IngestError(f"原始文件落盘失败：{source.name}（{exc}）") from exc
