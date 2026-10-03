"""数据访问层。

设计依据：docs/06-接口文档.md §1.5

范围说明：本模块按「用到才写」推进。Sprint 1 只实现 documents 与 ingest_tasks
（M1-11 的范围）；conversations / messages / feedback / qa_metrics 的访问函数
将随 Sprint 2、Sprint 3 各自的功能块补齐，避免提前造无人调用的代码。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.store.db import get_conn

logger = logging.getLogger(__name__)

DOC_ACTIVE = "active"
DOC_DELETED = "deleted"

TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_DONE = "done"
TASK_FAILED = "failed"

# 允许通过 update_task 修改的字段白名单，防止调用方拼出非法列名
_TASK_UPDATABLE = {"status", "total_chunks", "done_chunks", "error"}


@dataclass
class Document:
    id: int
    user_id: int | None
    is_public: bool
    filename: str
    filetype: str
    category: str
    size_bytes: int
    chunk_count: int
    status: str
    created_at: str
    updated_at: str


@dataclass
class IngestTask:
    id: int
    doc_id: int
    user_id: int | None
    status: str
    total_chunks: int
    done_chunks: int
    error: str | None
    created_at: str
    updated_at: str

    @property
    def progress(self) -> float:
        """入库进度，0.0 ~ 1.0。"""
        if self.total_chunks <= 0:
            return 1.0 if self.status == TASK_DONE else 0.0
        return min(self.done_chunks / self.total_chunks, 1.0)


PathLike = Path | str | None


def _to_document(row) -> Document:
    return Document(
        id=row["id"],
        user_id=row["user_id"],
        is_public=bool(row["is_public"]),
        filename=row["filename"],
        filetype=row["filetype"] or "",
        category=row["category"] or "",
        size_bytes=row["size_bytes"] or 0,
        chunk_count=row["chunk_count"] or 0,
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _to_task(row) -> IngestTask:
    return IngestTask(
        id=row["id"],
        doc_id=row["doc_id"],
        user_id=row["user_id"],
        status=row["status"],
        total_chunks=row["total_chunks"] or 0,
        done_chunks=row["done_chunks"] or 0,
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ==================== documents ====================


def create_document(
    *,
    filename: str,
    filetype: str,
    category: str,
    size_bytes: int = 0,
    user_id: int | None = None,
    is_public: bool = False,
    db_path: PathLike = None,
) -> int:
    """写入文档元数据，返回 doc_id。"""
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents (user_id, is_public, filename, filetype, category, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, 1 if is_public else 0, filename, filetype, category, size_bytes),
        )
        return int(cursor.lastrowid)


def get_document(doc_id: int, db_path: PathLike = None) -> Document | None:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return _to_document(row) if row else None


def list_documents(
    *,
    user_id: int | None = None,
    include_public: bool = False,
    page: int = 1,
    page_size: int = 10,
    category: str | None = None,
    db_path: PathLike = None,
) -> tuple[list[Document], int]:
    """分页查询文档。

    可见性规则：
    - `user_id` 非空        → 该用户自己的文档
    - `include_public=True` → 结果并集加上公共文档
    - `user_id` 为空且 `include_public=True` → 只返回公共文档（管理员公共文档列表）

    始终过滤 `status='deleted'`（DR-06），否则删除后仍会出现在列表中。
    """
    conditions = ["status != ?"]
    params: list[object] = [DOC_DELETED]

    if user_id is not None and include_public:
        conditions.append("(user_id = ? OR is_public = 1)")
        params.append(user_id)
    elif user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)
    elif include_public:
        conditions.append("is_public = 1")
    else:
        # 既没有用户身份也不要求公共文档：返回空集，绝不返回全部数据
        return [], 0

    if category:
        conditions.append("category = ?")
        params.append(category)

    where = " AND ".join(conditions)
    offset = max(page - 1, 0) * page_size

    with get_conn(db_path) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM documents WHERE {where}", params
        ).fetchone()["n"]
        rows = conn.execute(
            f"""
            SELECT * FROM documents
            WHERE {where}
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            [*params, page_size, offset],
        ).fetchall()

    return [_to_document(row) for row in rows], int(total)


def soft_delete_document(doc_id: int, db_path: PathLike = None) -> bool:
    """软删除：置 status='deleted'。向量的清理由调用方负责（见 docs/02 §4.9）。"""
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE documents
               SET status = ?, updated_at = datetime('now','localtime')
             WHERE id = ? AND status != ?
            """,
            (DOC_DELETED, doc_id, DOC_DELETED),
        )
        return cursor.rowcount > 0


def update_chunk_count(doc_id: int, chunk_count: int, db_path: PathLike = None) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE documents
               SET chunk_count = ?, updated_at = datetime('now','localtime')
             WHERE id = ?
            """,
            (chunk_count, doc_id),
        )


def exists_active_task(doc_id: int, db_path: PathLike = None) -> bool:
    """该文档是否已有未完成的任务。

    用于异步入库的幂等校验（DR-07）：Streamlit 会重跑脚本，
    若不校验，同一文件可能被重复提交入库。
    """
    with get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM ingest_tasks
             WHERE doc_id = ? AND status IN (?, ?)
             LIMIT 1
            """,
            (doc_id, TASK_PENDING, TASK_RUNNING),
        ).fetchone()
    return row is not None


# ==================== ingest_tasks ====================


def create_task(doc_id: int, user_id: int | None = None, db_path: PathLike = None) -> int:
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO ingest_tasks (doc_id, user_id, status) VALUES (?, ?, ?)",
            (doc_id, user_id, TASK_PENDING),
        )
        return int(cursor.lastrowid)


def get_task(task_id: int, db_path: PathLike = None) -> IngestTask | None:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM ingest_tasks WHERE id = ?", (task_id,)).fetchone()
    return _to_task(row) if row else None


def get_task_by_doc(doc_id: int, db_path: PathLike = None) -> IngestTask | None:
    """取该文档最新的一条任务。重试复用原任务（DR-05），故一个文档只有一条活跃任务。"""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM ingest_tasks WHERE doc_id = ? ORDER BY id DESC LIMIT 1",
            (doc_id,),
        ).fetchone()
    return _to_task(row) if row else None


def update_task(task_id: int, db_path: PathLike = None, **fields) -> None:
    """更新任务字段，只允许白名单内的列。"""
    unknown = set(fields) - _TASK_UPDATABLE
    if unknown:
        raise ValueError(f"不允许更新 ingest_tasks 的字段：{sorted(unknown)}")
    if not fields:
        return

    assignments = ", ".join(f"{name} = ?" for name in fields)
    values = [*fields.values(), task_id]

    with get_conn(db_path) as conn:
        conn.execute(
            f"""
            UPDATE ingest_tasks
               SET {assignments}, updated_at = datetime('now','localtime')
             WHERE id = ?
            """,
            values,
        )


def reset_task(task_id: int, db_path: PathLike = None) -> None:
    """把任务重置为待执行，用于失败重试（DR-05：复用原任务，不新建记录）。"""
    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE ingest_tasks
               SET status = ?, total_chunks = 0, done_chunks = 0, error = NULL,
                   updated_at = datetime('now','localtime')
             WHERE id = ?
            """,
            (TASK_PENDING, task_id),
        )


def list_tasks_by_user(
    user_id: int | None, *, page: int = 1, page_size: int = 10, db_path: PathLike = None
) -> tuple[list[IngestTask], int]:
    conditions = ["1 = 1"]
    params: list[object] = []

    if user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)

    where = " AND ".join(conditions)
    offset = max(page - 1, 0) * page_size

    with get_conn(db_path) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM ingest_tasks WHERE {where}", params
        ).fetchone()["n"]
        rows = conn.execute(
            f"""
            SELECT * FROM ingest_tasks
             WHERE {where}
             ORDER BY created_at DESC, id DESC
             LIMIT ? OFFSET ?
            """,
            [*params, page_size, offset],
        ).fetchall()

    return [_to_task(row) for row in rows], int(total)


def recover_stale_tasks(db_path: PathLike = None) -> int:
    """服务启动时，把残留的 running 任务标记为 failed。

    异步入库用后台线程实现，进程重启会丢失运行中的任务（docs/02 §12 技术债），
    此处让它们回到可重试状态。返回处理的条数。
    """
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE ingest_tasks
               SET status = ?, error = ?, updated_at = datetime('now','localtime')
             WHERE status = ?
            """,
            (TASK_FAILED, "服务重启导致任务中断，请重试", TASK_RUNNING),
        )
        count = cursor.rowcount

    if count:
        logger.warning("发现 %d 个中断的入库任务，已标记为失败待重试", count)
    return count
