"""数据访问层。

设计依据：docs/06-接口文档.md §1.5

范围说明：本模块按「用到才写」推进。Sprint 1 实现了 documents 与 ingest_tasks
（M1-11）；FB-2.2 的问答主链路落库需要，补齐 conversations / messages /
message_sources / qa_metrics 的写入与读取；M2-06 补会话切换 / 重命名 / 删除，
M2-05 补 feedback；M3-08 补 `cleanup_metrics` 日志清理。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from config.settings import METRICS_RETENTION_DAYS
from src.store.db import get_conn, retry_on_write_lock

if TYPE_CHECKING:  # 仅用于类型标注，避免数据访问层反向依赖向量层
    from src.store.chroma import SearchHit

logger = logging.getLogger(__name__)

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

RATING_USEFUL = "useful"
RATING_USELESS = "useless"

DOC_ACTIVE = "active"
DOC_DELETED = "deleted"

TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_DONE = "done"
TASK_FAILED = "failed"

# 允许通过 update_task 修改的字段白名单，防止调用方拼出非法列名
_TASK_UPDATABLE = {"status", "total_chunks", "done_chunks", "error", "warning"}


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
    warning: str | None
    created_at: str
    updated_at: str

    @property
    def progress(self) -> float:
        """入库进度，0.0 ~ 1.0。"""
        if self.total_chunks <= 0:
            return 1.0 if self.status == TASK_DONE else 0.0
        return min(self.done_chunks / self.total_chunks, 1.0)


PathLike = Path | str | None


@dataclass
class Conversation:
    id: int
    user_id: int
    title: str
    created_at: str
    updated_at: str


@dataclass
class Message:
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: str


def _to_conversation(row) -> Conversation:
    return Conversation(
        id=row["id"],
        user_id=row["user_id"],
        title=row["title"] or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _to_message(row) -> Message:
    return Message(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
    )


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
        warning=row["warning"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ==================== users ====================
#
# 注意区分命名：上面的 `ROLE_USER` / `ROLE_ASSISTANT` 是**消息角色**（messages.role），
# 这里的是**账号角色**（users.role）与账号状态（users.status）。

USER_ROLE_ADMIN = "admin"
USER_ROLE_USER = "user"
USER_STATUS_ACTIVE = "active"
USER_STATUS_DISABLED = "disabled"


@dataclass
class User:
    id: int
    username: str
    password_hash: str
    display_name: str
    role: str
    status: str
    created_at: str

    @property
    def is_admin(self) -> bool:
        return self.role == USER_ROLE_ADMIN

    @property
    def is_active(self) -> bool:
        return self.status == USER_STATUS_ACTIVE


def _to_user(row) -> User:
    return User(
        id=row["id"],
        username=row["username"],
        password_hash=row["password_hash"],
        display_name=row["display_name"] or "",
        role=row["role"],
        status=row["status"],
        created_at=row["created_at"],
    )


@retry_on_write_lock
def create_user(
    *,
    username: str,
    password_hash: str,
    display_name: str | None = None,
    role: str = USER_ROLE_USER,
    db_path: PathLike = None,
) -> int:
    """创建账号，返回 user_id。用户名重复时由 UNIQUE 约束抛 IntegrityError。"""
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (username, password_hash, display_name, role)
            VALUES (?, ?, ?, ?)
            """,
            (username, password_hash, (display_name or "").strip() or username, role),
        )
        return int(cursor.lastrowid)


def get_user(user_id: int, db_path: PathLike = None) -> User | None:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _to_user(row) if row else None


def get_user_by_name(username: str, db_path: PathLike = None) -> User | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return _to_user(row) if row else None


@retry_on_write_lock
def update_password(user_id: int, password_hash: str, db_path: PathLike = None) -> bool:
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id)
        )
        return cursor.rowcount > 0


@retry_on_write_lock
def update_display_name(user_id: int, display_name: str, db_path: PathLike = None) -> bool:
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            "UPDATE users SET display_name = ? WHERE id = ?",
            ((display_name or "").strip(), user_id),
        )
        return cursor.rowcount > 0


def list_users(
    *,
    page: int = 1,
    page_size: int = 10,
    keyword: str | None = None,
    db_path: PathLike = None,
) -> tuple[list[User], int]:
    """分页列出账号（管理员页 FR-16）。`keyword` 模糊匹配用户名与显示名。"""
    conditions = ["1 = 1"]
    params: list[object] = []
    if keyword and keyword.strip():
        conditions.append("(username LIKE ? OR display_name LIKE ?)")
        like = f"%{keyword.strip()}%"
        params += [like, like]

    where = " AND ".join(conditions)
    offset = max(page - 1, 0) * page_size

    with get_conn(db_path) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM users WHERE {where}", params
        ).fetchone()["n"]
        rows = conn.execute(
            f"""
            SELECT * FROM users
             WHERE {where}
             ORDER BY created_at DESC, id DESC
             LIMIT ? OFFSET ?
            """,
            [*params, page_size, offset],
        ).fetchall()
    return [_to_user(row) for row in rows], int(total)


@retry_on_write_lock
def set_user_status(user_id: int, status: str, db_path: PathLike = None) -> bool:
    """启用 / 禁用账号（FR-16）。禁用后该账号无法登录，且已建立的登录态立即失效。"""
    if status not in (USER_STATUS_ACTIVE, USER_STATUS_DISABLED):
        raise ValueError(f"未知的账号状态：{status}")
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            "UPDATE users SET status = ? WHERE id = ?", (status, user_id)
        )
        return cursor.rowcount > 0


@retry_on_write_lock
def delete_user(user_id: int, db_path: PathLike = None) -> None:
    """删除账号在 SQLite 中的**全部**关联数据（FR-25，单事务）。

    顺序按外键依赖：feedback → message_sources → messages → conversations
    → ingest_tasks → documents → schedules → summaries → users。

    向量库与磁盘文件不在这里处理（数据访问层不碰向量层）——
    由 `auth.service.delete_account` 先清向量与文件、再调本函数，
    保证「要么全删，要么不留半删状态」（docs/02 §4.9）。
    """
    with get_conn(db_path) as conn:
        conn.execute(
            """
            DELETE FROM feedback
             WHERE user_id = ?
                OR message_id IN (
                    SELECT m.id FROM messages m
                     JOIN conversations c ON c.id = m.conversation_id
                    WHERE c.user_id = ?
                )
            """,
            (user_id, user_id),
        )
        conn.execute(
            """
            DELETE FROM message_sources
             WHERE message_id IN (
                SELECT m.id FROM messages m
                 JOIN conversations c ON c.id = m.conversation_id
                WHERE c.user_id = ?
             )
            """,
            (user_id,),
        )
        conn.execute(
            """
            DELETE FROM messages
             WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id = ?)
            """,
            (user_id,),
        )
        conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM ingest_tasks WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM documents WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM schedules WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM summaries WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


# ==================== documents ====================


@retry_on_write_lock
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


@retry_on_write_lock
def soft_delete_document(doc_id: int, db_path: PathLike = None) -> bool:
    """软删除：置 status='deleted'。向量的清理由调用方负责（见 docs/02 §4.9）。

    同时清掉该文档已生成的课件总结 / 复习提纲（二期 2.4）：源文档已删，
    留着这些产物只会在「课件助手」里变成打不开的悬空记录。
    """
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE documents
               SET status = ?, updated_at = datetime('now','localtime')
             WHERE id = ? AND status != ?
            """,
            (DOC_DELETED, doc_id, DOC_DELETED),
        )
        if cursor.rowcount > 0:
            conn.execute("DELETE FROM summaries WHERE doc_id = ?", (doc_id,))
        return cursor.rowcount > 0


@retry_on_write_lock
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


@retry_on_write_lock
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


@retry_on_write_lock
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


@retry_on_write_lock
def reset_task(task_id: int, db_path: PathLike = None) -> None:
    """把任务重置为待执行，用于失败重试（DR-05：复用原任务，不新建记录）。"""
    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE ingest_tasks
               SET status = ?, total_chunks = 0, done_chunks = 0, error = NULL,
                   warning = NULL, updated_at = datetime('now','localtime')
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


@retry_on_write_lock
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


# ==================== conversations / messages ====================


@retry_on_write_lock
def create_conversation(
    user_id: int, *, title: str | None = None, db_path: PathLike = None
) -> int:
    """新建会话，返回 conversation_id。空标题存 NULL，展示层再兜底。"""
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
            (user_id, (title or "").strip() or None),
        )
        return int(cursor.lastrowid)


def get_conversation(conversation_id: int, db_path: PathLike = None) -> Conversation | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return _to_conversation(row) if row else None


def list_conversations(
    user_id: int, *, page: int = 1, page_size: int = 20, db_path: PathLike = None
) -> tuple[list[Conversation], int]:
    """按最近活动倒序列出会话（FR-12）。"""
    offset = max(page - 1, 0) * page_size
    with get_conn(db_path) as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM conversations WHERE user_id = ?", (user_id,)
        ).fetchone()["n"]
        rows = conn.execute(
            """
            SELECT * FROM conversations
             WHERE user_id = ?
             ORDER BY updated_at DESC, id DESC
             LIMIT ? OFFSET ?
            """,
            (user_id, page_size, offset),
        ).fetchall()
    return [_to_conversation(row) for row in rows], int(total)


@retry_on_write_lock
def rename_conversation(
    conversation_id: int, title: str, db_path: PathLike = None
) -> bool:
    """重命名会话。空标题存 NULL，展示层再兜底。"""
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE conversations
               SET title = ?, updated_at = datetime('now','localtime')
             WHERE id = ?
            """,
            ((title or "").strip() or None, conversation_id),
        )
        return cursor.rowcount > 0


@retry_on_write_lock
def delete_conversation(conversation_id: int, db_path: PathLike = None) -> int:
    """删除会话及其全部消息、引用与反馈，返回删除的消息条数。

    几张表之间有外键，必须按依赖顺序删且放在**同一个事务**里，
    否则会留下「消息删了、引用还在」的半删状态（docs/02 §4.9 的同类要求）。
    """
    with get_conn(db_path) as conn:
        conn.execute(
            """
            DELETE FROM feedback WHERE message_id IN
                (SELECT id FROM messages WHERE conversation_id = ?)
            """,
            (conversation_id,),
        )
        conn.execute(
            """
            DELETE FROM message_sources WHERE message_id IN
                (SELECT id FROM messages WHERE conversation_id = ?)
            """,
            (conversation_id,),
        )
        removed = conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).rowcount
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    return int(removed)


@retry_on_write_lock
def add_message(
    conversation_id: int, role: str, content: str, db_path: PathLike = None
) -> int:
    """追加一条消息，并顺带刷新会话的 updated_at（列表按最近活动排序）。"""
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )
        conn.execute(
            """
            UPDATE conversations
               SET updated_at = datetime('now','localtime')
             WHERE id = ?
            """,
            (conversation_id,),
        )
        return int(cursor.lastrowid)


def list_messages(
    conversation_id: int, *, limit: int | None = None, db_path: PathLike = None
) -> list[Message]:
    """按时间正序返回消息。

    `limit` 表示**取最近 N 条**（用于多轮改写），返回时仍按时间正序——
    顺序反了会把历史倒着喂给模型。
    """
    with get_conn(db_path) as conn:
        if limit is not None and limit > 0:
            rows = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM messages WHERE conversation_id = ?
                     ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
                """,
                (conversation_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
    return [_to_message(row) for row in rows]


# ==================== message_sources / qa_metrics ====================


@retry_on_write_lock
def add_sources(
    message_id: int, sources: Sequence["SearchHit"], db_path: PathLike = None
) -> None:
    """写入引用溯源。`matched_by` 记录该来源是向量路 / 关键词路 / 两路命中（DR-15）。"""
    if not sources:
        return
    with get_conn(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO message_sources (message_id, doc_id, filename, snippet, score, matched_by)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (message_id, hit.doc_id, hit.filename, hit.text, hit.score, hit.matched_by)
                for hit in sources
            ],
        )


def list_sources_by_message(message_id: int, db_path: PathLike = None) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM message_sources WHERE message_id = ? ORDER BY id ASC",
            (message_id,),
        ).fetchall()
    return [dict(row) for row in rows]


# ==================== feedback ====================


@retry_on_write_lock
def add_feedback(
    message_id: int, user_id: int, rating: str, db_path: PathLike = None
) -> bool:
    """写一条反馈。

    表上有 `UNIQUE (message_id, user_id)`：同一用户对同一回答只记一次，
    重复提交被忽略并返回 False，界面据此把按钮置为不可重复点击（FR-18）。
    """
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO feedback (message_id, user_id, rating) VALUES (?, ?, ?)",
            (message_id, user_id, rating),
        )
        return cursor.rowcount > 0


def get_feedback(message_id: int, user_id: int, db_path: PathLike = None) -> str | None:
    """取该用户对该回答的反馈，用于回显已选态。"""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT rating FROM feedback WHERE message_id = ? AND user_id = ?",
            (message_id, user_id),
        ).fetchone()
    return row["rating"] if row else None


@retry_on_write_lock
def add_metric(
    *,
    question_len: int,
    answerable: bool,
    top_doc_id: int | None = None,
    top_score: float | None = None,
    latency_ms: int | None = None,
    degraded: bool = False,
    db_path: PathLike = None,
) -> None:
    """写一条脱敏问答指标（不存 user_id、不存问题原文）。"""
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO qa_metrics
                (question_len, answerable, top_doc_id, top_score, latency_ms, degraded)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                question_len,
                1 if answerable else 0,
                top_doc_id,
                top_score,
                latency_ms,
                1 if degraded else 0,
            ),
        )


@retry_on_write_lock
def cleanup_metrics(
    days: int = METRICS_RETENTION_DAYS, db_path: PathLike = None
) -> int:
    """删除超过保留期的问答指标，返回删除条数（M3-08，docs/02 §6.3）。

    时间基准必须在同一侧：`created_at` 由 SQLite 以本地时间写入
    （`datetime('now','localtime')`），这里也用 SQLite 的时间函数算出阈值再比较，
    不能拿 Python 的 `datetime.now()` 去比，否则时区一错就整片误删或整片漏删。
    """
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM qa_metrics WHERE created_at < datetime('now', 'localtime', ?)",
            (f"-{days} days",),
        )
        return cursor.rowcount


# ==================== schedules（日程，二期 2.3）====================
#
# 全部为**个人数据**，只有本人可见（沿用 FR-31 的口径：日程不设公共可见）。
# 任何查询都必须带 `user_id`，杜绝「读到别人日程」的可能。

SCHEDULE_HOMEWORK = "homework"
SCHEDULE_EXAM = "exam"
SCHEDULE_OTHER = "other"
SCHEDULE_TYPES = (SCHEDULE_HOMEWORK, SCHEDULE_EXAM, SCHEDULE_OTHER)

SCHEDULE_PENDING = "pending"
SCHEDULE_DONE = "done"

# 允许通过 update_schedule 修改的字段白名单，防止调用方拼出非法列名
_SCHEDULE_UPDATABLE = {"title", "type", "course", "due_at", "remind_days", "note"}


@dataclass
class Schedule:
    id: int
    user_id: int
    title: str
    type: str
    course: str
    due_at: str
    remind_days: int
    status: str
    note: str
    reminded_at: str | None
    created_at: str
    updated_at: str

    @property
    def is_done(self) -> bool:
        return self.status == SCHEDULE_DONE


def _to_schedule(row) -> Schedule:
    return Schedule(
        id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        type=row["type"],
        course=row["course"] or "",
        due_at=row["due_at"],
        remind_days=row["remind_days"] or 0,
        status=row["status"],
        note=row["note"] or "",
        reminded_at=row["reminded_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# 日程截止时间的存储格式。解析逻辑放在数据访问层，供日程页与 Agent 工具共用，
# 避免两处各写一份、格式改了对不上（二期 2.2 引入工具调用时上移）。
SCHEDULE_DUE_FORMAT = "%Y-%m-%d %H:%M"


def parse_schedule_due(due_at: str) -> datetime:
    """解析日程截止时间（`YYYY-MM-DD HH:MM`，本地时间）。"""
    return datetime.strptime(due_at, SCHEDULE_DUE_FORMAT)


def schedule_days_until(schedule: Schedule, *, now: datetime | None = None) -> int:
    """距截止还有几天（按自然日算，负数表示已逾期）。"""
    today = (now or datetime.now()).date()
    return (parse_schedule_due(schedule.due_at).date() - today).days


@retry_on_write_lock
def create_schedule(
    *,
    user_id: int,
    title: str,
    due_at: str,
    type: str = SCHEDULE_HOMEWORK,
    course: str | None = None,
    remind_days: int = 1,
    note: str | None = None,
    db_path: PathLike = None,
) -> int:
    """新建日程，返回 schedule_id。空课程 / 备注存 NULL，展示层再兜底。"""
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO schedules (user_id, title, type, course, due_at, remind_days, note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                title.strip(),
                type,
                (course or "").strip() or None,
                due_at,
                remind_days,
                (note or "").strip() or None,
            ),
        )
        return int(cursor.lastrowid)


def get_schedule(schedule_id: int, db_path: PathLike = None) -> Schedule | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM schedules WHERE id = ?", (schedule_id,)
        ).fetchone()
    return _to_schedule(row) if row else None


def list_schedules(
    user_id: int, *, status: str | None = None, db_path: PathLike = None
) -> list[Schedule]:
    """列出某用户的日程，按截止时间升序。

    `user_id` 必传：这是个人数据，漏传会变成「谁都能拿到全部日程」，
    因此这里不提供「不传 user_id 即返回全部」的退化行为。
    """
    sql = "SELECT * FROM schedules WHERE user_id = ?"
    params: list[object] = [user_id]
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY due_at ASC, id ASC"
    with get_conn(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_to_schedule(row) for row in rows]


@retry_on_write_lock
def update_schedule(schedule_id: int, *, db_path: PathLike = None, **fields) -> bool:
    """更新日程字段（白名单校验）。空课程 / 备注归一为 NULL。"""
    unknown = set(fields) - _SCHEDULE_UPDATABLE
    if unknown:
        raise ValueError(f"schedule 不支持的字段：{sorted(unknown)}")
    if not fields:
        return False

    cleaned: dict[str, object] = {}
    for name, value in fields.items():
        if isinstance(value, str):
            value = value.strip()
        if name in ("course", "note"):
            value = value or None
        cleaned[name] = value

    assignments = ", ".join(f"{name} = ?" for name in cleaned)
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            f"""
            UPDATE schedules
               SET {assignments}, updated_at = datetime('now','localtime')
             WHERE id = ?
            """,
            (*cleaned.values(), schedule_id),
        )
        return cursor.rowcount > 0


@retry_on_write_lock
def set_schedule_status(
    schedule_id: int, status: str, db_path: PathLike = None
) -> bool:
    """切换日程状态（pending / done）。"""
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE schedules
               SET status = ?, updated_at = datetime('now','localtime')
             WHERE id = ?
            """,
            (status, schedule_id),
        )
        return cursor.rowcount > 0


@retry_on_write_lock
def mark_schedules_reminded(
    schedule_ids: Sequence[int], db_path: PathLike = None
) -> int:
    """把若干日程标记为「已提醒」（写入 reminded_at），返回更新条数。"""
    ids = [int(item) for item in schedule_ids]
    if not ids:
        return 0
    placeholders = ", ".join("?" for _ in ids)
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            f"""
            UPDATE schedules
               SET reminded_at = datetime('now','localtime')
             WHERE id IN ({placeholders})
            """,
            ids,
        )
        return int(cursor.rowcount)


@retry_on_write_lock
def delete_schedule(schedule_id: int, db_path: PathLike = None) -> bool:
    with get_conn(db_path) as conn:
        cursor = conn.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
        return cursor.rowcount > 0


# ==================== summaries（课件总结 / 复习提纲，二期 2.4）====================
#
# 产物是**个人数据**：即使是公共文档，每个人生成的总结也只属于其本人。
# 同一用户对同一文档的同一产物只保留最新一份（UNIQUE 约束 + upsert 覆盖）。

SUMMARY_KIND_SUMMARY = "summary"
SUMMARY_KIND_OUTLINE = "outline"
SUMMARY_KINDS = (SUMMARY_KIND_SUMMARY, SUMMARY_KIND_OUTLINE)


@dataclass
class Summary:
    id: int
    user_id: int
    doc_id: int
    kind: str
    content: str
    source_chunks: int
    created_at: str
    updated_at: str


def _to_summary(row) -> Summary:
    return Summary(
        id=row["id"],
        user_id=row["user_id"],
        doc_id=row["doc_id"],
        kind=row["kind"],
        content=row["content"] or "",
        source_chunks=row["source_chunks"] or 0,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@retry_on_write_lock
def upsert_summary(
    *,
    user_id: int,
    doc_id: int,
    kind: str,
    content: str,
    source_chunks: int = 0,
    db_path: PathLike = None,
) -> int:
    """写入或覆盖一份产物，返回 summary_id（重新生成即覆盖旧的那一份）。"""
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO summaries (user_id, doc_id, kind, content, source_chunks)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (user_id, doc_id, kind) DO UPDATE SET
                content = excluded.content,
                source_chunks = excluded.source_chunks,
                updated_at = datetime('now','localtime')
            """,
            (user_id, doc_id, kind, content, source_chunks),
        )
        row = conn.execute(
            "SELECT id FROM summaries WHERE user_id = ? AND doc_id = ? AND kind = ?",
            (user_id, doc_id, kind),
        ).fetchone()
        return int(row["id"])


def get_summary(
    user_id: int, doc_id: int, kind: str, db_path: PathLike = None
) -> Summary | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM summaries WHERE user_id = ? AND doc_id = ? AND kind = ?",
            (user_id, doc_id, kind),
        ).fetchone()
    return _to_summary(row) if row else None


def list_summaries_by_doc(
    user_id: int, doc_id: int, db_path: PathLike = None
) -> list[Summary]:
    """列出某用户对某文档已生成的全部产物（总结 / 提纲）。`user_id` 必传。"""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM summaries WHERE user_id = ? AND doc_id = ? ORDER BY kind",
            (user_id, doc_id),
        ).fetchall()
    return [_to_summary(row) for row in rows]


# ==================== 备份 / 恢复（FR-24，二期）====================
#
# 备份涉及的表与恢复顺序由外键依赖决定：users → documents → conversations
# → messages → message_sources / feedback / schedules / summaries。
# qa_metrics 是脱敏日志（90 天自动清理，可丢）、ingest_tasks 是瞬时状态，
# 都不进备份。

BACKUP_TABLES = (
    "users",
    "documents",
    "conversations",
    "messages",
    "message_sources",
    "feedback",
    "schedules",
    "summaries",
)


def export_tables(db_path: PathLike = None) -> dict[str, list[dict]]:
    """导出备份涉及的全部表，按 id 升序，返回 `{表名: [行字典]}`。"""
    with get_conn(db_path) as conn:
        return {
            name: [dict(row) for row in conn.execute(f"SELECT * FROM {name} ORDER BY id")]
            for name in BACKUP_TABLES
        }


@retry_on_write_lock
def import_tables(
    tables: dict[str, list[dict]], *, db_path: PathLike = None
) -> dict[str, int]:
    """把备份数据整库写回，返回各表写入条数。

    语义是**整库替换**：先按反向外键顺序清空这些表，再按正向顺序插入。
    不清空会出现主键冲突，或新旧数据混在一起（恢复后看到的是两份数据的并集）。

    列名来自备份文件，属外部输入：逐一比对表结构，出现未知列直接抛错，
    避免被拼进 SQL。
    """
    counts: dict[str, int] = {}
    with get_conn(db_path) as conn:
        for name in reversed(BACKUP_TABLES):
            conn.execute(f"DELETE FROM {name}")

        for name in BACKUP_TABLES:
            rows = tables.get(name) or []
            _insert_rows(conn, name, rows)
            counts[name] = len(rows)
    return counts


def _insert_rows(conn, table: str, rows: list[dict]) -> None:
    if not rows:
        return

    allowed = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    columns = list(rows[0].keys())
    unknown = set(columns) - allowed
    if unknown:
        raise ValueError(f"备份数据包含 {table} 表不存在的列：{sorted(unknown)}")

    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        [tuple(row.get(column) for column in columns) for row in rows],
    )


# ==================== 管理员五维统计（FR-17，二期）====================
#
# 全部为只读聚合，不加重试装饰器（WAL 下读不会被写锁阻塞，见 db.retry_on_write_lock）。
# 时间口径统一用 SQLite 的本地时间函数，与各表 created_at 的写入方式一致。


@dataclass
class UserStats:
    total: int
    active_today: int
    disabled: int


@dataclass
class DocumentStats:
    total: int
    public: int
    personal: int
    failed_tasks: int

    @property
    def public_ratio(self) -> float:
        return self.public / self.total if self.total else 0.0


@dataclass
class QualityStats:
    total: int
    refused: int
    degraded: int
    avg_latency_ms: float

    @property
    def refusal_rate(self) -> float:
        return self.refused / self.total if self.total else 0.0


@dataclass
class WorstAnswer:
    message_id: int
    content: str
    useless: int
    rated: int


def stats_users(db_path: PathLike = None) -> UserStats:
    """用户维度：总数、今日活跃人数、被禁用人数。

    今日活跃按「今天产生过消息的用户数」统计（`qa_metrics` 脱敏无 user_id，
    只能从会话链路取）。
    """
    with get_conn(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        disabled = conn.execute(
            "SELECT COUNT(*) AS n FROM users WHERE status = ?", (USER_STATUS_DISABLED,)
        ).fetchone()["n"]
        active_today = conn.execute(
            """
            SELECT COUNT(DISTINCT c.user_id) AS n
              FROM messages m
              JOIN conversations c ON c.id = m.conversation_id
             WHERE date(m.created_at) = date('now','localtime')
            """
        ).fetchone()["n"]
    return UserStats(total=int(total), active_today=int(active_today), disabled=int(disabled))


def stats_documents(db_path: PathLike = None) -> DocumentStats:
    """文档维度：总数（不含已删除）、公共 / 个人数量、入库失败任务数。"""
    with get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(is_public = 1) AS public,
                   SUM(is_public = 0) AS personal
              FROM documents
             WHERE status != ?
            """,
            (DOC_DELETED,),
        ).fetchone()
        failed = conn.execute(
            "SELECT COUNT(*) AS n FROM ingest_tasks WHERE status = ?", (TASK_FAILED,)
        ).fetchone()["n"]
    return DocumentStats(
        total=int(row["total"] or 0),
        public=int(row["public"] or 0),
        personal=int(row["personal"] or 0),
        failed_tasks=int(failed),
    )


def stats_quality(db_path: PathLike = None) -> QualityStats:
    """质量维度：拒答率、降级次数、平均耗时（来自脱敏的 `qa_metrics`）。"""
    with get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(answerable = 0) AS refused,
                   SUM(degraded = 1) AS degraded,
                   AVG(latency_ms) AS avg_ms
              FROM qa_metrics
            """
        ).fetchone()
    return QualityStats(
        total=int(row["total"] or 0),
        refused=int(row["refused"] or 0),
        degraded=int(row["degraded"] or 0),
        avg_latency_ms=float(row["avg_ms"] or 0.0),
    )


def stats_qa_volume(days: int = 30, db_path: PathLike = None) -> list[tuple[str, int]]:
    """问答量维度：近 `days` 天按天聚合的问答次数，无数据的日期补 0。

    返回按日期升序的 `[(YYYY-MM-DD, 次数)]`，直接喂给图表，避免前端再补洞。
    """
    days = max(int(days), 1)
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT date(created_at) AS day, COUNT(*) AS n
              FROM qa_metrics
             WHERE date(created_at) >= date('now', 'localtime', ?)
             GROUP BY day
            """,
            (f"-{days - 1} days",),
        ).fetchall()
    counts = {row["day"]: int(row["n"]) for row in rows}

    today = datetime.now().date()
    return [
        (day, counts.get(day, 0))
        for day in (
            (today - timedelta(days=offset)).isoformat() for offset in range(days - 1, -1, -1)
        )
    ]


def stats_worst_answers(limit: int = 5, db_path: PathLike = None) -> list[WorstAnswer]:
    """差评榜：被点「没用」最多的回答 Top-N（`feedback` 关联 `messages`）。"""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT m.id AS message_id, m.content AS content,
                   SUM(f.rating = ?) AS useless,
                   COUNT(f.id) AS rated
              FROM feedback f
              JOIN messages m ON m.id = f.message_id
             GROUP BY m.id
            HAVING useless > 0
             ORDER BY useless DESC, m.id DESC
             LIMIT ?
            """,
            (RATING_USELESS, limit),
        ).fetchall()
    return [
        WorstAnswer(
            message_id=int(row["message_id"]),
            content=row["content"] or "",
            useless=int(row["useless"] or 0),
            rated=int(row["rated"] or 0),
        )
        for row in rows
    ]
