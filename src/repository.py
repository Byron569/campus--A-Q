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
    → ingest_tasks → documents → users。

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
