"""SQLite 连接与建表。

设计依据：docs/02-架构设计.md §5.1

并发说明（NFR-03，20 人并发）：
- 开启 WAL，读写不互相阻塞
- 设置 busy_timeout，写锁冲突时短暂等待而非立即报错
- 每次操作使用独立连接（不跨线程共享），避免 sqlite3 的线程限制
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config.settings import get_settings

logger = logging.getLogger(__name__)

BUSY_TIMEOUT_MS = 5000

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name  TEXT,
    role          TEXT NOT NULL DEFAULT 'user',
    status        TEXT NOT NULL DEFAULT 'active',
    created_at    TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER,
    is_public   INTEGER NOT NULL DEFAULT 0,
    filename    TEXT NOT NULL,
    filetype    TEXT,
    category    TEXT NOT NULL DEFAULT 'uncategorized',
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS ingest_tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       INTEGER NOT NULL,
    user_id      INTEGER,
    status       TEXT NOT NULL DEFAULT 'pending',
    total_chunks INTEGER NOT NULL DEFAULT 0,
    done_chunks  INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    FOREIGN KEY (doc_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    title      TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS message_sources (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    doc_id     INTEGER,
    filename   TEXT,
    snippet    TEXT,
    score      REAL,
    matched_by TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    rating     TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    UNIQUE (message_id, user_id)
);

CREATE TABLE IF NOT EXISTS qa_metrics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    question_len INTEGER,
    answerable   INTEGER,
    top_doc_id   INTEGER,
    top_score    REAL,
    latency_ms   INTEGER,
    degraded     INTEGER,
    created_at   TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_documents_user   ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_public ON documents(is_public, status);
CREATE INDEX IF NOT EXISTS idx_tasks_doc        ON ingest_tasks(doc_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status     ON ingest_tasks(status);
CREATE INDEX IF NOT EXISTS idx_conv_user        ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_msg_conv         ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_src_msg          ON message_sources(message_id);
CREATE INDEX IF NOT EXISTS idx_feedback_msg     ON feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_metrics_created  ON qa_metrics(created_at);
"""


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """建立一个 SQLite 连接。"""
    path = Path(db_path) if db_path else get_settings().database_path
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


@contextmanager
def get_conn(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """连接上下文：正常提交，异常回滚，始终关闭。"""
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> None:
    """建表与建索引。幂等，可重复调用。"""
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.executescript(INDEXES)
    logger.debug("数据库初始化完成：%s", db_path or get_settings().database_path)
