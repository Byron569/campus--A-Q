"""SQLite 连接层测试：旧库迁移（CR-03）与写锁退避重试（docs/02 §11）。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.store import db as db_module
from src.store.db import WRITE_RETRY_TIMES, get_conn, init_db, retry_on_write_lock


# ==================== CR-03：旧库补列 ====================

# 加 warning 列之前的 ingest_tasks 结构（客户旧库就是这个样子）
_LEGACY_TASKS = """
CREATE TABLE ingest_tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       INTEGER NOT NULL,
    user_id      INTEGER,
    status       TEXT NOT NULL DEFAULT 'pending',
    total_chunks INTEGER NOT NULL DEFAULT 0,
    done_chunks  INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
"""


def make_legacy_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_LEGACY_TASKS)
        conn.commit()
    finally:
        conn.close()


def test_init_db_adds_missing_warning_column(tmp_path: Path) -> None:
    """老库升级必须自动补列，否则写 warning 会直接报 no such column。"""
    path = tmp_path / "legacy.db"
    make_legacy_db(path)

    init_db(path)

    with get_conn(path) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(ingest_tasks)")}
    assert "warning" in columns


def test_init_db_migration_keeps_existing_rows(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    make_legacy_db(path)
    conn = sqlite3.connect(str(path))
    conn.execute("INSERT INTO ingest_tasks (doc_id, status) VALUES (1, 'done')")
    conn.commit()
    conn.close()

    init_db(path)

    with get_conn(path) as conn:
        row = conn.execute("SELECT doc_id, status, warning FROM ingest_tasks").fetchone()
    assert (row["doc_id"], row["status"], row["warning"]) == (1, "done", None)


def test_init_db_is_idempotent_on_migrated_schema(tmp_path: Path) -> None:
    """重复启动不能因为列已存在而报错。"""
    path = tmp_path / "legacy.db"
    make_legacy_db(path)

    init_db(path)
    init_db(path)

    with get_conn(path) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(ingest_tasks)")}
    assert "warning" in columns


# ==================== docs/02 §11：写锁重试 ====================


@pytest.fixture()
def no_sleep(monkeypatch):
    """把退避等待换掉，否则测试要真睡 0.6 秒。"""
    monkeypatch.setattr(db_module.time, "sleep", lambda _seconds: None)


def test_retry_succeeds_after_transient_lock(no_sleep) -> None:
    calls: list[int] = []

    @retry_on_write_lock
    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise sqlite3.OperationalError("database is locked")
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 3


def test_retry_gives_up_after_configured_attempts(no_sleep) -> None:
    calls: list[int] = []

    @retry_on_write_lock
    def always_locked():
        calls.append(1)
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError):
        always_locked()

    assert WRITE_RETRY_TIMES == 3
    assert len(calls) == 3


def test_retry_also_covers_busy(no_sleep) -> None:
    calls: list[int] = []

    @retry_on_write_lock
    def busy_once() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is busy")
        return "ok"

    assert busy_once() == "ok"
    assert len(calls) == 2


def test_retry_does_not_mask_other_sqlite_errors(no_sleep) -> None:
    """只重试锁冲突；「表不存在」这类错误重试三次毫无意义，必须立即抛。"""
    calls: list[int] = []

    @retry_on_write_lock
    def broken():
        calls.append(1)
        raise sqlite3.OperationalError("no such table: nope")

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        broken()

    assert len(calls) == 1


def test_retry_does_not_mask_business_errors(no_sleep) -> None:
    calls: list[int] = []

    @retry_on_write_lock
    def invalid():
        calls.append(1)
        raise ValueError("字段不在白名单")

    with pytest.raises(ValueError):
        invalid()

    assert len(calls) == 1


def test_retry_preserves_function_metadata() -> None:
    @retry_on_write_lock
    def documented() -> None:
        """说明文字"""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "说明文字"
