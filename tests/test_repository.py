"""数据访问层单元测试。

覆盖 docs/03 §6.1 的 TC-U11 / TC-U12 / TC-U18，以及 DR-06（软删除过滤）、
DR-05（重试复用原任务）、DR-07（幂等校验）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.repository import (
    DOC_DELETED,
    RATING_USEFUL,
    RATING_USELESS,
    ROLE_ASSISTANT,
    ROLE_USER,
    TASK_DONE,
    TASK_FAILED,
    TASK_PENDING,
    TASK_RUNNING,
    USER_STATUS_ACTIVE,
    USER_STATUS_DISABLED,
    add_feedback,
    add_message,
    add_metric,
    add_sources,
    cleanup_metrics,
    create_conversation,
    create_document,
    create_task,
    create_user,
    delete_conversation,
    delete_user,
    exists_active_task,
    get_conversation,
    get_document,
    get_feedback,
    get_task,
    get_task_by_doc,
    get_user,
    list_conversations,
    list_documents,
    list_messages,
    list_sources_by_message,
    list_tasks_by_user,
    list_users,
    recover_stale_tasks,
    rename_conversation,
    reset_task,
    set_user_status,
    soft_delete_document,
    update_chunk_count,
    update_task,
)
from src.store.chroma import SearchHit
from src.store.db import get_conn, init_db


def make_doc(db: Path, filename: str = "学生手册.pdf", **kwargs) -> int:
    params = {"filename": filename, "filetype": "pdf", "category": "freshman"}
    params.update(kwargs)
    return create_document(db_path=db, **params)


# ==================== 建表 ====================


def test_init_db_creates_all_eight_tables(db: Path) -> None:
    with get_conn(db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    names = {row["name"] for row in rows}
    assert names == {
        "users",
        "documents",
        "ingest_tasks",
        "conversations",
        "messages",
        "message_sources",
        "feedback",
        "qa_metrics",
    }


def test_init_db_is_idempotent(db: Path) -> None:
    init_db(db)
    init_db(db)


def test_wal_mode_enabled(db: Path) -> None:
    """NFR-03：20 人并发下靠 WAL 缓解写锁冲突。"""
    with get_conn(db) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


# ==================== documents ====================


def test_create_and_get_document(db: Path) -> None:
    doc_id = make_doc(db, filename="新生报到须知.docx", filetype="docx", size_bytes=1024)
    document = get_document(doc_id, db_path=db)

    assert document is not None
    assert document.filename == "新生报到须知.docx"
    assert document.filetype == "docx"
    assert document.size_bytes == 1024
    assert document.is_public is False
    assert document.status == "active"
    assert document.chunk_count == 0


def test_get_missing_document_returns_none(db: Path) -> None:
    assert get_document(999, db_path=db) is None


def test_update_chunk_count(db: Path) -> None:
    doc_id = make_doc(db)
    update_chunk_count(doc_id, 128, db_path=db)
    assert get_document(doc_id, db_path=db).chunk_count == 128


# ---- TC-U12：分页 ----


def test_list_documents_pagination(db: Path) -> None:
    for index in range(25):
        make_doc(db, filename=f"doc-{index:02d}.pdf", user_id=1)

    first, total = list_documents(user_id=1, page=1, page_size=10, db_path=db)
    second, _ = list_documents(user_id=1, page=2, page_size=10, db_path=db)
    third, _ = list_documents(user_id=1, page=3, page_size=10, db_path=db)

    assert total == 25
    assert len(first) == 10
    assert len(second) == 10
    assert len(third) == 5
    # 分页不重叠
    ids = {d.id for d in first} | {d.id for d in second} | {d.id for d in third}
    assert len(ids) == 25


def test_list_documents_category_filter(db: Path) -> None:
    make_doc(db, filename="a.pdf", category="freshman", user_id=1)
    make_doc(db, filename="b.pdf", category="admin", user_id=1)

    rows, total = list_documents(user_id=1, category="admin", db_path=db)

    assert total == 1
    assert rows[0].category == "admin"


def test_list_documents_without_identity_and_public_returns_empty(db: Path) -> None:
    """既没有用户身份、也不要求公共文档时，必须返回空集，绝不返回全量。"""
    make_doc(db, filename="a.pdf", user_id=1)

    rows, total = list_documents(db_path=db)

    assert rows == []
    assert total == 0


def test_list_documents_public_only_for_admin_view(db: Path) -> None:
    make_doc(db, filename="public.pdf", is_public=True)
    make_doc(db, filename="private.pdf", user_id=1)

    rows, total = list_documents(include_public=True, db_path=db)

    assert total == 1
    assert rows[0].filename == "public.pdf"


def test_list_documents_union_own_and_public(db: Path) -> None:
    make_doc(db, filename="public.pdf", is_public=True)
    make_doc(db, filename="mine.pdf", user_id=1)
    make_doc(db, filename="other.pdf", user_id=2)

    rows, total = list_documents(user_id=1, include_public=True, db_path=db)

    filenames = {row.filename for row in rows}
    assert total == 2
    assert filenames == {"public.pdf", "mine.pdf"}


# ---- DR-06：软删除必须被过滤 ----


def test_soft_deleted_document_disappears_from_list(db: Path) -> None:
    doc_id = make_doc(db, user_id=1)

    assert soft_delete_document(doc_id, db_path=db) is True

    rows, total = list_documents(user_id=1, db_path=db)
    assert total == 0
    assert rows == []


def test_soft_delete_keeps_row_and_sets_status(db: Path) -> None:
    doc_id = make_doc(db)
    soft_delete_document(doc_id, db_path=db)

    document = get_document(doc_id, db_path=db)
    assert document is not None
    assert document.status == DOC_DELETED


def test_soft_delete_twice_is_noop(db: Path) -> None:
    doc_id = make_doc(db)
    assert soft_delete_document(doc_id, db_path=db) is True
    assert soft_delete_document(doc_id, db_path=db) is False


# ==================== ingest_tasks ====================


def test_create_and_update_task(db: Path) -> None:
    doc_id = make_doc(db)
    task_id = create_task(doc_id, user_id=1, db_path=db)

    update_task(task_id, db_path=db, status=TASK_RUNNING, total_chunks=100, done_chunks=32)
    task = get_task(task_id, db_path=db)

    assert task.status == TASK_RUNNING
    assert task.total_chunks == 100
    assert task.done_chunks == 32
    assert task.progress == pytest.approx(0.32)


def test_task_progress_defaults(db: Path) -> None:
    doc_id = make_doc(db)
    pending = get_task(create_task(doc_id, db_path=db), db_path=db)
    assert pending.progress == 0.0

    done_id = create_task(doc_id, db_path=db)
    update_task(done_id, db_path=db, status=TASK_DONE)
    assert get_task(done_id, db_path=db).progress == 1.0


def test_update_task_rejects_unknown_field(db: Path) -> None:
    task_id = create_task(make_doc(db), db_path=db)

    with pytest.raises(ValueError, match="不允许更新"):
        update_task(task_id, db_path=db, doc_id=999)
    with pytest.raises(ValueError, match="不允许更新"):
        update_task(task_id, db_path=db, id=1)


# ---- DR-07：幂等校验 ----


def test_exists_active_task_detects_pending_and_running(db: Path) -> None:
    doc_id = make_doc(db)
    assert exists_active_task(doc_id, db_path=db) is False

    task_id = create_task(doc_id, db_path=db)
    assert exists_active_task(doc_id, db_path=db) is True

    update_task(task_id, db_path=db, status=TASK_RUNNING)
    assert exists_active_task(doc_id, db_path=db) is True

    update_task(task_id, db_path=db, status=TASK_DONE)
    assert exists_active_task(doc_id, db_path=db) is False


def test_exists_active_task_ignores_failed(db: Path) -> None:
    doc_id = make_doc(db)
    task_id = create_task(doc_id, db_path=db)
    update_task(task_id, db_path=db, status=TASK_FAILED, error="解析失败")

    assert exists_active_task(doc_id, db_path=db) is False


# ---- DR-05：重试复用原任务 ----


def test_reset_task_reuses_same_record(db: Path) -> None:
    doc_id = make_doc(db)
    task_id = create_task(doc_id, db_path=db)
    update_task(task_id, db_path=db, status=TASK_FAILED, total_chunks=50,
                done_chunks=20, error="未能从文件中提取到文本")

    reset_task(task_id, db_path=db)
    task = get_task(task_id, db_path=db)

    assert task.id == task_id
    assert task.status == TASK_PENDING
    assert task.done_chunks == 0
    assert task.total_chunks == 0
    assert task.error is None
    # 一个文档始终只有一条任务记录
    assert get_task_by_doc(doc_id, db_path=db).id == task_id


# ---- CR-03：告警字段落库 ----

WARNING_TEXT = "该文件疑似扫描件，建议后续启用 OCR"


def test_update_task_persists_warning(db: Path) -> None:
    """告警要落库，文档列表才显示得出「疑似扫描件」（CR-03）。"""
    doc_id = make_doc(db)
    task_id = create_task(doc_id, db_path=db)

    update_task(task_id, db_path=db, status=TASK_DONE, warning=WARNING_TEXT)

    task = get_task(task_id, db_path=db)
    assert task.status == TASK_DONE
    assert task.warning == WARNING_TEXT


def test_reset_task_clears_warning(db: Path) -> None:
    """重试后上一次的告警不能残留。"""
    doc_id = make_doc(db)
    task_id = create_task(doc_id, db_path=db)
    update_task(task_id, db_path=db, status=TASK_DONE, warning=WARNING_TEXT)

    reset_task(task_id, db_path=db)

    task = get_task(task_id, db_path=db)
    assert task.warning is None
    assert task.error is None


def test_task_without_warning_is_none(db: Path) -> None:
    doc_id = make_doc(db)
    assert get_task(create_task(doc_id, db_path=db), db_path=db).warning is None


# ==================== 用户管理（M3-04 / FR-16）====================


def test_list_users_pagination(db: Path) -> None:
    for index in range(5):
        create_user(username=f"stu-{index}", password_hash="x", db_path=db)

    rows, total = list_users(page=1, page_size=3, db_path=db)

    assert total == 7  # 夹具预置 2 个 + 这里 5 个
    assert len(rows) == 3


def test_list_users_keyword_matches_username_and_display_name(db: Path) -> None:
    create_user(username="alice", password_hash="x", display_name="小艾", db_path=db)
    create_user(username="bob", password_hash="x", display_name="小波", db_path=db)

    by_name, _ = list_users(keyword="ali", db_path=db)
    by_display, _ = list_users(keyword="小波", db_path=db)

    assert [u.username for u in by_name] == ["alice"]
    assert [u.username for u in by_display] == ["bob"]


def test_set_user_status_toggles(db: Path) -> None:
    assert set_user_status(1, USER_STATUS_DISABLED, db_path=db) is True
    assert get_user(1, db_path=db).is_active is False

    assert set_user_status(1, USER_STATUS_ACTIVE, db_path=db) is True
    assert get_user(1, db_path=db).is_active is True


def test_set_user_status_rejects_unknown_value(db: Path) -> None:
    with pytest.raises(ValueError, match="未知的账号状态"):
        set_user_status(1, "banana", db_path=db)


def test_delete_user_removes_every_related_row(db: Path) -> None:
    """FR-25：账号删除必须连带清掉其文档、任务、会话、消息、引用与反馈。"""
    doc_id = create_document(
        filename="我的资料.txt", filetype="txt", category="uncategorized",
        user_id=1, db_path=db,
    )
    create_task(doc_id, user_id=1, db_path=db)
    conversation_id = create_conversation(1, title="会话", db_path=db)
    message_id = add_message(conversation_id, ROLE_ASSISTANT, "答案", db_path=db)
    add_sources(
        message_id,
        [SearchHit(text="片段", score=0.5, filename="a.txt", doc_id=doc_id,
                   chunk_index=0, category="", matched_by="vector")],
        db_path=db,
    )
    add_feedback(message_id, 1, RATING_USEFUL, db_path=db)

    delete_user(1, db_path=db)

    assert get_user(1, db_path=db) is None
    assert list_documents(user_id=1, db_path=db) == ([], 0)
    assert list_tasks_by_user(1, db_path=db) == ([], 0)
    assert list_conversations(1, db_path=db) == ([], 0)
    assert list_messages(conversation_id, db_path=db) == []
    assert list_sources_by_message(message_id, db_path=db) == []
    with get_conn(db) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"] == 0


def test_delete_user_keeps_other_users_data(db: Path) -> None:
    create_document(
        filename="a.txt", filetype="txt", category="", user_id=1, db_path=db
    )
    create_document(
        filename="b.txt", filetype="txt", category="", user_id=2, db_path=db
    )
    other_conversation = create_conversation(2, title="别人的", db_path=db)
    add_message(other_conversation, ROLE_USER, "别人的消息", db_path=db)

    delete_user(1, db_path=db)

    assert get_user(2, db_path=db) is not None
    assert list_documents(user_id=2, db_path=db)[1] == 1
    assert [m.content for m in list_messages(other_conversation, db_path=db)] == ["别人的消息"]


def test_list_tasks_by_user_pagination(db: Path) -> None:
    for index in range(5):
        create_task(make_doc(db, filename=f"{index}.pdf"), user_id=7, db_path=db)
    create_task(make_doc(db, filename="other.pdf"), user_id=8, db_path=db)

    rows, total = list_tasks_by_user(7, page=1, page_size=3, db_path=db)

    assert total == 5
    assert len(rows) == 3


# ---- TC-U18：启动时回收中断任务 ----


def test_recover_stale_tasks_marks_running_as_failed(db: Path) -> None:
    doc_a, doc_b = make_doc(db, filename="a.pdf"), make_doc(db, filename="b.pdf")
    running = create_task(doc_a, db_path=db)
    done = create_task(doc_b, db_path=db)
    update_task(running, db_path=db, status=TASK_RUNNING, total_chunks=10, done_chunks=4)
    update_task(done, db_path=db, status=TASK_DONE)

    recovered = recover_stale_tasks(db_path=db)

    assert recovered == 1
    assert get_task(running, db_path=db).status == TASK_FAILED
    assert "重试" in get_task(running, db_path=db).error
    # 已完成的任务不受影响
    assert get_task(done, db_path=db).status == TASK_DONE


def test_recover_stale_tasks_noop_when_nothing_running(db: Path) -> None:
    create_task(make_doc(db), db_path=db)
    assert recover_stale_tasks(db_path=db) == 0


# ==================== conversations / messages ====================


def test_create_and_get_conversation(db: Path) -> None:
    conversation_id = create_conversation(1, title="宿舍怎么搬", db_path=db)
    conversation = get_conversation(conversation_id, db_path=db)

    assert conversation is not None
    assert conversation.user_id == 1
    assert conversation.title == "宿舍怎么搬"


def test_create_conversation_blank_title_stored_as_empty(db: Path) -> None:
    assert get_conversation(create_conversation(1, title="   ", db_path=db), db_path=db).title == ""


def test_get_missing_conversation_returns_none(db: Path) -> None:
    assert get_conversation(999, db_path=db) is None


def test_messages_are_returned_in_time_order(db: Path) -> None:
    conversation_id = create_conversation(1, db_path=db)
    add_message(conversation_id, ROLE_USER, "第一问", db_path=db)
    add_message(conversation_id, ROLE_ASSISTANT, "第一答", db_path=db)
    add_message(conversation_id, ROLE_USER, "第二问", db_path=db)

    messages = list_messages(conversation_id, db_path=db)

    assert [m.content for m in messages] == ["第一问", "第一答", "第二问"]
    assert [m.role for m in messages] == [ROLE_USER, ROLE_ASSISTANT, ROLE_USER]


def test_list_messages_limit_keeps_latest_in_forward_order(db: Path) -> None:
    """取最近 N 条时要**正序**返回，否则会把历史倒着喂给模型。"""
    conversation_id = create_conversation(1, db_path=db)
    for index in range(5):
        add_message(conversation_id, ROLE_USER, f"第{index}问", db_path=db)

    messages = list_messages(conversation_id, limit=2, db_path=db)

    assert [m.content for m in messages] == ["第3问", "第4问"]


def test_add_message_touches_conversation_updated_at(db: Path) -> None:
    conversation_id = create_conversation(1, db_path=db)
    with get_conn(db) as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = '2000-01-01 00:00:00' WHERE id = ?",
            (conversation_id,),
        )

    add_message(conversation_id, ROLE_USER, "新消息", db_path=db)

    assert get_conversation(conversation_id, db_path=db).updated_at > "2000-01-01 00:00:00"


# ==================== 会话管理（M2-06 / FR-12）====================


def test_list_conversations_most_recent_first(db: Path) -> None:
    first = create_conversation(1, title="第一个", db_path=db)
    second = create_conversation(1, title="第二个", db_path=db)
    with get_conn(db) as conn:
        conn.execute("UPDATE conversations SET updated_at = '2026-01-01 00:00:00' WHERE id = ?", (first,))
        conn.execute("UPDATE conversations SET updated_at = '2026-02-01 00:00:00' WHERE id = ?", (second,))

    rows, total = list_conversations(1, db_path=db)

    assert total == 2
    assert [row.id for row in rows] == [second, first]


def test_list_conversations_is_per_user(db: Path) -> None:
    create_conversation(1, title="我的", db_path=db)
    create_conversation(2, title="别人的", db_path=db)

    rows, total = list_conversations(1, db_path=db)

    assert total == 1
    assert rows[0].title == "我的"


def test_rename_conversation(db: Path) -> None:
    conversation_id = create_conversation(1, title="旧名", db_path=db)

    assert rename_conversation(conversation_id, "新名", db_path=db) is True
    assert get_conversation(conversation_id, db_path=db).title == "新名"
    assert rename_conversation(999, "无处可改", db_path=db) is False


def test_delete_conversation_removes_messages_sources_and_feedback(db: Path) -> None:
    """删会话不能留下半删状态：消息、引用、反馈必须一起清掉。"""
    conversation_id = create_conversation(1, db_path=db)
    user_message = add_message(conversation_id, ROLE_USER, "问题", db_path=db)
    assistant = add_message(conversation_id, ROLE_ASSISTANT, "答案【来源1】", db_path=db)
    add_sources(
        assistant,
        [SearchHit(text="片段", score=0.8, filename="a.pdf", doc_id=1,
                   chunk_index=0, category="freshman", matched_by="vector")],
        db_path=db,
    )
    add_feedback(assistant, 1, RATING_USEFUL, db_path=db)

    removed = delete_conversation(conversation_id, db_path=db)

    assert removed == 2
    assert get_conversation(conversation_id, db_path=db) is None
    assert list_messages(conversation_id, db_path=db) == []
    assert list_sources_by_message(assistant, db_path=db) == []
    assert get_feedback(assistant, 1, db_path=db) is None
    with get_conn(db) as conn:
        left = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
    assert left == 0
    assert user_message  # 用户消息同样被删掉（仅用于表明它确实建过）


def test_delete_conversation_keeps_others(db: Path) -> None:
    keep = create_conversation(1, title="保留", db_path=db)
    drop = create_conversation(1, title="删除", db_path=db)
    add_message(keep, ROLE_USER, "保留的消息", db_path=db)
    add_message(drop, ROLE_USER, "要删的消息", db_path=db)

    delete_conversation(drop, db_path=db)

    rows, total = list_conversations(1, db_path=db)
    assert total == 1
    assert rows[0].id == keep
    assert [m.content for m in list_messages(keep, db_path=db)] == ["保留的消息"]


# ==================== message_sources / qa_metrics ====================


def test_add_and_list_sources(db: Path) -> None:
    conversation_id = create_conversation(1, db_path=db)
    message_id = add_message(conversation_id, ROLE_ASSISTANT, "答案【来源1】", db_path=db)
    hits = [
        SearchHit(text="片段一", score=0.8, filename="a.pdf", doc_id=1,
                  chunk_index=0, category="freshman", matched_by="both"),
        SearchHit(text="片段二", score=None, filename="b.pdf", doc_id=2,
                  chunk_index=3, category="admin", matched_by="bm25"),
    ]

    add_sources(message_id, hits, db_path=db)
    saved = list_sources_by_message(message_id, db_path=db)

    assert [row["filename"] for row in saved] == ["a.pdf", "b.pdf"]
    assert [row["matched_by"] for row in saved] == ["both", "bm25"]
    assert saved[1]["score"] is None  # 仅向量路可得相似度（DR-15）


def test_add_sources_with_empty_list_is_noop(db: Path) -> None:
    conversation_id = create_conversation(1, db_path=db)
    message_id = add_message(conversation_id, ROLE_ASSISTANT, "拒答", db_path=db)

    add_sources(message_id, [], db_path=db)

    assert list_sources_by_message(message_id, db_path=db) == []


def test_add_metric_stores_no_user_identity(db: Path) -> None:
    """qa_metrics 是脱敏表：不存 user_id、不存问题原文。"""
    add_metric(question_len=12, answerable=True, top_doc_id=7, top_score=0.79,
               latency_ms=830, degraded=False, db_path=db)

    with get_conn(db) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(qa_metrics)")}
        row = conn.execute("SELECT * FROM qa_metrics").fetchone()

    assert "user_id" not in columns
    assert row["question_len"] == 12
    assert row["answerable"] == 1
    assert row["top_doc_id"] == 7
    assert row["degraded"] == 0


def test_add_metric_marks_refusal_and_degradation(db: Path) -> None:
    add_metric(question_len=5, answerable=False, latency_ms=3, db_path=db)
    add_metric(question_len=5, answerable=True, degraded=True, db_path=db)

    with get_conn(db) as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM qa_metrics ORDER BY id")]

    assert [row["answerable"] for row in rows] == [0, 1]
    assert [row["degraded"] for row in rows] == [0, 1]


def test_tc_u20_cleanup_removes_only_expired_metrics(db: Path) -> None:
    """TC-U20（M3-08）：只删 90 天前的记录，保留期内的原样保留。"""
    add_metric(question_len=5, answerable=True, db_path=db)  # 刚写入，应保留
    with get_conn(db) as conn:
        conn.execute(
            """
            INSERT INTO qa_metrics (question_len, answerable, created_at)
            VALUES (?, ?, datetime('now', 'localtime', '-91 days'))
            """,
            (5, 0),
        )

    removed = cleanup_metrics(days=90, db_path=db)

    with get_conn(db) as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM qa_metrics")]

    assert removed == 1
    assert [row["answerable"] for row in rows] == [1]


# ==================== feedback ====================


def test_add_feedback_then_read_back(db: Path) -> None:
    conversation_id = create_conversation(1, db_path=db)
    message_id = add_message(conversation_id, ROLE_ASSISTANT, "答案", db_path=db)

    assert get_feedback(message_id, 1, db_path=db) is None
    assert add_feedback(message_id, 1, RATING_USEFUL, db_path=db) is True
    assert get_feedback(message_id, 1, db_path=db) == RATING_USEFUL


def test_second_feedback_from_same_user_is_ignored(db: Path) -> None:
    """FR-18：同一用户对同一回答只能记一次，重复提交不覆盖也不报错。"""
    conversation_id = create_conversation(1, db_path=db)
    message_id = add_message(conversation_id, ROLE_ASSISTANT, "答案", db_path=db)
    add_feedback(message_id, 1, RATING_USEFUL, db_path=db)

    assert add_feedback(message_id, 1, RATING_USELESS, db_path=db) is False
    assert get_feedback(message_id, 1, db_path=db) == RATING_USEFUL


def test_feedback_is_per_user(db: Path) -> None:
    conversation_id = create_conversation(1, db_path=db)
    message_id = add_message(conversation_id, ROLE_ASSISTANT, "答案", db_path=db)

    add_feedback(message_id, 1, RATING_USEFUL, db_path=db)
    add_feedback(message_id, 2, RATING_USELESS, db_path=db)

    with get_conn(db) as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]

    assert total == 2
    assert get_feedback(message_id, 2, db_path=db) == RATING_USELESS
