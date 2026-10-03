"""管理员五维统计单元测试（FR-17，二期）。

覆盖 docs/02 §6.4 的五个维度聚合口径：用户 / 文档 / 问答量 / 质量 / 差评榜。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.repository import (
    RATING_USEFUL,
    RATING_USELESS,
    TASK_FAILED,
    add_feedback,
    add_message,
    add_metric,
    create_conversation,
    create_document,
    create_task,
    soft_delete_document,
    stats_documents,
    stats_qa_volume,
    stats_quality,
    stats_users,
    stats_worst_answers,
    update_task,
)


# ==================== 用户维度 ====================


def test_stats_users_counts_total_disabled_and_active_today(db: Path) -> None:
    from src.repository import USER_STATUS_DISABLED, set_user_status

    set_user_status(2, USER_STATUS_DISABLED, db_path=db)
    conversation = create_conversation(1, db_path=db)
    add_message(conversation, "user", "今天提过问", db_path=db)

    stats = stats_users(db_path=db)

    assert stats.total == 2
    assert stats.disabled == 1
    assert stats.active_today == 1


def test_stats_users_on_empty_activity(db: Path) -> None:
    stats = stats_users(db_path=db)

    assert stats.total == 2
    assert stats.active_today == 0
    assert stats.disabled == 0


# ==================== 文档维度 ====================


def test_stats_documents_splits_public_and_personal(db: Path) -> None:
    create_document(
        filename="公共.txt", filetype="txt", category="admin", is_public=True, db_path=db
    )
    create_document(
        filename="个人.txt", filetype="txt", category="course", user_id=1, db_path=db
    )
    deleted = create_document(
        filename="已删.txt", filetype="txt", category="course", user_id=1, db_path=db
    )
    soft_delete_document(deleted, db_path=db)

    failed_doc = create_document(
        filename="失败.txt", filetype="txt", category="course", user_id=1, db_path=db
    )
    task_id = create_task(failed_doc, user_id=1, db_path=db)
    update_task(task_id, db_path=db, status=TASK_FAILED)

    stats = stats_documents(db_path=db)

    assert stats.total == 3  # 已删除的不计入
    assert stats.public == 1
    assert stats.personal == 2
    assert stats.failed_tasks == 1
    assert stats.public_ratio == 1 / 3


def test_stats_documents_on_empty_database(db: Path) -> None:
    stats = stats_documents(db_path=db)

    assert stats.total == 0
    assert stats.public_ratio == 0.0  # 不除零


# ==================== 问答量维度 ====================


def test_stats_qa_volume_fills_missing_days_with_zero(db: Path) -> None:
    add_metric(question_len=5, answerable=True, db_path=db)
    add_metric(question_len=6, answerable=False, db_path=db)

    volume = stats_qa_volume(7, db_path=db)

    assert len(volume) == 7
    assert volume[-1][0] == datetime.now().date().isoformat()
    assert volume[-1][1] == 2
    assert all(count == 0 for _, count in volume[:-1])


# ==================== 质量维度 ====================


def test_stats_quality_aggregates_refusal_and_latency(db: Path) -> None:
    add_metric(question_len=5, answerable=True, latency_ms=100, db_path=db)
    add_metric(question_len=5, answerable=False, latency_ms=200, db_path=db)
    add_metric(question_len=5, answerable=True, degraded=True, latency_ms=300, db_path=db)

    stats = stats_quality(db_path=db)

    assert stats.total == 3
    assert stats.refused == 1
    assert stats.degraded == 1
    assert stats.avg_latency_ms == 200.0
    assert stats.refusal_rate == 1 / 3


def test_stats_quality_on_empty_metrics(db: Path) -> None:
    stats = stats_quality(db_path=db)

    assert stats.total == 0
    assert stats.refusal_rate == 0.0
    assert stats.avg_latency_ms == 0.0


# ==================== 差评榜 ====================


def test_stats_worst_answers_orders_by_useless_count(db: Path) -> None:
    conversation = create_conversation(1, db_path=db)
    bad = add_message(conversation, "assistant", "被两个人点没用的回答", db_path=db)
    worse = add_message(conversation, "assistant", "被一个人点没用", db_path=db)
    good = add_message(conversation, "assistant", "只有好评", db_path=db)

    add_feedback(bad, 1, RATING_USELESS, db_path=db)
    add_feedback(bad, 2, RATING_USELESS, db_path=db)
    add_feedback(worse, 1, RATING_USELESS, db_path=db)
    add_feedback(good, 1, RATING_USEFUL, db_path=db)

    worst = stats_worst_answers(db_path=db)

    assert [item.message_id for item in worst] == [bad, worse]
    assert worst[0].useless == 2
    assert worst[0].rated == 2
    assert all(item.message_id != good for item in worst)


def test_stats_worst_answers_respects_limit(db: Path) -> None:
    conversation = create_conversation(1, db_path=db)
    for index in range(3):
        message_id = add_message(conversation, "assistant", f"回答 {index}", db_path=db)
        add_feedback(message_id, 1, RATING_USELESS, db_path=db)

    assert len(stats_worst_answers(2, db_path=db)) == 2
