"""日程 CRUD 与提醒（二期 2.3）单元测试。

覆盖：数据访问层 CRUD、按用户隔离、注销级联、备份纳入，
以及提醒判定的纯逻辑（时间窗口 / 分组 / 校验）。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.repository import (
    SCHEDULE_DONE,
    SCHEDULE_PENDING,
    Schedule,
    create_schedule,
    delete_schedule,
    delete_user,
    export_tables,
    get_schedule,
    import_tables,
    list_schedules,
    mark_schedules_reminded,
    set_schedule_status,
    update_schedule,
)
from src.store.db import get_conn
from src.ui.schedule import (
    days_until,
    due_state,
    group_schedules,
    reminder_due,
    validate_schedule,
)

NOW = datetime(2026, 10, 4, 12, 0)


def _schedule(**overrides) -> Schedule:
    """构造一个 Schedule 纯对象，用于测试不依赖数据库的判定逻辑。"""
    base = dict(
        id=1,
        user_id=1,
        title="编译原理实验三",
        type="homework",
        course="编译原理",
        due_at="2026-10-06 23:59",
        remind_days=1,
        status=SCHEDULE_PENDING,
        note="",
        reminded_at=None,
        created_at="",
        updated_at="",
    )
    base.update(overrides)
    return Schedule(**base)


# ==================== 数据访问层 ====================


def test_create_and_get_schedule(db: Path) -> None:
    schedule_id = create_schedule(
        user_id=1,
        title="  编译原理实验三  ",
        due_at="2026-10-06 23:59",
        course="编译原理",
        remind_days=2,
        note="第三章",
        db_path=db,
    )
    item = get_schedule(schedule_id, db_path=db)
    assert item is not None
    assert item.user_id == 1
    assert item.title == "编译原理实验三"  # 首尾空白被清理
    assert item.type == "homework"  # 默认类型
    assert item.course == "编译原理"
    assert item.remind_days == 2
    assert item.status == SCHEDULE_PENDING
    assert item.reminded_at is None


def test_create_schedule_blank_course_and_note_become_empty(db: Path) -> None:
    schedule_id = create_schedule(
        user_id=1, title="高数考试", due_at="2026-10-09 09:00", course="  ", note="",
        db_path=db,
    )
    item = get_schedule(schedule_id, db_path=db)
    assert item is not None
    assert item.course == ""
    assert item.note == ""


def test_list_schedules_sorted_by_due(db: Path) -> None:
    for due in ("2026-10-08 09:00", "2026-10-05 09:00", "2026-10-06 09:00"):
        create_schedule(user_id=1, title=due, due_at=due, db_path=db)
    dues = [item.due_at for item in list_schedules(1, db_path=db)]
    assert dues == ["2026-10-05 09:00", "2026-10-06 09:00", "2026-10-08 09:00"]


def test_list_schedules_isolated_by_user(db: Path) -> None:
    create_schedule(user_id=1, title="我的作业", due_at="2026-10-05 09:00", db_path=db)
    create_schedule(user_id=2, title="别人的作业", due_at="2026-10-05 09:00", db_path=db)

    mine = list_schedules(1, db_path=db)
    assert [item.title for item in mine] == ["我的作业"]

    others = list_schedules(2, db_path=db)
    assert [item.title for item in others] == ["别人的作业"]


def test_list_schedules_filter_by_status(db: Path) -> None:
    done_id = create_schedule(user_id=1, title="已完成", due_at="2026-10-05 09:00", db_path=db)
    create_schedule(user_id=1, title="待办", due_at="2026-10-06 09:00", db_path=db)
    set_schedule_status(done_id, SCHEDULE_DONE, db_path=db)

    pending = list_schedules(1, status=SCHEDULE_PENDING, db_path=db)
    assert [item.title for item in pending] == ["待办"]


def test_update_schedule_fields(db: Path) -> None:
    schedule_id = create_schedule(
        user_id=1, title="旧标题", due_at="2026-10-05 09:00", course="旧课", db_path=db
    )
    assert update_schedule(
        schedule_id,
        title="新标题",
        course="  ",
        due_at="2026-10-07 10:00",
        remind_days=0,
        db_path=db,
    )

    item = get_schedule(schedule_id, db_path=db)
    assert item is not None
    assert item.title == "新标题"
    assert item.course == ""  # 空白归一为空
    assert item.due_at == "2026-10-07 10:00"
    assert item.remind_days == 0


def test_update_schedule_rejects_unknown_field(db: Path) -> None:
    schedule_id = create_schedule(user_id=1, title="t", due_at="2026-10-05 09:00", db_path=db)
    with pytest.raises(ValueError):
        update_schedule(schedule_id, user_id=2, db_path=db)


def test_set_schedule_status_toggles(db: Path) -> None:
    schedule_id = create_schedule(user_id=1, title="t", due_at="2026-10-05 09:00", db_path=db)
    assert set_schedule_status(schedule_id, SCHEDULE_DONE, db_path=db)
    item = get_schedule(schedule_id, db_path=db)
    assert item is not None and item.is_done
    assert set_schedule_status(schedule_id, SCHEDULE_PENDING, db_path=db)
    item = get_schedule(schedule_id, db_path=db)
    assert item is not None and not item.is_done


def test_delete_schedule(db: Path) -> None:
    schedule_id = create_schedule(user_id=1, title="t", due_at="2026-10-05 09:00", db_path=db)
    assert delete_schedule(schedule_id, db_path=db)
    assert get_schedule(schedule_id, db_path=db) is None
    assert delete_schedule(schedule_id, db_path=db) is False


def test_mark_schedules_reminded_writes_timestamp(db: Path) -> None:
    first = create_schedule(user_id=1, title="a", due_at="2026-10-05 09:00", db_path=db)
    second = create_schedule(user_id=1, title="b", due_at="2026-10-06 09:00", db_path=db)

    assert mark_schedules_reminded([first], db_path=db) == 1
    assert get_schedule(first, db_path=db).reminded_at is not None
    assert get_schedule(second, db_path=db).reminded_at is None
    assert mark_schedules_reminded([], db_path=db) == 0


def test_delete_user_cascades_schedules(db: Path) -> None:
    create_schedule(user_id=1, title="待删", due_at="2026-10-05 09:00", db_path=db)
    create_schedule(user_id=2, title="保留", due_at="2026-10-05 09:00", db_path=db)

    delete_user(1, db_path=db)

    with get_conn(db) as conn:
        remaining = conn.execute("SELECT user_id FROM schedules").fetchall()
    assert [row["user_id"] for row in remaining] == [2]


def test_backup_includes_schedules(db: Path) -> None:
    create_schedule(user_id=1, title="备份我", due_at="2026-10-05 09:00", db_path=db)

    tables = export_tables(db_path=db)
    assert "schedules" in tables
    assert len(tables["schedules"]) == 1

    # 整库替换后再写回，日程不丢
    import_tables(tables, db_path=db)
    assert [item.title for item in list_schedules(1, db_path=db)] == ["备份我"]


# ==================== 提醒与分组的纯逻辑 ====================


def test_reminder_due_window() -> None:
    # 进入提前提醒窗口（截止前一天）：该提醒
    assert reminder_due(_schedule(due_at="2026-10-05 23:59", remind_days=1), now=NOW)
    # 窗口外（还有两天，只提前一天提醒）：不提醒
    assert not reminder_due(_schedule(due_at="2026-10-07 23:59", remind_days=1), now=NOW)
    # 已逾期仍未完成：继续提醒
    assert reminder_due(_schedule(due_at="2026-10-01 09:00", remind_days=0), now=NOW)


def test_reminder_due_skips_done_and_reminded() -> None:
    assert not reminder_due(
        _schedule(due_at="2026-10-05 23:59", status=SCHEDULE_DONE), now=NOW
    )
    assert not reminder_due(
        _schedule(due_at="2026-10-05 23:59", reminded_at="2026-10-04 09:00"), now=NOW
    )


def test_days_until_and_due_state() -> None:
    assert days_until(_schedule(due_at="2026-10-06 09:00"), now=NOW) == 2
    assert days_until(_schedule(due_at="2026-10-03 09:00"), now=NOW) == -1

    assert due_state(_schedule(due_at="2026-10-03 09:00"), now=NOW) == "overdue"
    assert due_state(_schedule(due_at="2026-10-04 09:00"), now=NOW) == "today"
    assert due_state(_schedule(due_at="2026-10-09 09:00"), now=NOW) == "upcoming"
    assert (
        due_state(_schedule(due_at="2026-10-09 09:00", status=SCHEDULE_DONE), now=NOW)
        == "done"
    )


def test_group_schedules_buckets_and_order() -> None:
    items = [
        _schedule(id=1, due_at="2026-10-09 09:00"),  # upcoming
        _schedule(id=2, due_at="2026-10-03 09:00"),  # overdue
        _schedule(id=3, due_at="2026-10-04 09:00"),  # today
        _schedule(id=4, due_at="2026-10-01 09:00", status=SCHEDULE_DONE),  # done
    ]
    groups = group_schedules(items, now=NOW)
    labels = [label for label, _ in groups]
    assert labels == ["逾期", "今天截止", "未来", "已完成"]
    assert [item.id for item in dict(groups)["逾期"]] == [2]
    assert [item.id for item in dict(groups)["已完成"]] == [4]


def test_group_schedules_skips_empty_buckets() -> None:
    groups = group_schedules([_schedule(id=1, due_at="2026-10-09 09:00")], now=NOW)
    assert [label for label, _ in groups] == ["未来"]


def test_validate_schedule() -> None:
    ok_due = NOW + timedelta(days=2)
    assert validate_schedule(title="  ", due=ok_due, remind_days=1) == "请填写事项名称"
    assert validate_schedule(title="作业", due=ok_due, remind_days=31) != ""
    assert validate_schedule(title="作业", due=ok_due, remind_days=1) == ""
