"""Agent 工具调度（二期 2.2）单元测试。

覆盖：规则层（含弱词需本人指代的收窄）、LLM 兜底层、降级回退、
工具执行与落库、日程工具的按用户隔离。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent import runtime, tools
from src.repository import (
    ROLE_ASSISTANT,
    ROLE_USER,
    Schedule,
    create_conversation,
    create_schedule,
    list_messages,
    list_sources_by_message,
)
from src.store.db import get_conn

NOW = datetime(2026, 10, 4, 12, 0)


class StubLLM:
    """按队列返回内容；exc 非空时模拟调用失败，并记录每次调用。"""

    def __init__(self, responses: list[str] | None = None, exc: Exception | None = None) -> None:
        self.responses = list(responses or [])
        self.exc = exc
        self.calls: list[object] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(content=self.responses.pop(0) if self.responses else "")


def _schedule(**overrides) -> Schedule:
    base = dict(
        id=1, user_id=1, title="编译原理实验三", type="homework", course="编译原理",
        due_at="2026-10-06 23:59", remind_days=1, status="pending", note="",
        reminded_at=None, created_at="", updated_at="",
    )
    base.update(overrides)
    return Schedule(**base)


# ==================== 规则层 ====================


def test_rule_route_time() -> None:
    assert runtime.rule_route("今天几号").name == tools.TOOL_TIME
    assert runtime.rule_route("现在几点").name == tools.TOOL_TIME
    assert runtime.rule_route("今天星期几").name == tools.TOOL_TIME


def test_rule_route_schedule_by_strong_term() -> None:
    choice = runtime.rule_route("我今天的日程")
    assert choice is not None
    assert choice.name == tools.TOOL_SCHEDULES
    assert choice.range_key == tools.RANGE_TODAY


def test_rule_route_weak_term_needs_mine_word() -> None:
    # 「考试 / 安排」这类弱词单独出现不足以判定为日程意图（可能是知识库问题）
    assert runtime.rule_route("考试安排在哪里查") is None
    assert runtime.rule_route("我的作业有哪些").name == tools.TOOL_SCHEDULES


def test_parse_range_variants() -> None:
    assert runtime.parse_range("今天的日程") == tools.RANGE_TODAY
    assert runtime.parse_range("本周的安排") == tools.RANGE_WEEK
    assert runtime.parse_range("有哪些逾期了") == tools.RANGE_OVERDUE
    assert runtime.parse_range("全部日程") == tools.RANGE_ALL
    assert runtime.parse_range("我的日程") == tools.RANGE_PENDING


def test_has_tool_signal() -> None:
    assert runtime.has_tool_signal("考试安排在哪里查")
    assert not runtime.has_tool_signal("补考怎么申请")


# ==================== LLM 兜底层 ====================


def test_parse_tool_name() -> None:
    assert runtime.parse_tool_name("list_schedules") == tools.TOOL_SCHEDULES
    assert runtime.parse_tool_name("current_time") == tools.TOOL_TIME
    assert runtime.parse_tool_name("knowledge_search") == tools.TOOL_KNOWLEDGE
    assert runtime.parse_tool_name("日程") == tools.TOOL_SCHEDULES
    assert runtime.parse_tool_name("毫无关系") is None


def test_decide_uses_rule_without_calling_llm() -> None:
    llm = StubLLM()
    choice = runtime.decide("我今天的日程", llm=llm)
    assert choice.name == tools.TOOL_SCHEDULES
    assert llm.calls == []


def test_decide_skips_llm_when_no_signal() -> None:
    llm = StubLLM()
    choice = runtime.decide("补考怎么申请", llm=llm)
    assert choice.name == tools.TOOL_KNOWLEDGE
    assert llm.calls == []


def test_decide_llm_fallback_selects_schedule() -> None:
    llm = StubLLM(["list_schedules"])
    choice = runtime.decide("考试安排在哪里查", llm=llm)
    assert choice.name == tools.TOOL_SCHEDULES
    assert len(llm.calls) == 1


def test_decide_llm_failure_falls_back_to_knowledge() -> None:
    llm = StubLLM(exc=RuntimeError("boom"))
    choice = runtime.decide("考试安排在哪里查", llm=llm)
    assert choice.name == tools.TOOL_KNOWLEDGE


def test_decide_llm_unrecognized_output_falls_back() -> None:
    llm = StubLLM(["随便说点什么"])
    choice = runtime.decide("考试安排在哪里查", llm=llm)
    assert choice.name == tools.TOOL_KNOWLEDGE


# ==================== 工具执行（纯） ====================


def test_current_time_text() -> None:
    text = tools.current_time_text(now=NOW)
    assert "2026-10-04 12:00" in text
    assert tools.WEEKDAYS[NOW.weekday()] in text


def test_filter_schedules_by_range() -> None:
    items = [
        _schedule(id=1, due_at="2026-10-03 09:00"),                 # 逾期
        _schedule(id=2, due_at="2026-10-04 09:00"),                 # 今天
        _schedule(id=3, due_at="2026-10-06 09:00"),                 # 未来
        _schedule(id=4, due_at="2026-10-20 09:00", status="done"),  # 已完成
    ]
    assert [s.id for s in tools.filter_schedules(items, range_key=tools.RANGE_TODAY, now=NOW)] == [2]
    assert [s.id for s in tools.filter_schedules(items, range_key=tools.RANGE_OVERDUE, now=NOW)] == [1]
    assert [s.id for s in tools.filter_schedules(items, range_key=tools.RANGE_WEEK, now=NOW)] == [2, 3]
    assert [s.id for s in tools.filter_schedules(items, range_key=tools.RANGE_PENDING, now=NOW)] == [1, 2, 3]
    assert [s.id for s in tools.filter_schedules(items, range_key=tools.RANGE_ALL, now=NOW)] == [1, 2, 3, 4]


def test_schedule_text_mentions_state_and_course() -> None:
    text = tools.schedule_text(
        [_schedule(id=1, due_at="2026-10-03 09:00")], range_key=tools.RANGE_OVERDUE, now=NOW
    )
    assert "编译原理实验三" in text
    assert "已逾期 1 天" in text
    assert "编译原理" in text


def test_schedule_text_empty_state() -> None:
    text = tools.schedule_text([], now=NOW)
    assert "没有" in text
    assert "日程" in text


# ==================== 工具执行（落库与隔离） ====================


def test_execute_time_tool_persists_message_and_metric(db: Path) -> None:
    conversation_id = create_conversation(1, db_path=db)
    result = runtime.execute(
        runtime.ToolChoice(tools.TOOL_TIME),
        "现在几点",
        user_id=1,
        conversation_id=conversation_id,
        db_path=db,
        now=NOW,
    )

    messages = list_messages(conversation_id, db_path=db)
    assert [m.role for m in messages] == [ROLE_USER, ROLE_ASSISTANT]
    assert messages[1].id == result.message_id
    assert "2026-10-04 12:00" in result.text

    # 工具回答没有引用来源
    assert list_sources_by_message(result.message_id, db_path=db) == []

    with get_conn(db) as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM qa_metrics")]
    assert len(rows) == 1
    assert rows[0]["answerable"] == 1


def test_execute_schedule_tool_isolated_by_user(db: Path) -> None:
    create_schedule(user_id=1, title="我的作业", due_at="2099-01-01 09:00", db_path=db)
    create_schedule(user_id=2, title="别人的作业", due_at="2099-01-01 09:00", db_path=db)
    conversation_id = create_conversation(1, db_path=db)

    result = runtime.execute(
        runtime.ToolChoice(tools.TOOL_SCHEDULES, range_key=tools.RANGE_PENDING),
        "我有哪些作业",
        user_id=1,
        conversation_id=conversation_id,
        db_path=db,
        now=NOW,
    )
    assert "我的作业" in result.text
    assert "别人的作业" not in result.text


def test_execute_rejects_knowledge_tool(db: Path) -> None:
    conversation_id = create_conversation(1, db_path=db)
    with pytest.raises(ValueError):
        runtime.execute(
            runtime.ToolChoice(tools.TOOL_KNOWLEDGE),
            "问题",
            user_id=1,
            conversation_id=conversation_id,
            db_path=db,
        )
