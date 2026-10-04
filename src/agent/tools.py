"""Agent 工具集（二期 2.2，只读）。

设计依据：docs/02 v1.14 §4.14；客户 2026-10-04 裁决：
① v1 只做**只读**工具（知识库检索 / 查询我的日程 / 当前时间），
   日程的新增、修改、删除仍在「日程」页完成；
② 路由为「规则优先、LLM 兜底」。

本模块不依赖 Streamlit，工具执行与结果格式化均为纯函数，可直接单元测试。
`knowledge_search` 只是路由的默认档位，实际执行仍走既有 `src/rag/chain.py`，
不在本模块实现。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from src.repository import (
    Schedule,
    list_schedules,
    parse_schedule_due,
    schedule_days_until,
)

TOOL_KNOWLEDGE = "knowledge_search"
TOOL_SCHEDULES = "list_schedules"
TOOL_TIME = "current_time"

# 工具清单：描述供 LLM 路由选择，标签供界面提示
TOOL_DESCRIPTIONS = {
    TOOL_KNOWLEDGE: "查询校园知识库（通知、政策、流程、规定等公共资料）",
    TOOL_SCHEDULES: "查询用户自己的作业 / 考试日程",
    TOOL_TIME: "查询当前日期与时间",
}
TOOL_LABELS = {
    TOOL_KNOWLEDGE: "检索知识库",
    TOOL_SCHEDULES: "查询你的日程",
    TOOL_TIME: "查看当前时间",
}

# 日程查询范围
RANGE_TODAY = "today"
RANGE_WEEK = "week"
RANGE_OVERDUE = "overdue"
RANGE_ALL = "all"
RANGE_PENDING = "pending"  # 默认：未完成（含逾期）

RANGE_LABELS = {
    RANGE_TODAY: "今天",
    RANGE_WEEK: "本周",
    RANGE_OVERDUE: "已逾期",
    RANGE_ALL: "全部",
    RANGE_PENDING: "未完成",
}

SCHEDULE_TYPE_LABELS = {"homework": "作业", "exam": "考试", "other": "其他"}
WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


def current_time_text(*, now: datetime | None = None) -> str:
    """当前日期时间文案。"""
    moment = now or datetime.now()
    return f"现在是 {moment.strftime('%Y-%m-%d %H:%M')}，{WEEKDAYS[moment.weekday()]}。"


def filter_schedules(
    schedules: list[Schedule], *, range_key: str = RANGE_PENDING, now: datetime | None = None
) -> list[Schedule]:
    """按范围筛选日程。`RANGE_ALL` 含已完成，其余范围只看未完成的。"""
    if range_key == RANGE_ALL:
        return list(schedules)
    today = (now or datetime.now()).date()
    dates = {item.id: parse_schedule_due(item.due_at).date() for item in schedules}

    if range_key == RANGE_TODAY:
        return [item for item in schedules if dates[item.id] == today]
    if range_key == RANGE_WEEK:
        limit = today + timedelta(days=7)
        return [
            item
            for item in schedules
            if not item.is_done and today <= dates[item.id] <= limit
        ]
    if range_key == RANGE_OVERDUE:
        return [item for item in schedules if not item.is_done and dates[item.id] < today]
    return [item for item in schedules if not item.is_done]  # RANGE_PENDING


def schedule_text(
    schedules: list[Schedule], *, range_key: str = RANGE_PENDING, now: datetime | None = None
) -> str:
    """把日程列表格式化成一段可读文本（作为助手回答直接展示）。"""
    items = filter_schedules(schedules, range_key=range_key, now=now)
    label = RANGE_LABELS.get(range_key, "")
    if not items:
        # 区分「一条都没加过」与「有记录但不在当前范围」：否则用户刚加完日程却在
        # 「未完成」里看不到，会误以为调度没生效（实测反馈过这个问题）。
        if schedules:
            return (
                f"你当前没有{label}的日程；共有 {len(schedules)} 条日程记录，"
                "可在「日程」页查看全部。"
            )
        return "你还没有添加任何日程。可以在「日程」页添加作业或考试。"

    lines = [f"你{label}的日程共 {len(items)} 项："]
    for item in items:
        left = schedule_days_until(item, now=now)
        if item.is_done:
            state = "已完成"
        elif left < 0:
            state = f"已逾期 {-left} 天"
        elif left == 0:
            state = "今天截止"
        else:
            state = f"还有 {left} 天"
        course = f"{item.course} · " if item.course else ""
        kind = SCHEDULE_TYPE_LABELS.get(item.type, item.type)
        lines.append(f"- {item.due_at}　{item.title}（{kind}）· {course}{state}")
    return "\n".join(lines)


def list_schedule_text(
    user_id: int,
    *,
    range_key: str = RANGE_PENDING,
    db_path=None,
    now: datetime | None = None,
) -> str:
    """查询某用户的日程并格式化。`user_id` 必传，沿用 FR-31 的隔离口径。"""
    schedules = list_schedules(user_id, db_path=db_path)
    return schedule_text(schedules, range_key=range_key, now=now)
