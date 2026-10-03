"""作业 / 考试日程页（二期 2.3）。

设计依据：二期新增功能，设计要点经客户 2026-10-04 裁决：
- 范围为**仅个人日程**，只有本人可见（沿用 FR-31 的隔离口径）；
- 提醒为**仅站内提醒**——打开 / 刷新应用时计算并展示，不做推送；
- `schedules` 纳入备份 / 恢复（FR-24）。

提醒口径：某条日程处于「未完成、未提醒过、且已进入提前提醒窗口」时，
在页面顶部提示；用户点「知道了，不再提醒」后写入 `reminded_at`，不再重复提示。
之所以不「一展示就自动标记已提醒」：Streamlit 每次交互都会重跑脚本，
自动标记会让提示条闪现一次就消失，用户根本来不及看清。

纯逻辑（时间解析 / 分组 / 提醒判定）不依赖 Streamlit，可直接单元测试；
渲染函数只做 st.* 编排。
"""

from __future__ import annotations

import html
from datetime import date, datetime, time, timedelta

import streamlit as st

from config.settings import Settings
from src.repository import (
    SCHEDULE_DONE,
    SCHEDULE_EXAM,
    SCHEDULE_HOMEWORK,
    SCHEDULE_OTHER,
    SCHEDULE_PENDING,
    Schedule,
    User,
    create_schedule,
    delete_schedule,
    list_schedules,
    mark_schedules_reminded,
    set_schedule_status,
    update_schedule,
)

STATE_NOTICE = "schedule_notice"

TYPE_LABELS = {
    SCHEDULE_HOMEWORK: "作业",
    SCHEDULE_EXAM: "考试",
    SCHEDULE_OTHER: "其他",
}
TYPE_BADGES = {
    SCHEDULE_HOMEWORK: "作业",
    SCHEDULE_EXAM: "考试",
    SCHEDULE_OTHER: "其他",
}

DUE_FORMAT = "%Y-%m-%d %H:%M"

# 提醒可提前的天数上限（校验用）
MAX_REMIND_DAYS = 30


# ==================== 纯逻辑（不依赖 Streamlit） ====================


def parse_due(due_at: str) -> datetime:
    return datetime.strptime(due_at, DUE_FORMAT)


def days_until(schedule: Schedule, *, now: datetime | None = None) -> int:
    """距截止还有几天（按自然日算，可为负表示已逾期）。"""
    today = (now or datetime.now()).date()
    return (parse_due(schedule.due_at).date() - today).days


def reminder_due(schedule: Schedule, *, now: datetime | None = None) -> bool:
    """该日程当前是否需要站内提醒。

    按**自然日**比较，而不是精确到分秒：提前 1 天表示「截止日的前一天整天都提醒」。
    若按精确时间算，23:59 截止的日程要等到前一天 23:59 才提醒，形同失效。
    """
    if schedule.is_done or schedule.reminded_at:
        return False
    today = (now or datetime.now()).date()
    start = parse_due(schedule.due_at).date() - timedelta(days=schedule.remind_days)
    return today >= start


def due_state(schedule: Schedule, *, now: datetime | None = None) -> str:
    """展示分组用：overdue / today / upcoming / done。"""
    if schedule.is_done:
        return "done"
    left = days_until(schedule, now=now)
    if left < 0:
        return "overdue"
    if left == 0:
        return "today"
    return "upcoming"


def group_schedules(
    schedules: list[Schedule], *, now: datetime | None = None
) -> list[tuple[str, list[Schedule]]]:
    """按「逾期 / 今天 / 未来 / 已完成」分组，组内保持时间升序。"""
    buckets: dict[str, list[Schedule]] = {
        "overdue": [],
        "today": [],
        "upcoming": [],
        "done": [],
    }
    for item in schedules:
        buckets[due_state(item, now=now)].append(item)
    labels = {
        "overdue": "逾期",
        "today": "今天截止",
        "upcoming": "未来",
        "done": "已完成",
    }
    return [(labels[key], buckets[key]) for key in labels if buckets[key]]


def validate_schedule(
    *, title: str, due: datetime, remind_days: int
) -> str:
    """校验新建 / 编辑的日程字段，返回空串表示通过。"""
    if not (title or "").strip():
        return "请填写事项名称"
    if remind_days < 0 or remind_days > MAX_REMIND_DAYS:
        return f"提前提醒天数需在 0 ~ {MAX_REMIND_DAYS} 之间"
    if due is None:
        return "请选择截止时间"
    return ""


# ==================== 页面渲染 ====================


def render(*, settings: Settings, user: User) -> None:
    """渲染整个日程页。"""
    _drain_notice()
    st.markdown('<div class="cqa-page-title">日程</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cqa-page-desc">记录作业与考试的截止时间，临近时页面顶部会提醒你。'
        "日程只属于你自己，其他人看不到。</div>",
        unsafe_allow_html=True,
    )

    _render_create(settings=settings, user=user)
    _render_list(settings=settings, user=user)


def render_reminder(*, settings: Settings, user: User) -> None:
    """站内提醒条：在任意页面顶部展示当前需要提醒的日程。

    由 `app.py` 统一调用，因此「打开 / 刷新应用」即可看到提醒，而不必先进日程页。
    """
    now = datetime.now()
    pending = list_schedules(
        user.id, status=SCHEDULE_PENDING, db_path=settings.database_path
    )
    due = [item for item in pending if reminder_due(item, now=now)]
    if not due:
        return

    rows = "".join(
        f'<div class="cqa-remind-item">'
        f'<span class="cqa-chip cqa-chip--wait">{html.escape(TYPE_LABELS.get(item.type, item.type))}</span>'
        f"<span class=\"cqa-remind-title\">{html.escape(item.title)}</span>"
        f'<span class="cqa-remind-due">{html.escape(_due_text(item, now=now))}</span>'
        f"</div>"
        for item in due
    )
    st.markdown(
        f'<div class="cqa-remind"><div class="cqa-remind-head">日程提醒 · {len(due)} 项临近</div>'
        f"{rows}</div>",
        unsafe_allow_html=True,
    )
    if st.button("知道了，不再提醒", key="schedule-remind-dismiss"):
        mark_schedules_reminded([item.id for item in due], db_path=settings.database_path)
        st.rerun()


def _drain_notice() -> None:
    notice = st.session_state.pop(STATE_NOTICE, "")
    if notice:
        st.success(notice)


def _render_create(*, settings: Settings, user: User) -> None:
    _panel_head("新增日程")
    with st.form("schedule-create-form"):
        title = st.text_input("事项名称", placeholder="如：编译原理实验三", key="sch-new-title")
        left, mid, right = st.columns([2, 2, 1])
        with left:
            type_label = st.selectbox(
                "类型", list(TYPE_LABELS.values()), key="sch-new-type"
            )
        with mid:
            course = st.text_input("课程（可选）", key="sch-new-course")
        with right:
            remind_days = st.number_input(
                "提前提醒（天）", min_value=0, max_value=MAX_REMIND_DAYS, value=1,
                step=1, key="sch-new-remind",
            )
        date_col, time_col = st.columns(2)
        with date_col:
            due_date = st.date_input("截止日期", value=date.today(), key="sch-new-date")
        with time_col:
            due_time = st.time_input("截止时间", value=time(23, 59), key="sch-new-time")
        note = st.text_area("备注（可选）", key="sch-new-note", height=68)
        submitted = st.form_submit_button("保存日程", type="primary")

    if not submitted:
        return

    due = datetime.combine(due_date, due_time)
    reason = validate_schedule(title=title, due=due, remind_days=int(remind_days))
    if reason:
        st.error(reason)
        return

    create_schedule(
        user_id=user.id,
        title=title,
        due_at=due.strftime(DUE_FORMAT),
        type=_label_to_type(type_label),
        course=course,
        remind_days=int(remind_days),
        note=note,
        db_path=settings.database_path,
    )
    st.session_state[STATE_NOTICE] = "日程已保存"
    st.rerun()


def _render_list(*, settings: Settings, user: User) -> None:
    schedules = list_schedules(user.id, db_path=settings.database_path)
    header, _ = st.columns([3, 2])
    with header:
        _panel_head("我的日程", f"共 {len(schedules)} 项")

    if not schedules:
        st.markdown(
            '<div class="cqa-panel"><div class="cqa-empty">'
            "还没有日程，在上面记一条作业或考试试试。</div></div>",
            unsafe_allow_html=True,
        )
        return

    for label, items in group_schedules(schedules):
        st.markdown(
            f'<div class="cqa-group-head">{html.escape(label)} · {len(items)}</div>',
            unsafe_allow_html=True,
        )
        for item in items:
            _render_row(item, settings=settings, user=user)


def _render_row(item: Schedule, *, settings: Settings, user: User) -> None:
    info_col, action_col = st.columns([7, 3])
    with info_col:
        st.markdown(_row_html(item), unsafe_allow_html=True)
    with action_col:
        done_col, edit_col, del_col = st.columns(3)
        done_label = "恢复" if item.is_done else "完成"
        if done_col.button(done_label, key=f"sch-done-{item.id}"):
            next_status = SCHEDULE_PENDING if item.is_done else SCHEDULE_DONE
            set_schedule_status(item.id, next_status, db_path=settings.database_path)
            st.rerun()
        with edit_col.popover("编辑"):
            _render_edit(item, settings=settings)
        with del_col.popover("删除"):
            st.markdown(
                f'<div class="cqa-page-desc">确认删除《{html.escape(item.title)}》？'
                "不可恢复。</div>",
                unsafe_allow_html=True,
            )
            if st.button("确认删除", key=f"sch-del-{item.id}", type="primary"):
                delete_schedule(item.id, db_path=settings.database_path)
                st.rerun()


def _render_edit(item: Schedule, *, settings: Settings) -> None:
    due = parse_due(item.due_at)
    with st.form(f"schedule-edit-form-{item.id}"):
        title = st.text_input("事项名称", value=item.title, key=f"sch-edit-title-{item.id}")
        type_label = st.selectbox(
            "类型",
            list(TYPE_LABELS.values()),
            index=list(TYPE_LABELS).index(item.type) if item.type in TYPE_LABELS else 0,
            key=f"sch-edit-type-{item.id}",
        )
        course = st.text_input("课程（可选）", value=item.course, key=f"sch-edit-course-{item.id}")
        due_date = st.date_input("截止日期", value=due.date(), key=f"sch-edit-date-{item.id}")
        due_time = st.time_input("截止时间", value=due.time(), key=f"sch-edit-time-{item.id}")
        remind_days = st.number_input(
            "提前提醒（天）", min_value=0, max_value=MAX_REMIND_DAYS,
            value=int(item.remind_days), step=1, key=f"sch-edit-remind-{item.id}",
        )
        note = st.text_area("备注（可选）", value=item.note, key=f"sch-edit-note-{item.id}", height=68)
        submitted = st.form_submit_button("保存修改", type="primary")

    if not submitted:
        return

    new_due = datetime.combine(due_date, due_time)
    reason = validate_schedule(title=title, due=new_due, remind_days=int(remind_days))
    if reason:
        st.error(reason)
        return

    update_schedule(
        item.id,
        title=title,
        type=_label_to_type(type_label),
        course=course,
        due_at=new_due.strftime(DUE_FORMAT),
        remind_days=int(remind_days),
        note=note,
        db_path=settings.database_path,
    )
    st.session_state[STATE_NOTICE] = "日程已更新"
    st.rerun()


def _panel_head(title: str, subtitle: str = "") -> None:
    sub = f'<span class="cqa-sub">{subtitle}</span>' if subtitle else ""
    st.markdown(
        f'<div class="cqa-panel-head" style="border:0;padding:16px 0 6px">'
        f"<h3>{title}</h3>{sub}</div>",
        unsafe_allow_html=True,
    )


def _label_to_type(label: str) -> str:
    for key, name in TYPE_LABELS.items():
        if name == label:
            return key
    return SCHEDULE_OTHER


def _due_text(item: Schedule, *, now: datetime | None = None) -> str:
    left = days_until(item, now=now)
    if item.is_done:
        return "已完成"
    if left < 0:
        return f"已逾期 {-left} 天"
    if left == 0:
        return "今天截止"
    return f"还有 {left} 天"


def _row_html(item: Schedule) -> str:
    """一行日程的 HTML：类型徽标 + 名称/元信息 + 状态胶囊。"""
    badge = html.escape(TYPE_BADGES.get(item.type, item.type))
    meta_parts = [parse_due(item.due_at).strftime("%Y-%m-%d %H:%M")]
    if item.course:
        meta_parts.insert(0, item.course)
    meta = " · ".join(html.escape(part) for part in meta_parts)

    state = due_state(item)
    chip_class = {
        "done": "cqa-chip--ok",
        "overdue": "cqa-chip--fail",
        "today": "cqa-chip--run",
        "upcoming": "cqa-chip--wait",
    }[state]
    chip = f'<span class="cqa-chip {chip_class}">{html.escape(_due_text(item))}</span>'

    return (
        f'<div class="cqa-row">'
        f'<div class="cqa-fileicon">{badge}</div>'
        f'<div class="cqa-rowmain"><div class="cqa-name">{html.escape(item.title)}</div>'
        f'<div class="cqa-meta">{meta}</div></div>'
        f'<div class="cqa-rowstatus">{chip}</div>'
        f"</div>"
    )
