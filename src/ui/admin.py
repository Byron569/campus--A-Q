"""管理员页（PG-04 / FR-15、FR-16、FR-17）。

设计依据：
- docs/05-产品原型与交互说明.md §PG-04（公共文档 / 用户管理两个页签）
- docs/01-需求规格说明书.md FR-15（公共文档管理）、FR-16（用户管理）、FR-17（五维统计）
- docs/02-架构设计.md §4.10（公共文档必须指定分类）、§4.9（删除的跨库清理）、§6.4（五维统计）

范围：三个页签——公共文档、用户管理、统计。
五维统计（FR-17）为二期实现，指标口径见 docs/02 §6.4，聚合全部由 `repository` 完成。

上传与删除直接复用「我的文档」页那套已测过的函数（`stage_upload` / `enqueue_new`
/ `delete_document`），区别只在 `is_public=True`、`user_id=None`（公共文档在
Chroma 元数据里的 user_id 固定为 -1）。

安全：管理员**不能禁用或删除自己**。单管理员部署下误操作会把自己锁在系统外，
恢复只能直接改数据库。
"""

from __future__ import annotations

import html

import streamlit as st

from config.settings import ALLOWED_SUFFIXES, UNCATEGORIZED_KEY, Settings, category_options
from src.auth import service
from src.errors import CampusQAError
from src.repository import (
    USER_STATUS_ACTIVE,
    User,
    get_task_by_doc,
    list_documents,
    list_users,
    set_user_status,
    stats_documents,
    stats_qa_volume,
    stats_quality,
    stats_users,
    stats_worst_answers,
)
from src.store.chroma import VectorStore
from src.ui.documents import delete_document, enqueue_new, stage_upload, validate_upload

STATE_PUBLIC_PAGE = "admin_public_page"
STATE_USER_PAGE = "admin_user_page"
STATE_USER_KEYWORD = "admin_user_keyword"
STATE_RESULTS = "admin_results"
STATE_SUBMITTED = "admin_submitted"

VOLUME_DAYS = 30
WORST_LIMIT = 5
SNIPPET_LENGTH = 60


def render(*, settings: Settings, store: VectorStore, user: User) -> None:
    """渲染管理员页。角色守卫在 app.py 已做，这里再兜一次（纵深防御）。"""
    service.require_admin(user)

    st.markdown('<div class="cqa-page-title">管理员</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cqa-page-desc">公共文档对所有登录用户可检索；'
        "用户管理可禁用、启用或删除账号；统计展示知识库与问答的整体情况。</div>",
        unsafe_allow_html=True,
    )

    _drain_notice()
    public_tab, users_tab, stats_tab = st.tabs(["公共文档", "用户管理", "统计"])
    with public_tab:
        _render_public_tab(settings=settings, store=store)
    with users_tab:
        _render_users_tab(settings=settings, store=store, current=user)
    with stats_tab:
        _render_stats_tab(settings=settings)
    _drain_results()


# ==================== 公共文档 ====================


def _render_public_tab(*, settings: Settings, store: VectorStore) -> None:
    _render_public_upload(settings=settings, store=store)
    _render_public_list(settings=settings, store=store)


def _render_public_upload(*, settings: Settings, store: VectorStore) -> None:
    suffix_hint = " / ".join(sorted(s.lstrip(".").upper() for s in ALLOWED_SUFFIXES))
    st.markdown(
        f'<div class="cqa-panel-head" style="border:0;padding:8px 0 6px">'
        f"<h3>上传公共文档</h3>"
        f'<span class="cqa-sub">支持 {suffix_hint}，单文件不超过 {settings.max_upload_mb}MB，分类必选</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # 公共文档必须指定分类（docs/02 §4.10），因此不提供「未分类」
    options = [item for item in category_options() if item["key"] != UNCATEGORIZED_KEY]
    label_to_key = {item["name"]: item["key"] for item in options}

    left, right = st.columns([3, 2])
    with left:
        files = st.file_uploader(
            "选择要上传为公共文档的文件（可多选）",
            type=sorted(suffix.lstrip(".") for suffix in ALLOWED_SUFFIXES),
            accept_multiple_files=True,
            key="admin-uploader",
        )
    with right:
        picked = st.selectbox(
            "分类（必选）",
            list(label_to_key),
            index=None,
            placeholder="请选择分类",
            key="admin-category",
        )
        start = st.button("上传公共文档", type="primary", key="admin-upload-start")

    if not start:
        return
    if not files:
        st.warning("请先选择文件。")
        return
    if not picked:
        st.warning("公共文档必须指定分类。")
        return

    category = label_to_key[picked]
    submitted: set[str] = st.session_state.setdefault(STATE_SUBMITTED, set())
    results = [
        _submit_public(uploaded, category=category, submitted=submitted, settings=settings, store=store)
        for uploaded in files
    ]
    st.session_state[STATE_RESULTS] = results
    st.rerun()


def _submit_public(
    uploaded, *, category: str, submitted: set[str], settings: Settings, store: VectorStore
) -> tuple[str, str]:
    fingerprint = f"{uploaded.name}|{uploaded.size}"
    if fingerprint in submitted:
        return "info", f"已跳过（本次会话已提交过）：{uploaded.name}"

    reason = validate_upload(uploaded.name, uploaded.size, settings=settings)
    if reason:
        return "error", reason

    try:
        staged = stage_upload(
            uploaded.name,
            uploaded.getvalue(),
            category=category,
            is_public=True,
            user_id=None,
            settings=settings,
        )
        enqueue_new(
            staged,
            category=category,
            is_public=True,
            user_id=None,
            store=store,
            settings=settings,
        )
    except CampusQAError as exc:
        return "error", f"{uploaded.name}：{exc}"

    submitted.add(fingerprint)
    return "success", f"{uploaded.name} 已提交入库"


def _render_public_list(*, settings: Settings, store: VectorStore) -> None:
    page = int(st.session_state.get(STATE_PUBLIC_PAGE, 1))
    page_size = settings.page_size
    # user_id 为空 + include_public → 只返回公共文档
    rows, total = list_documents(
        include_public=True, page=page, page_size=page_size, db_path=settings.database_path
    )
    total_pages = max((total + page_size - 1) // page_size, 1)

    header, nav = st.columns([3, 2])
    with header:
        st.markdown(
            f'<div class="cqa-panel-head" style="border:0;padding:16px 0 6px">'
            f"<h3>公共文档</h3><span class=\"cqa-sub\">共 {total} 条 · 每页 {page_size} 条</span></div>",
            unsafe_allow_html=True,
        )
    with nav:
        prev_col, page_col, next_col = st.columns([1, 2, 1])
        if prev_col.button("上一页", disabled=page <= 1 or not rows, key="admin-pub-prev"):
            st.session_state[STATE_PUBLIC_PAGE] = max(page - 1, 1)
            st.rerun()
        page_col.markdown(
            f'<div class="cqa-pager">第 {page} / {total_pages} 页</div>', unsafe_allow_html=True
        )
        if next_col.button("下一页", disabled=page >= total_pages or not rows, key="admin-pub-next"):
            st.session_state[STATE_PUBLIC_PAGE] = page + 1
            st.rerun()

    if not rows:
        st.markdown(
            '<div class="cqa-panel"><div class="cqa-empty">'
            "还没有公共文档，上传后可被所有学生检索。</div></div>",
            unsafe_allow_html=True,
        )
        return

    for document in rows:
        _render_public_row(document, settings=settings, store=store)


def _render_public_row(document, *, settings: Settings, store: VectorStore) -> None:
    name = html.escape(document.filename)
    badge = html.escape((document.filetype or "?").upper()[:4])
    meta = (
        f"{html.escape(document.category)} · {document.size_bytes / 1024:.0f} KB · "
        f"{document.chunk_count} 个切片"
    )

    info_col, action_col = st.columns([7, 3])
    with info_col:
        st.markdown(
            f'<div class="cqa-row">'
            f'<div class="cqa-fileicon">{badge}</div>'
            f'<div class="cqa-rowmain"><div class="cqa-name">{name}</div>'
            f'<div class="cqa-meta">{meta}</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with action_col:
        with st.popover("删除", key=f"admin-del-{document.id}"):
            st.markdown(
                f'<div class="cqa-page-desc">确认删除公共文档《{name}》？'
                "删除后所有用户都将检索不到，原文与向量一并清除。</div>",
                unsafe_allow_html=True,
            )
            if st.button("确认删除", key=f"admin-del-ok-{document.id}", type="primary"):
                try:
                    delete_document(document.id, store=store, settings=settings)
                except CampusQAError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()


# ==================== 用户管理 ====================


def _render_users_tab(*, settings: Settings, store: VectorStore, current: User) -> None:
    keyword = st.text_input(
        "搜索用户名或显示名", key=STATE_USER_KEYWORD, placeholder="留空显示全部"
    )
    page = int(st.session_state.get(STATE_USER_PAGE, 1))
    page_size = settings.page_size
    rows, total = list_users(
        page=page, page_size=page_size, keyword=keyword, db_path=settings.database_path
    )
    total_pages = max((total + page_size - 1) // page_size, 1)

    header, nav = st.columns([3, 2])
    with header:
        st.markdown(
            f'<div class="cqa-panel-head" style="border:0;padding:16px 0 6px">'
            f"<h3>用户</h3><span class=\"cqa-sub\">共 {total} 人</span></div>",
            unsafe_allow_html=True,
        )
    with nav:
        prev_col, page_col, next_col = st.columns([1, 2, 1])
        if prev_col.button("上一页", disabled=page <= 1 or not rows, key="admin-user-prev"):
            st.session_state[STATE_USER_PAGE] = max(page - 1, 1)
            st.rerun()
        page_col.markdown(
            f'<div class="cqa-pager">第 {page} / {total_pages} 页</div>', unsafe_allow_html=True
        )
        if next_col.button("下一页", disabled=page >= total_pages or not rows, key="admin-user-next"):
            st.session_state[STATE_USER_PAGE] = page + 1
            st.rerun()

    if not rows:
        st.markdown(
            '<div class="cqa-panel"><div class="cqa-empty">没有匹配的用户。</div></div>',
            unsafe_allow_html=True,
        )
        return

    for target in rows:
        _render_user_row(target, current=current, settings=settings, store=store)


def _render_user_row(
    target: User, *, current: User, settings: Settings, store: VectorStore
) -> None:
    is_self = target.id == current.id
    name = html.escape(target.username)
    display = html.escape(target.display_name or target.username)
    role = "管理员" if target.is_admin else "学生"
    chip = (
        '<span class="cqa-chip cqa-chip--ok">正常</span>'
        if target.is_active
        else '<span class="cqa-chip cqa-chip--fail">已禁用</span>'
    )

    info_col, action_col = st.columns([6, 4])
    with info_col:
        st.markdown(
            f'<div class="cqa-row">'
            f'<div class="cqa-fileicon">{html.escape(role[:2])}</div>'
            f'<div class="cqa-rowmain"><div class="cqa-name">{name}'
            f'{"（当前登录）" if is_self else ""}</div>'
            f'<div class="cqa-meta">{display} · {role}</div></div>'
            f'<div class="cqa-rowstatus">{chip}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with action_col:
        toggle_col, delete_col = st.columns(2)
        if target.is_active:
            if toggle_col.button(
                "禁用", key=f"admin-disable-{target.id}", disabled=is_self,
                help="不能操作自己的账号" if is_self else None,
            ):
                set_user_status(target.id, "disabled", db_path=settings.database_path)
                st.session_state[STATE_RESULTS] = [("success", f"已禁用 {target.username}")]
                st.rerun()
        else:
            if toggle_col.button("启用", key=f"admin-enable-{target.id}"):
                set_user_status(target.id, "active", db_path=settings.database_path)
                st.session_state[STATE_RESULTS] = [("success", f"已启用 {target.username}")]
                st.rerun()

        with delete_col.popover("删除", key=f"admin-user-del-{target.id}"):
            st.markdown(
                f'<div class="cqa-page-desc">确认删除账号《{name}》？'
                "其个人文档、向量、会话与反馈将一并清除，不可恢复。</div>",
                unsafe_allow_html=True,
            )
            if st.button(
                "确认删除",
                key=f"admin-user-del-ok-{target.id}",
                type="primary",
                disabled=is_self,
            ):
                try:
                    service.delete_account(
                        target.id, store=store, settings=settings
                    )
                except CampusQAError as exc:
                    st.error(str(exc))
                else:
                    st.session_state[STATE_RESULTS] = [("success", f"已删除 {target.username}")]
                    st.rerun()


# ==================== 统计（FR-17，二期）====================


def _render_stats_tab(*, settings: Settings) -> None:
    """五维统计：用户 / 文档 / 问答量 / 质量 / 差评榜（口径见 docs/02 §6.4）。"""
    db = settings.database_path
    users = stats_users(db)
    documents = stats_documents(db)
    quality = stats_quality(db)
    volume = stats_qa_volume(VOLUME_DAYS, db)

    _stats_heading("用户")
    _metrics(
        [
            ("用户总数", str(users.total)),
            ("今日活跃", str(users.active_today)),
            ("已禁用", str(users.disabled)),
        ]
    )

    _stats_heading("文档")
    _metrics(
        [
            ("文档总数", str(documents.total)),
            ("公共文档", str(documents.public)),
            ("个人文档", str(documents.personal)),
            ("入库失败任务", str(documents.failed_tasks)),
        ]
    )
    st.caption(f"公共文档占比 {documents.public_ratio * 100:.1f}%")

    _stats_heading("问答量")
    _metrics(
        [
            ("近 7 天", str(sum(count for _, count in volume[-7:]))),
            ("近 30 天", str(sum(count for _, count in volume))),
        ]
    )
    if any(count for _, count in volume):
        _volume_chart(volume)
    else:
        _stats_empty("近 30 天还没有问答记录。")

    _stats_heading("质量")
    _metrics(
        [
            ("拒答率", f"{quality.refusal_rate * 100:.1f}%"),
            ("降级次数", str(quality.degraded)),
            ("平均耗时", f"{quality.avg_latency_ms:.0f} ms"),
        ]
    )
    st.caption(f"样本 {quality.total} 次问答；指标已脱敏，不含问题原文与用户身份。")

    _stats_heading("差评榜")
    worst = stats_worst_answers(WORST_LIMIT, db)
    if not worst:
        _stats_empty("还没有「没用」反馈。")
    for answer in worst:
        _render_worst_row(answer)


def _stats_heading(title: str) -> None:
    st.markdown(
        f'<div class="cqa-panel-head" style="border:0;padding:14px 0 6px"><h3>{html.escape(title)}</h3></div>',
        unsafe_allow_html=True,
    )


def _stats_empty(text: str) -> None:
    st.markdown(
        f'<div class="cqa-panel"><div class="cqa-empty">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def _metrics(items: list[tuple[str, str]]) -> None:
    for column, (label, value) in zip(st.columns(len(items)), items):
        column.metric(label, value)


def _volume_chart(volume: list[tuple[str, int]]) -> None:
    """问答量趋势图。pandas 由 Streamlit 传递依赖，此处延迟导入。"""
    import pandas as pd

    frame = pd.DataFrame(volume, columns=["日期", "问答次数"]).set_index("日期")
    st.bar_chart(frame, height=220)


def _render_worst_row(answer) -> None:
    snippet = " ".join((answer.content or "").split())
    if len(snippet) > SNIPPET_LENGTH:
        snippet = snippet[:SNIPPET_LENGTH] + "…"
    snippet = snippet or "（空回答）"
    st.markdown(
        f'<div class="cqa-row">'
        f'<div class="cqa-fileicon">差</div>'
        f'<div class="cqa-rowmain"><div class="cqa-name">{html.escape(snippet)}</div>'
        f'<div class="cqa-meta">没用 {answer.useless} / 共 {answer.rated} 次评价</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ==================== 结果与提示 ====================


def _drain_notice() -> None:
    notice = st.session_state.pop("admin_notice", "")
    if notice:
        st.warning(notice)


def _drain_results() -> None:
    results: list[tuple[str, str]] = st.session_state.pop(STATE_RESULTS, [])
    for level, text in results:
        if level == "success":
            st.success(text)
        elif level == "error":
            st.error(text)
        else:
            st.info(text)
