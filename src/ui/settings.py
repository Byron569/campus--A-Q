"""设置页（PG-05 / FR-25、FR-26）。

设计依据：
- docs/05-产品原型与交互说明.md §PG-05（显示名、改密、注销账号）、§G-08（危险操作需二次确认）
- docs/03-开发任务清单.md M3-05（注销）、M3-16（设置页）

三个表单都用 `st.form`：Streamlit 的 `text_input` 只在失焦/回车时提交值，
配普通按钮会出现「填了却没带上」（FB-2.4 踩过这个坑）。

注销的「二次确认」按原型实现为：**先勾选「我已知晓数据不可恢复」，确认按钮才可用**。
"""

from __future__ import annotations

import streamlit as st

from config.settings import Settings
from src.auth import service
from src.errors import AuthError, CampusQAError
from src.repository import User
from src.store.chroma import VectorStore

STATE_NOTICE = "settings_notice"


def render(*, settings: Settings, user: User, store: VectorStore) -> None:
    """渲染设置页。"""
    _drain_notice()

    st.markdown('<div class="cqa-page-title">设置</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="cqa-page-desc">当前账号：{user.username}'
        f'（{"管理员" if user.is_admin else "学生"}）</div>',
        unsafe_allow_html=True,
    )

    _render_display_name(user, settings)
    _render_password(user, settings)
    _render_delete_account(user, settings=settings, store=store)


def _drain_notice() -> None:
    notice = st.session_state.pop(STATE_NOTICE, "")
    if notice:
        st.success(notice)


def _render_display_name(user: User, settings: Settings) -> None:
    st.markdown(
        '<div class="cqa-panel-head" style="border:0;padding:16px 0 6px"><h3>显示名</h3></div>',
        unsafe_allow_html=True,
    )
    with st.form("settings-name-form"):
        name = st.text_input(
            "显示名", value=user.display_name or user.username, key="settings-name-input"
        )
        if st.form_submit_button("保存", type="primary"):
            try:
                service.change_display_name(user.id, name, db_path=settings.database_path)
            except AuthError as exc:
                st.error(str(exc))
            else:
                # 侧边栏的账号名要跟着变，必须重跑一次重新取用户
                st.session_state[STATE_NOTICE] = "已保存"
                st.rerun()


def _render_password(user: User, settings: Settings) -> None:
    st.markdown(
        '<div class="cqa-panel-head" style="border:0;padding:16px 0 6px"><h3>修改密码</h3></div>',
        unsafe_allow_html=True,
    )
    with st.form("settings-password-form"):
        old_password = st.text_input("原密码", type="password", key="settings-old-password")
        new_password = st.text_input("新密码（至少 6 位）", type="password", key="settings-new-password")
        confirm = st.text_input("确认新密码", type="password", key="settings-confirm-password")
        if st.form_submit_button("修改密码", type="primary"):
            _do_change_password(user, old_password, new_password, confirm, settings)


def _do_change_password(
    user: User, old_password: str, new_password: str, confirm: str, settings: Settings
) -> None:
    try:
        service.change_password(
            user.id,
            old_password,
            new_password,
            confirm=confirm,
            db_path=settings.database_path,
        )
    except AuthError as exc:
        st.error(str(exc))
        return

    st.success("密码已更新")


def _render_delete_account(user: User, *, settings: Settings, store: VectorStore) -> None:
    """注销账号（FR-25）：勾选确认后才可执行，成功后回登录页。"""
    st.markdown(
        '<div class="cqa-panel-head" style="border:0;padding:16px 0 6px"><h3>注销账号</h3></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cqa-page-desc">注销后你的个人文档、向量、会话与反馈会被一并清除，'
        "不可恢复。公共文档与他人数据不受影响。</div>",
        unsafe_allow_html=True,
    )

    agreed = st.checkbox("我已知晓数据不可恢复", key="delete-account-agreed")
    if st.button(
        "确认注销账号",
        key="delete-account-confirm",
        type="primary",
        disabled=not agreed,
    ):
        _do_delete_account(user, settings=settings, store=store)


def _do_delete_account(user: User, *, settings: Settings, store: VectorStore) -> None:
    try:
        service.delete_account(user.id, store=store, settings=settings)
    except CampusQAError as exc:
        # 失败时不进入 SQLite 事务，因此不会留下半删状态，直接重试即可
        st.error(f"注销失败，请重试：{exc}")
        return

    from src.ui import login

    login.logout()
