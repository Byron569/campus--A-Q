"""登录 / 注册页（PG-01 / FR-01、FR-02、FR-28）。

设计依据：
- docs/05-产品原型与交互说明.md §PG-01（元素与交互）、§G-01（访客拦截）、§G-06（就地提示）
- docs/02-架构设计.md §4.8（登录态存 `st.session_state`）
- docs/03-开发任务清单.md M3-03

三个交互细节值得说明：

1. **登录与注册同页切换、不改 URL**（PG-01 明确要求），靠 `st.session_state` 的模式标记切换。
2. **协议勾选框放在表单之外**：表单内的控件只有点提交按钮才会把值发给服务端，
   勾选框若在表单里，注册按钮的「未勾选即不可用」就没法实时生效。
   放在表单外，勾选会立刻触发一次重跑，按钮的 disabled 状态随即更新。
3. **文本框放表单里**：Streamlit 的 `text_input` 只在失焦/回车时才提交值，
   直接配普通按钮会出现「填了却没带上」的问题（FB-2.4 踩过）。
"""

from __future__ import annotations

import html

import streamlit as st

from config.settings import Settings, load_kb_config
from src.auth import service
from src.errors import AuthError
from src.repository import User
from src.ui import about

# 登录态：只存 user_id，用户资料每次从库里取，避免会话里的资料被改旧
STATE_USER_ID = "auth_user_id"
# 当前表单模式（登录 / 注册）
STATE_MODE = "auth_mode"

MODE_LOGIN = "login"
MODE_REGISTER = "register"


def start_session(user: User) -> None:
    """写入登录态并回到默认落地页（PG-01 → PG-02）。"""
    st.session_state[STATE_USER_ID] = user.id
    st.session_state.pop(STATE_MODE, None)
    _clear_page_state()
    st.rerun()


def logout() -> None:
    """退出登录，回登录页。

    必须同时清掉**页面级的会话态**（当前会话 id 等），否则下一个登录的人
    会直接落进上一个人的会话里（数据隔离红线）。
    """
    st.session_state.pop(STATE_USER_ID, None)
    st.session_state.pop(STATE_MODE, None)
    _clear_page_state()
    st.rerun()


def _clear_page_state() -> None:
    from src.ui import chat

    for key in (chat.STATE_CONVERSATION, chat.STATE_CATEGORY, chat.STATE_NOTICE):
        st.session_state.pop(key, None)


def render(*, settings: Settings) -> None:
    """渲染登录 / 注册页（未登录时的唯一入口）。"""
    school = load_kb_config()["school"]
    st.markdown(
        f'<div class="cqa-login-hero">'
        f'<div class="cqa-brandmark cqa-login-mark">校</div>'
        f'<div class="cqa-login-title">校答</div>'
        f'<div class="cqa-login-sub">{html.escape(str(school.get("name", "")))} 校园知识库问答</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    _, middle, _ = st.columns([1, 1.6, 1])
    with middle:
        if st.session_state.get(STATE_MODE) == MODE_REGISTER:
            _render_register(settings)
        else:
            _render_login(settings)


# ==================== 登录 ====================


def _render_login(settings: Settings) -> None:
    st.markdown(
        '<div class="cqa-panel-head" style="border:0;padding:0 0 8px"><h3>登录</h3></div>',
        unsafe_allow_html=True,
    )

    with st.form("login-form"):
        username = st.text_input("用户名", key="login-username")
        password = st.text_input("密码", type="password", key="login-password")
        submitted = st.form_submit_button("登 录", type="primary", use_container_width=True)

    if submitted:
        _do_login(username, password, settings)

    if st.button("还没有账号？去注册 →", key="to-register", use_container_width=True):
        st.session_state[STATE_MODE] = MODE_REGISTER
        st.rerun()


def _do_login(username: str, password: str, settings: Settings) -> None:
    if not (username or "").strip() or not password:
        st.error(service.INVALID_CREDENTIALS_TEXT)
        return

    try:
        user = service.login(username, password, db_path=settings.database_path)
    except AuthError as exc:
        # AccountDisabled 也是 AuthError 子类，两者都就地提示，不进 PG-02
        st.error(str(exc))
        return

    start_session(user)


# ==================== 注册 ====================


def _render_register(settings: Settings) -> None:
    st.markdown(
        '<div class="cqa-panel-head" style="border:0;padding:0 0 8px"><h3>注册</h3></div>',
        unsafe_allow_html=True,
    )

    # 放在表单之外，勾选后立即重跑，下面的注册按钮才会实时变可用（PG-01）
    agreed = st.checkbox("我已阅读并同意《用户协议与隐私说明》", key="register-agreed")
    with st.popover("查看《用户协议与隐私说明》", use_container_width=True):
        for title, body in about.privacy_sections():
            st.markdown(f"**{title}**")
            st.markdown(f'<div class="cqa-page-desc">{body}</div>', unsafe_allow_html=True)

    with st.form("register-form"):
        username = st.text_input("用户名", key="register-username")
        password = st.text_input("密码", type="password", key="register-password")
        confirm = st.text_input("确认密码", type="password", key="register-confirm")
        submitted = st.form_submit_button(
            "注册", type="primary", use_container_width=True, disabled=not agreed
        )

    if submitted:
        _do_register(username, password, confirm, agreed, settings)

    if st.button("已有账号？去登录 →", key="to-login", use_container_width=True):
        st.session_state[STATE_MODE] = MODE_LOGIN
        st.rerun()


def _do_register(
    username: str, password: str, confirm: str, agreed: bool, settings: Settings
) -> None:
    try:
        user = service.register(
            username,
            password,
            confirm=confirm,
            agreed=agreed,
            db_path=settings.database_path,
        )
    except AuthError as exc:
        st.error(str(exc))
        return

    st.success("注册成功，已自动登录。")
    start_session(user)
