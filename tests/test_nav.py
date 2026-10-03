"""导航框架测试。

覆盖 docs/05 §G-03（侧边栏为 PG-02~PG-06 共用）与 docs/07 §7（图标映射）。

这里能安全 `import app`，是因为 app.py 用 `if __name__ == "__main__"` 包住了
启动调用；否则 import 会把整个应用跑一遍、连真实数据库都建出来。
"""

from __future__ import annotations

import app as app_module
from app import DEFAULT_PAGE, PAGES, PAGES_BY_KEY, current_user, resolve_page
from src.auth.security import hash_password
from src.repository import (
    USER_STATUS_DISABLED,
    create_user,
    delete_user,
    set_user_status,
)
from src.ui import login

# 截至 M3 已实现的页面：问答（M2-05）、我的文档（M1-18）、管理员（M3-04）、
# 设置（M3-16）、关于（M3-17）——五个页面全部交付
IMPLEMENTED_PAGES = {"qa", "documents", "admin", "settings", "about"}


def test_navigation_has_the_five_designed_entries() -> None:
    """docs/05 §G-03：导航为 PG-02~PG-06 五页共用，顺序与原型一致。"""
    assert [page["key"] for page in PAGES] == [
        "qa", "documents", "admin", "settings", "about",
    ]
    assert [page["label"] for page in PAGES] == [
        "问答", "我的文档", "管理员", "设置", "关于",
    ]


def test_every_nav_entry_has_an_icon() -> None:
    """五个导航项各有图标，不能漏。

    图标实现由设计指定的 lucide 换成 Streamlit 的 Material（导航已改为按钮，
    按钮只接受 Material 图标），该差异见 FB-3.2 自审清单。
    """
    assert all(page["icon"].startswith(":material/") for page in PAGES)
    assert len({page["icon"] for page in PAGES}) == len(PAGES)


def test_unimplemented_pages_declare_their_milestone() -> None:
    """未实现的页面必须写明计划里程碑，不能留空页骗点击。"""
    for page in PAGES:
        if page["key"] in IMPLEMENTED_PAGES:
            assert page["plan"] == "", f"{page['label']} 已实现，不应再写计划里程碑"
        else:
            assert page["plan"], f"{page['label']} 未标注里程碑"


def test_default_page_is_qa_after_m2() -> None:
    """G-04：进入系统默认落地问答页，且该页已实现。"""
    assert DEFAULT_PAGE == "qa"
    assert DEFAULT_PAGE in IMPLEMENTED_PAGES
    assert PAGES_BY_KEY[DEFAULT_PAGE]["plan"] == ""


def test_resolve_page_falls_back_on_missing_or_unknown() -> None:
    """查询参数可能被手改，非法值一律回落，不能拿它当可信输入。"""
    assert resolve_page(None) == DEFAULT_PAGE
    assert resolve_page("") == DEFAULT_PAGE
    assert resolve_page("qa") == "qa"
    assert resolve_page("../../etc/passwd") == DEFAULT_PAGE
    assert resolve_page("<script>alert(1)</script>") == DEFAULT_PAGE


# ==================== C-07：禁用 / 删除账号后登录态立即失效 ====================


def _seed_login_state(monkeypatch, user_id: int) -> dict:
    """把 st.session_state 换成普通 dict，便于在测试里驱动登录态。"""
    state = {login.STATE_USER_ID: user_id}
    monkeypatch.setattr(app_module.st, "session_state", state)
    return state


def test_current_user_accepts_active_account(db, settings, monkeypatch) -> None:
    user_id = create_user(username="alice", password_hash=hash_password("secret123"), db_path=db)
    _seed_login_state(monkeypatch, user_id)

    assert current_user(settings=settings).id == user_id


def test_current_user_invalidates_disabled_account(db, settings, monkeypatch) -> None:
    """账号被禁用后，已建立的登录态**立即**失效，不必等用户重新登录。"""
    user_id = create_user(username="alice", password_hash=hash_password("secret123"), db_path=db)
    state = _seed_login_state(monkeypatch, user_id)

    set_user_status(user_id, USER_STATUS_DISABLED, db_path=db)

    assert current_user(settings=settings) is None
    assert login.STATE_USER_ID not in state


def test_current_user_invalidates_deleted_account(db, settings, monkeypatch) -> None:
    user_id = create_user(username="alice", password_hash=hash_password("secret123"), db_path=db)
    state = _seed_login_state(monkeypatch, user_id)

    delete_user(user_id, db_path=db)

    assert current_user(settings=settings) is None
    assert login.STATE_USER_ID not in state
