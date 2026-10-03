"""导航框架测试。

覆盖 docs/05 §G-03（侧边栏为 PG-02~PG-06 共用）与 docs/07 §7（图标映射）。

这里能安全 `import app`，是因为 app.py 用 `if __name__ == "__main__"` 包住了
启动调用；否则 import 会把整个应用跑一遍、连真实数据库都建出来。
"""

from __future__ import annotations

from app import DEFAULT_PAGE, PAGES, PAGES_BY_KEY, resolve_page

# 截至 M3 已实现的页面：问答（M2-05）、我的文档（M1-18）、设置（M3-16）、关于（M3-17）
IMPLEMENTED_PAGES = {"qa", "documents", "settings", "about"}


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
