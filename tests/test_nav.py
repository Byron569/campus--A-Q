"""导航框架测试。

覆盖 docs/05 §G-03（侧边栏为 PG-02~PG-06 共用）与 docs/07 §7（图标映射）。

这里能安全 `import app`，是因为 app.py 用 `if __name__ == "__main__"` 包住了
启动调用；否则 import 会把整个应用跑一遍、连真实数据库都建出来。
"""

from __future__ import annotations

from app import DEFAULT_PAGE, PAGES, PAGES_BY_KEY, _ICONS, resolve_page


def test_navigation_has_the_five_designed_entries() -> None:
    """docs/05 §G-03：导航为 PG-02~PG-06 五页共用，顺序与原型一致。"""
    assert [page["key"] for page in PAGES] == [
        "qa", "documents", "admin", "settings", "about",
    ]
    assert [page["label"] for page in PAGES] == [
        "问答", "我的文档", "管理员", "设置", "关于",
    ]


def test_every_nav_entry_has_its_lucide_icon() -> None:
    """docs/07 §7：五个导航项各有指定图标，不能漏画。"""
    assert set(_ICONS) == {page["key"] for page in PAGES}
    assert all(svg.strip() for svg in _ICONS.values())


def test_unimplemented_pages_declare_their_milestone() -> None:
    """未实现的页面必须写明计划里程碑，不能留空页骗点击。"""
    for page in PAGES:
        if page["key"] != DEFAULT_PAGE:
            assert page["plan"], f"{page['label']} 未标注里程碑"


def test_default_page_is_the_only_implemented_one() -> None:
    assert DEFAULT_PAGE == "documents"
    assert PAGES_BY_KEY[DEFAULT_PAGE]["plan"] == ""


def test_resolve_page_falls_back_on_missing_or_unknown() -> None:
    """查询参数可能被手改，非法值一律回落，不能拿它当可信输入。"""
    assert resolve_page(None) == DEFAULT_PAGE
    assert resolve_page("") == DEFAULT_PAGE
    assert resolve_page("qa") == "qa"
    assert resolve_page("../../etc/passwd") == DEFAULT_PAGE
    assert resolve_page("<script>alert(1)</script>") == DEFAULT_PAGE
