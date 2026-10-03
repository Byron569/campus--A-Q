"""主题与设计令牌测试。

覆盖 docs/07-设计令牌.md 的落地约束：
- §8.1 `config.toml` 能配的部分（含 DR-03：maxUploadSize 要比业务上限宽）
- §8.2 CSS 覆盖项里必须落到位的几条
- §3.1 / §5.2 / §4.1 的关键令牌值
- CR-01 仅浅色、DR-13 保留键盘焦点、A-04 卡片不用阴影
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from src.ui import theme

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_light_tokens_are_present() -> None:
    css = theme.stylesheet()

    assert "--background: #ffffff" in css
    assert "--foreground: #0e1115" in css
    assert "--primary: #0065fd" in css
    assert "--border: #e7eaef" in css
    assert "--ring: #557fff" in css
    assert "--sidebar: #eff1f4" in css
    # DR-04：模板原值 #7f8d9f 对比度不足，本项目加深（注释里会提到原值，故只比对声明）
    assert "--muted-foreground: #5b6675" in css
    assert "--muted-foreground: #7f8d9f" not in css


def test_semantic_colors_added_by_project() -> None:
    """DR-02：模板只有 destructive，入库完成 / 入库中无颜色可用，故新增两个。"""
    css = theme.stylesheet()

    assert "--success: #16a34a" in css
    assert "--info: #0ea5e9" in css
    assert "--destructive: #ef4444" in css


def test_three_tier_radius_replaces_template_value() -> None:
    """D-03：模板单一 19.2px（1.2rem）拆成控件 / 卡片 / 胶囊三级。"""
    css = theme.stylesheet()

    assert "--radius-control: 10px" in css
    assert "--radius-card: 14px" in css
    assert "--radius-pill: 999px" in css
    assert "1.2rem" not in css


def test_chinese_font_stack_is_included() -> None:
    """A-01：模板字体栈只有英文，中文界面会回退到系统默认。"""
    css = theme.stylesheet()

    assert "PingFang SC" in css
    assert "Microsoft YaHei" in css


def test_only_light_mode_no_dark_tokens() -> None:
    """CR-01：仅浅色，不注入模板的 .dark 块。"""
    css = theme.stylesheet()

    assert ".dark" not in css
    assert "--sidebar: #171717" not in css      # 模板暗色块里的侧边栏取值


def test_keyboard_focus_is_never_removed() -> None:
    """DR-13：可聚焦元素必须保留可见焦点，禁止整体 outline:none。"""
    css = theme.stylesheet()

    assert "outline: none" not in css
    assert "outline:none" not in css
    assert f"outline: 2px solid var(--ring)" in css


def _rule(css: str, selector: str) -> str:
    """取出某个选择器对应规则的声明块，用于精确断言而不是全文 grep。"""
    for block in css.split("}"):
        head, _, body = block.partition("{")
        if selector in head:
            return body
    raise AssertionError(f"样式中找不到选择器：{selector}")


def test_cards_and_panels_do_not_use_box_shadow() -> None:
    """A-04：模板为「边框分隔、不用阴影」，只有弹窗允许一层轻阴影（D-02）。"""
    # 自有面板 / 列表行 / 状态胶囊一律无阴影
    assert "box-shadow" not in theme.COMPONENTS
    # Streamlit 提示条自带的阴影必须显式取消，否则破调性
    assert "box-shadow: none" in _rule(theme.OVERRIDES, '[data-testid="stAlert"]')
    # 唯一允许的阴影在二次确认弹窗上
    assert "var(--shadow-dialog)" in _rule(theme.OVERRIDES, '[data-testid="stPopoverBody"]')


def test_upload_size_limit_leaves_room_for_business_check() -> None:
    """DR-03：Streamlit 的上传上限要比业务上限（20MB）宽。

    否则超限文件会整批被 Streamlit 拦下，用户看到的是「整批失败」，
    违反 FR-05「单文件失败不影响同批」。
    """
    config = tomllib.loads(
        (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    )

    assert config["theme"]["base"] == "light"
    assert config["theme"]["primaryColor"] == "#0065fd"
    assert config["server"]["maxUploadSize"] == 25
