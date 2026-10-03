"""关于 / 隐私说明页（PG-06 / FR-28）。

设计依据：
- docs/05-产品原型与交互说明.md §PG-06（展示产品说明、数据使用说明、日志保留策略、学校联系方式）
- docs/01-需求规格说明书.md FR-28（协议须说明数据发送给 LLM API、日志范围与保留期）
- docs/02-架构设计.md §4.10（学校信息来自 `knowledge_base.yaml`）

内容与注册页的《用户协议与隐私说明》同源——注册页展示同一份文案的简版，
避免两处各写一套说法。
"""

from __future__ import annotations

import html

import streamlit as st

from config.settings import METRICS_RETENTION_DAYS, Settings, load_kb_config


def _section(title: str, body: str) -> None:
    st.markdown(
        f'<div class="cqa-panel" style="margin-bottom:16px">'
        f'<div class="cqa-panel-head"><h3>{html.escape(title)}</h3></div>'
        f'<div style="padding:14px 16px;font-size:13px;line-height:1.8">{body}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def privacy_sections() -> list[tuple[str, str]]:
    """返回协议与隐私说明的条目（标题, HTML 正文）。

    注册页的勾选弹窗与「关于」页**共用这一份文案**，避免两处各写一套说法。
    """
    school = load_kb_config()["school"]
    name = html.escape(str(school.get("name", "")))
    contact = html.escape(str(school.get("contact", "")))

    sections = [
        (
            "这是什么",
            f"{name} 校园知识库问答助手「校答」，用于把学校的公开通知、手册、管理办法等资料"
            "整理成可检索的知识库，学生用自然语言提问即可得到带出处的回答。"
            "答案只依据已入库资料生成，资料无法回答时会明确拒答，不做猜测。",
        ),
        (
            "数据如何使用",
            "1. 你上传的资料保存在本机服务器的 <code>data/</code> 目录，向量化在本机完成，"
            "向量不出本机。<br>"
            "2. 提问时，系统会把检索到的相关资料片段与你的问题一起发送给大模型接口"
            "（默认 DeepSeek，可在配置中更换供应商）用于生成回答。<br>"
            "3. 会话记录与引用来源保存在本机数据库，仅你本人可见；"
            "公共资料由管理员上传，所有登录用户均可检索。<br>"
            "4. 系统不会把你的密码以明文形式保存，密码使用 pbkdf2 加盐哈希存储。",
        ),
        (
            "日志与保留期",
            f"问答指标日志（问题长度、是否命中、耗时、是否降级等）保留 {METRICS_RETENTION_DAYS} 天，"
            "到期自动清理；该日志**不记录用户身份，也不记录问题原文**。<br>"
            "你可以随时在「设置」页注销账号，注销后个人文档、向量、会话与反馈会被一并清除。",
        ),
    ]

    notice = str(school.get("notice", "")).strip()
    if notice:
        sections.append(("免责说明", html.escape(notice)))

    sections.append(("联系方式", f"如对知识库内容有疑问，请联系：<strong>{contact}</strong>"))
    return sections


def render(*, settings: Settings) -> None:
    """渲染关于 / 隐私说明页。"""
    school = load_kb_config()["school"]
    name = html.escape(str(school.get("name", "")))

    st.markdown('<div class="cqa-page-title">关于</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="cqa-page-desc">{name} 校园知识库问答，只依据学校公开资料作答。</div>',
        unsafe_allow_html=True,
    )

    for title, body in privacy_sections():
        _section(title, body)
