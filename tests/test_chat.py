"""问答页纯逻辑单元测试（M2-05 / M2-07）。

只覆盖不依赖 Streamlit 运行时的函数（校验、导出、文案），渲染逻辑靠浏览器验收。
"""

from __future__ import annotations

from src.repository import ROLE_ASSISTANT, ROLE_USER, Message
from src.ui.chat import (
    MAX_QUESTION_LEN,
    answer_clipboard_text,
    category_filter_options,
    conversation_title,
    export_markdown,
    score_text,
    source_labels,
    validate_question,
)


def msg(role: str, content: str, *, msg_id: int = 1) -> Message:
    return Message(
        id=msg_id, conversation_id=1, role=role, content=content,
        created_at="2026-10-03 10:00:00",
    )


# ==================== 输入校验 ====================


def test_empty_question_rejected() -> None:
    assert validate_question("") != ""
    assert validate_question("   ") != ""


def test_too_long_question_rejected() -> None:
    assert validate_question("问" * (MAX_QUESTION_LEN + 1)) != ""
    assert validate_question("问" * MAX_QUESTION_LEN) == ""


def test_valid_question_passes() -> None:
    assert validate_question("搬宿舍需要提前申请吗") == ""


# ==================== 相似度文案 ====================


def test_score_text_for_vector_hit() -> None:
    assert score_text(0.7967) == "相似度 0.80"


def test_score_text_for_keyword_only_hit() -> None:
    """纯 BM25 命中没有余弦分数，如实说明来源方式，不编造数值（DR-15）。"""
    assert score_text(None) == "关键词命中"


# ==================== 会话标题 ====================


def test_conversation_title_uses_first_question() -> None:
    messages = [msg(ROLE_USER, "搬宿舍需要提前申请吗"), msg(ROLE_ASSISTANT, "需要")]

    assert conversation_title(messages) == "搬宿舍需要提前申请吗"


def test_conversation_title_truncates_to_fifteen_chars() -> None:
    assert conversation_title([msg(ROLE_USER, "一二三四五六七八九十一二三四五六七八")]) == "一二三四五六七八九十一二三四五"


def test_conversation_title_for_empty_history() -> None:
    assert conversation_title([]) == "新会话"


# ==================== 复制答案 ====================


def test_clipboard_text_without_sources_is_the_answer() -> None:
    assert answer_clipboard_text("答案正文", []) == "答案正文"


def test_clipboard_text_appends_citation_list() -> None:
    text = answer_clipboard_text(
        "答案正文【来源1】",
        [{"filename": "学生手册.pdf", "score": 0.8}, {"filename": "通知.docx", "score": None}],
    )

    assert text.startswith("答案正文【来源1】")
    assert "1. 学生手册.pdf（相似度 0.80）" in text
    assert "2. 通知.docx（关键词命中）" in text


# ==================== 导出会话 ====================


def test_export_markdown_contains_messages_and_sources() -> None:
    user = msg(ROLE_USER, "搬宿舍要提前几天", msg_id=1)
    assistant = msg(ROLE_ASSISTANT, "三个工作日【来源1】", msg_id=2)

    markdown = export_markdown(
        title="搬宿舍要提前几天",
        messages=[user, assistant],
        sources={2: [{"filename": "宿舍搬迁通知.docx", "score": 0.7967}]},
    )

    assert markdown.startswith("# 搬宿舍要提前几天")
    assert "## 我" in markdown and "## 校答" in markdown
    assert "三个工作日【来源1】" in markdown
    assert "1. 宿舍搬迁通知.docx（相似度 0.80）" in markdown


def test_export_markdown_without_messages() -> None:
    assert export_markdown(title="新会话", messages=[], sources={}) == "# 新会话\n"


# ==================== 引用编号对齐（PG-02 一一对应）====================


def test_source_labels_follow_numbers_written_in_text() -> None:
    """模型只引用了第 1、5 条时，卡片必须标 1、5，不能重编成 1、2。"""
    labels = source_labels(
        "结论甲【来源1】，结论乙【来源5】。",
        [{"filename": "a.docx"}, {"filename": "b.docx"}],
    )

    assert [number for number, _ in labels] == [1, 5]
    assert [row["filename"] for _, row in labels] == ["a.docx", "b.docx"]


def test_source_labels_fall_back_when_text_has_no_citation() -> None:
    """降级分支正文没有引用标记，此时按顺序编号。"""
    labels = source_labels("模型服务暂不可用，以下为原始资料：", [{"filename": "a"}, {"filename": "b"}])

    assert [number for number, _ in labels] == [1, 2]


def test_export_markdown_uses_citation_numbers_from_text() -> None:
    assistant = msg(ROLE_ASSISTANT, "结论【来源5】", msg_id=2)

    markdown = export_markdown(
        title="会话",
        messages=[assistant],
        sources={2: [{"filename": "x.docx", "score": None}]},
    )

    assert "5. x.docx（关键词命中）" in markdown


def test_clipboard_text_uses_citation_numbers_from_text() -> None:
    text = answer_clipboard_text("结论【来源5】", [{"filename": "x.docx", "score": 0.8}])

    assert "5. x.docx（相似度 0.80）" in text


# ==================== 分类过滤（M2-07 / FR-14）====================


def test_category_options_start_with_all() -> None:
    options = category_filter_options()

    assert options[0] == ("全部", None)
    keys = [key for _, key in options]
    assert "uncategorized" in keys  # 内置的「未分类」也可筛
    assert len(keys) == len(set(keys))
