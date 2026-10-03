"""防幻觉提示词与引用格式单元测试（M2-01）。

对应 docs/02 §4.5 的三条硬约束、`【来源N】` 格式，以及 DR-11 越界引用剔除。
"""

from __future__ import annotations

from src.rag.prompts import (
    SOURCE_PATTERN,
    build_context,
    build_refusal_text,
    build_system_prompt,
    parse_source_numbers,
    strip_invalid_sources,
)
from src.store.chroma import SearchHit


def hit(text: str, *, filename: str = "学生手册.pdf", doc_id: int = 1, chunk_index: int = 0) -> SearchHit:
    return SearchHit(
        text=text, score=0.8, filename=filename, doc_id=doc_id,
        chunk_index=chunk_index, category="freshman",
    )


# ==================== 三条硬约束 ====================


def test_system_prompt_carries_three_hard_constraints() -> None:
    prompt = build_system_prompt(contact="学生工作处")

    assert "只能依据" in prompt          # 约束一：只用资料
    assert "【来源N】" in prompt          # 约束二：引用格式
    assert "知识库中暂无相关资料" in prompt  # 约束三：无关即拒答
    assert "禁止猜测" in prompt


def test_system_prompt_injects_contact() -> None:
    assert "学生工作处" in build_system_prompt(contact="学生工作处")
    assert "{contact}" not in build_system_prompt(contact="学生工作处")


def test_refusal_text_uses_contact() -> None:
    assert build_refusal_text(contact="学生工作处") == "知识库中暂无相关资料，建议咨询学生工作处。"


def test_refusal_text_falls_back_to_config_or_default() -> None:
    """不给联系方式时取配置里的 school.contact；配置取不到则用兜底称谓，不抛异常。"""
    text = build_refusal_text()
    assert text.startswith("知识库中暂无相关资料，建议咨询")
    assert text.endswith("。")


# ==================== 上下文拼装 ====================


def test_build_context_numbers_sources_from_one() -> None:
    context = build_context([hit("第一段内容"), hit("第二段内容", doc_id=2)])

    assert context.startswith("【来源1】")
    assert "【来源2】" in context
    assert "第一段内容" in context and "第二段内容" in context


def test_build_context_empty_hits() -> None:
    assert build_context([]) == ""


# ==================== 引用解析与越界剔除（DR-11）====================


def test_parse_source_numbers_keeps_first_seen_order() -> None:
    text = "结论甲【来源2】，结论乙【来源1】【来源2】。"
    assert parse_source_numbers(text) == [2, 1]


def test_source_pattern_matches_citation() -> None:
    assert SOURCE_PATTERN.findall("见【来源3】") == ["3"]


def test_strip_invalid_sources_removes_out_of_range() -> None:
    """只给了 2 条资料，模型写的【来源3】必须被剔除，绝不伪造成有效来源。"""
    text = "结论甲【来源1】，结论乙【来源3】。"

    cleaned = strip_invalid_sources(text, max_index=2)

    assert "【来源1】" in cleaned
    assert "【来源3】" not in cleaned


def test_strip_invalid_sources_keeps_all_in_range() -> None:
    text = "甲【来源1】乙【来源2】丙【来源3】"
    assert strip_invalid_sources(text, max_index=3) == text
