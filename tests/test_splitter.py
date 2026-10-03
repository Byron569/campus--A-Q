"""中文切片单元测试。

覆盖 docs/03 §6.1 的 TC-U01 ~ TC-U04。
"""

from __future__ import annotations

from src.ingest.splitter import Chunk, is_heading, split_structural, split_text


# ---------------- is_heading ----------------


def test_recognizes_chinese_chapter_headings() -> None:
    assert is_heading("第二章 宿舍管理")
    assert is_heading("第一条 申请流程")
    assert is_heading("一、报到时间")
    assert is_heading("（三）缴费方式")
    assert is_heading("3. 校园卡办理")
    assert is_heading("## 搬迁申请")


def test_does_not_treat_long_sentence_as_heading() -> None:
    long_sentence = "1. " + "这是一段很长的正文内容，" * 5
    assert not is_heading(long_sentence)


def test_does_not_treat_plain_text_as_heading() -> None:
    assert not is_heading("学生应提前向辅导员提交申请。")
    assert not is_heading("")
    assert not is_heading("   ")


# ---------------- TC-U01：标题与其后正文同块 ----------------


def test_heading_stays_with_following_body() -> None:
    text = "\n".join(
        [
            "第二章 宿舍搬迁",
            "",
            "学生因转专业等原因需搬迁的，应提前三个工作日提交申请。",
            "",
            "第三章 校园卡",
            "",
            "校园卡遗失后应及时挂失。",
        ]
    )

    blocks = split_structural(text)

    assert len(blocks) == 2

    first = blocks[0]
    assert "第二章 宿舍搬迁" in first
    assert "应提前三个工作日提交申请" in first

    second = blocks[1]
    assert "第三章 校园卡" in second
    assert "应及时挂失" in second


def test_chunks_do_not_split_heading_from_body() -> None:
    text = "第二章 宿舍搬迁\n\n学生因转专业等原因需搬迁的，应提前三个工作日提交申请。"
    chunks = split_text(text)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("第二章 宿舍搬迁")
    assert "提前三个工作日" in chunks[0].text


# ---------------- TC-U02：超长文本被递归切分且不超上限 ----------------


def test_long_text_is_split_within_chunk_size() -> None:
    # 单段超长文本，无空行、无标题
    text = "校园卡遗失后应及时挂失并补办。" * 200
    assert len(text) > 2000

    chunks = split_text(text, chunk_size=500, chunk_overlap=80)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 500


def test_long_text_prefers_sentence_boundaries() -> None:
    text = "。" .join(f"第{i}条规定学生应当遵守校园管理秩序" for i in range(120)) + "。"

    chunks = split_text(text, chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    # 绝大多数切片应以句号收尾，说明优先在句子边界切分
    ending_with_period = sum(1 for c in chunks if c.text.endswith("。"))
    assert ending_with_period >= len(chunks) - 1


def test_chunk_index_is_contiguous() -> None:
    text = "宿舍搬迁需要提前申请。" * 300
    chunks = split_text(text, chunk_size=400, chunk_overlap=60)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ---------------- TC-U03：空输入 ----------------


def test_empty_text_returns_empty_list() -> None:
    assert split_text("") == []
    assert split_text("   \n\n  \t ") == []
    assert split_structural("") == []


# ---------------- TC-U04：连续短段落完整保留 ----------------


def test_short_paragraphs_are_kept_intact() -> None:
    paragraphs = [
        "报到时间：九月一日至九月二日。",
        "报到地点：学校体育馆。",
        "所需材料：录取通知书、身份证。",
    ]
    text = "\n\n".join(paragraphs)

    chunks = split_text(text)

    assert len(chunks) == 1
    for paragraph in paragraphs:
        assert paragraph in chunks[0].text


def test_no_chunk_is_blank() -> None:
    text = "第一条规定。\n\n\n\n第二条规定。\n\n   \n\n第三条规定。"
    chunks = split_text(text)

    assert chunks
    for chunk in chunks:
        assert chunk.text.strip()


# ---------------- PDF 场景：无空行、按单换行 ----------------


def test_single_newline_text_falls_back_to_line_split() -> None:
    """PDF 抽取常只有单换行，此时应按行切分而非整篇当成一段。"""
    text = "第一章 总则\n第一条 为规范管理\n第二条 适用范围\n第二章 细则\n第三条 具体规定"
    blocks = split_structural(text)

    assert len(blocks) >= 2
    assert any("第一章" in b for b in blocks)


def test_chunk_dataclass_shape() -> None:
    chunks = split_text("这是一段测试文本。")
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].chunk_index == 0
