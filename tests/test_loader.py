"""文档解析单元测试。

覆盖 docs/03 §6.1 相关用例，以及 docs/02 §11 异常处理策略表中与解析相关的分支：
文件类型不支持 / 文件为空 / 图片型 PDF / 加密 PDF / 文件不存在。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.errors import ParseError, UnsupportedFileType
from src.ingest.loader import clean_text, load_document


# ---------------- 辅助：构造测试文件 ----------------


def make_txt(tmp_path: Path, content: str, name: str = "notice.txt", encoding: str = "utf-8") -> Path:
    target = tmp_path / name
    target.write_text(content, encoding=encoding)
    return target


def make_blank_pdf(tmp_path: Path, pages: int = 1, name: str = "scan.pdf") -> Path:
    """生成无文本的 PDF —— 模拟扫描件。"""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    target = tmp_path / name
    with open(target, "wb") as fh:
        writer.write(fh)
    return target


def make_encrypted_pdf(tmp_path: Path, name: str = "locked.pdf") -> Path:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("secret")
    target = tmp_path / name
    with open(target, "wb") as fh:
        writer.write(fh)
    return target


def make_docx(tmp_path: Path, name: str = "handbook.docx") -> Path:
    import docx

    document = docx.Document()
    document.add_paragraph("第二章 宿舍搬迁")
    document.add_paragraph("学生因转专业等原因需搬迁的，应提前三个工作日提交申请。")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "报到时间"
    table.cell(0, 1).text = "九月一日至九月二日"
    target = tmp_path / name
    document.save(str(target))
    return target


# ---------------- clean_text ----------------


def test_clean_text_trims_and_collapses_blank_lines() -> None:
    raw = "  第一章   \n\n\n\n   第一条 规定。   \n   \n"
    # 中文字符之间的空格也被一并去掉（见下一条用例）
    assert clean_text(raw) == "第一章\n\n第一条规定。"


def test_clean_text_removes_spaces_between_cjk() -> None:
    """真实资料实证：Word 分散对齐与 PDF 转换会在中文字符间塞空格。

    这类空格会破坏 BM25 的 jieba 分词召回，必须清掉。
    """
    raw = "《编 译 原 理》 实 验 指 导 书\n\n前 言\n\n实 验 一　词 法 分 析"
    assert clean_text(raw) == "《编译原理》实验指导书\n\n前言\n\n实验一词法分析"


def test_clean_text_keeps_spaces_between_non_cjk() -> None:
    """英文、数字之间以及中英之间的空格必须保留，否则会把词粘在一起。"""
    raw = "安装 Python 3.12 与 Node.js 20，参见 Appendix A"
    assert clean_text(raw) == raw


def test_clean_text_on_empty_input() -> None:
    assert clean_text("") == ""
    assert clean_text("   \n\n  ") == ""


# ---------------- TXT / MD ----------------


def test_load_txt(tmp_path: Path) -> None:
    path = make_txt(tmp_path, "第一章 总则\n\n第一条 为规范宿舍管理，制定本办法。")
    result = load_document(path)

    assert not result.is_empty
    assert result.needs_ocr is False
    assert result.warning == ""
    assert len(result.sections) == 1
    assert result.sections[0].page is None
    assert "为规范宿舍管理" in result.sections[0].text


def test_load_markdown(tmp_path: Path) -> None:
    path = make_txt(tmp_path, "# 新生报到须知\n\n报到时间：九月一日。", name="guide.md")
    result = load_document(path)

    assert not result.is_empty
    assert "新生报到须知" in result.sections[0].text


def test_load_gbk_encoded_txt(tmp_path: Path) -> None:
    """中文校园文档常见 GBK 编码，应能兜底读取。"""
    path = make_txt(tmp_path, "校园卡遗失后应及时挂失。", name="gbk.txt", encoding="gbk")
    result = load_document(path)

    assert "校园卡遗失后应及时挂失" in result.sections[0].text


# ---------------- 空文件 ----------------


def test_empty_txt_reports_warning(tmp_path: Path) -> None:
    path = make_txt(tmp_path, "   \n\n   ", name="empty.txt")
    result = load_document(path)

    assert result.is_empty
    assert result.warning == "未能从文件中提取到文本"


# ---------------- 图片型 PDF ----------------


def test_blank_pdf_is_flagged_needs_ocr(tmp_path: Path) -> None:
    result = load_document(make_blank_pdf(tmp_path, pages=3))

    assert result.needs_ocr is True
    assert "扫描件" in result.warning
    # 空文本的通用告警不应覆盖更具体的 OCR 告警
    assert "未能从文件中提取到文本" not in result.warning


# ---------------- 加密 PDF ----------------


def test_encrypted_pdf_raises_parse_error(tmp_path: Path) -> None:
    with pytest.raises(ParseError, match="已加密"):
        load_document(make_encrypted_pdf(tmp_path))


# ---------------- DOCX ----------------


def test_load_docx_paragraphs_and_tables(tmp_path: Path) -> None:
    result = load_document(make_docx(tmp_path))

    assert not result.is_empty
    text = result.sections[0].text
    # 标题里的中文空格已被清洗掉（原文档写作「第二章 宿舍搬迁」）
    assert "第二章宿舍搬迁" in text
    assert "提前三个工作日提交申请" in text
    # 表格内容不能丢：校园通知大量信息以表格承载
    assert "报到时间" in text
    assert "九月一日至九月二日" in text


def make_merged_cell_docx(tmp_path: Path, name: str = "merged.docx") -> Path:
    """横向合并的单元格：python-docx 会把同一个 cell 在 row.cells 里返回多次。"""
    import docx

    document = docx.Document()
    table = document.add_table(rows=2, cols=3)
    table.cell(0, 0).text = "课程名称"
    table.cell(0, 1).merge(table.cell(0, 2))
    table.cell(0, 1).text = "《编译原理》"
    table.cell(1, 0).text = "学分"
    table.cell(1, 1).text = "3"
    table.cell(1, 2).text = "必修"
    target = tmp_path / name
    document.save(str(target))
    return target


def test_load_docx_deduplicates_merged_table_cells(tmp_path: Path) -> None:
    """合并单元格的重复值必须去重，否则一个值会被刷成「值 | 值 | 值」。"""
    text = load_document(make_merged_cell_docx(tmp_path)).sections[0].text

    assert "课程名称 | 《编译原理》" in text
    assert "《编译原理》 | 《编译原理》" not in text
    # 正常行不受影响
    assert "学分 | 3 | 必修" in text


def make_ordered_docx(tmp_path: Path, name: str = "ordered.docx") -> Path:
    """标题 → 表格 → 下一节，用来验证表格没有被搬到文末。"""
    import docx

    document = docx.Document()
    document.add_paragraph("二、审批权限")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "请假天数"
    table.cell(0, 1).text = "审批人"
    table.cell(1, 0).text = "超过 3 天"
    table.cell(1, 1).text = "学院分管领导审批"
    document.add_paragraph("三、销假")
    document.add_paragraph("请假期满后应于 1 个工作日内销假。")
    target = tmp_path / name
    document.save(str(target))
    return target


def test_load_docx_keeps_tables_in_document_order(tmp_path: Path) -> None:
    """表格必须留在它所属的标题下，不能被搬到文末粘到最后一节（FB-3.5 暴露）。

    搬走后「二、审批权限」下空无一物，检索命中标题却取不到表格内容。
    """
    text = load_document(make_ordered_docx(tmp_path)).sections[0].text

    assert text.index("二、审批权限") < text.index("请假天数")
    assert text.index("请假天数") < text.index("三、销假")
    assert "超过 3 天 | 学院分管领导审批" in text


# ---------------- 异常分支 ----------------


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    target = tmp_path / "setup.exe"
    target.write_bytes(b"fake")

    with pytest.raises(UnsupportedFileType) as excinfo:
        load_document(target)
    assert ".exe" in str(excinfo.value)


def test_uppercase_suffix_is_accepted(tmp_path: Path) -> None:
    target = tmp_path / "NOTICE.TXT"
    target.write_text("校园卡挂失流程。", encoding="utf-8")

    result = load_document(target)
    assert "校园卡挂失流程" in result.sections[0].text


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ParseError, match="不存在"):
        load_document(tmp_path / "not-exist.txt")


def test_directory_path_raises(tmp_path: Path) -> None:
    folder = tmp_path / "docs.txt"
    folder.mkdir()

    with pytest.raises(ParseError, match="不是文件"):
        load_document(folder)


def test_corrupted_pdf_raises_parse_error(tmp_path: Path) -> None:
    target = tmp_path / "broken.pdf"
    target.write_bytes(b"this is definitely not a pdf")

    with pytest.raises(ParseError, match="PDF 解析失败"):
        load_document(target)
