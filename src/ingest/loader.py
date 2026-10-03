"""文档解析：PDF / DOCX / TXT / MD / 图片（OCR）→ 原始文本段落。

设计依据：docs/02-架构设计.md §4.2
异常约定：docs/02 §11 —— 后缀不支持抛 UnsupportedFileType；解析失败抛 ParseError。

OCR（二期 M6）：图片直接识别；扫描件 PDF（pypdf 抽不出文字）逐页栅格化后识别。
识别实现见 `src/ingest/ocr.py`。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import (
    ALLOWED_SUFFIXES,
    IMAGE_SUFFIXES,
    OCR_CHAR_THRESHOLD_PER_PAGE,
    Settings,
    get_settings,
)
from src.errors import ParseError, UnsupportedFileType

logger = logging.getLogger(__name__)

# 中文（含中文标点）之间不该有空格，但 Word 的分散对齐、PDF 转换都会塞进来。
# 中文本来就不用空格分词，留着会破坏 BM25 的 jieba 分词召回，切片文字也很难看。
# 只清理「两侧都是中日韩字符或中文标点」的空白，因此英文单词之间、数字之间、
# 以及中英之间的空格（如 "Python 语言"、"第 3 章"）都不受影响。
_CJK = r"[\u3000-\u303f\u4e00-\u9fff\uff01-\uff60]"
_CJK_GAP = re.compile(rf"(?<={_CJK})[ \t\u00a0\u3000]+(?={_CJK})")


@dataclass
class RawSection:
    """一段原始文本。page 为 PDF 页码；DOCX / TXT / MD 无页概念，为 None。"""

    text: str
    page: int | None = None


@dataclass
class LoadResult:
    """解析结果。

    needs_ocr：疑似扫描件。docs/02 §4.2 的前置探测器——若大量文件命中此标记，
    说明需要把 PaddleOCR 从二期提前。
    """

    sections: list[RawSection] = field(default_factory=list)
    needs_ocr: bool = False
    warning: str = ""

    @property
    def text_length(self) -> int:
        return sum(len(section.text) for section in self.sections)

    @property
    def is_empty(self) -> bool:
        return self.text_length == 0


def clean_text(text: str) -> str:
    """清洗：去行首尾空白、连续空行压缩为一个、去掉中文字符之间的空白。"""
    lines: list[str] = []
    blank_run = 0

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        lines.append(line)

    return _CJK_GAP.sub("", "\n".join(lines).strip())


def _read_text_file(path: Path) -> str:
    """读取纯文本文件。中文校园文档常见 GBK 编码，做一次兜底。"""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="gbk")
        except UnicodeDecodeError as exc:
            raise ParseError(
                f"文件编码无法识别（既不是 UTF-8 也不是 GBK）：{path.name}"
            ) from exc
    except OSError as exc:
        raise ParseError(f"文件读取失败：{path.name}（{exc}）") from exc


def _load_plain(path: Path) -> LoadResult:
    text = clean_text(_read_text_file(path))
    sections = [RawSection(text=text)] if text else []
    return LoadResult(sections=sections)


def _load_pdf(path: Path, *, settings: Settings) -> LoadResult:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失属环境问题
        raise ParseError("缺少 pypdf 依赖，无法解析 PDF") from exc

    try:
        reader = PdfReader(str(path))

        if reader.is_encrypted:
            # 部分校园 PDF 只设了权限口令，空密码可解
            try:
                status = reader.decrypt("")
            except Exception:
                status = 0
            if not status:
                raise ParseError(f"PDF 已加密，无法解析：{path.name}")

        sections: list[RawSection] = []
        for page_no, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception as exc:
                logger.warning("PDF 第 %d 页文本抽取失败：%s（%s）", page_no, path.name, exc)
                raw = ""
            text = clean_text(raw)
            if text:
                sections.append(RawSection(text=text, page=page_no))

        page_count = len(reader.pages)
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(f"PDF 解析失败：{path.name}（{exc}）") from exc

    result = LoadResult(sections=sections)

    # 图片型 PDF 检测：平均每页可提取字符数过低
    if page_count:
        average = result.text_length / page_count
        if average < OCR_CHAR_THRESHOLD_PER_PAGE:
            ocr_sections = _ocr_pdf(path, page_count, settings=settings)
            if ocr_sections:
                # OCR 成功：内容已入库，但要如实告诉用户「这是识别结果，可能有误差」
                return LoadResult(
                    sections=ocr_sections,
                    needs_ocr=True,
                    warning="该文件为扫描件，已用 OCR 识别文字，可能存在识别误差。",
                )
            result.needs_ocr = True
            result.warning = (
                f"该文件疑似扫描件（平均每页仅 {average:.1f} 个可提取字符），"
                "未能识别出文字内容，建议改用清晰的原文件。"
            )

    return result


def _ocr_pdf(path: Path, page_count: int, *, settings: Settings) -> list[RawSection]:
    """扫描件 PDF 逐页 OCR。OCR 关闭、或整篇都没识别出文字时返回空列表。"""
    if not settings.ocr_enabled:
        logger.info("OCR 已关闭，跳过扫描件识别：%s", path.name)
        return []

    from src.ingest.ocr import extract_pdf_text

    sections: list[RawSection] = []
    for page_no, text in extract_pdf_text(path, page_count):
        cleaned = clean_text(text)
        if cleaned:
            sections.append(RawSection(text=cleaned, page=page_no))
    return sections


def _load_image(path: Path, *, settings: Settings) -> LoadResult:
    """图片文件 OCR。识别不到文字时给出明确告警，而不是静默入库空文档。"""
    if not settings.ocr_enabled:
        raise UnsupportedFileType(f"未启用 OCR，暂不支持图片文件：{path.name}")

    from src.ingest.ocr import extract_image_text

    text = clean_text(extract_image_text(path))
    if not text:
        return LoadResult(warning="未能从图片中识别到文字，请确认图片清晰、包含文字。")
    return LoadResult(sections=[RawSection(text=text)])


def _docx_table_text(table) -> str:
    """把一个表格转成文本，每行以「 | 」拼接。"""
    rows: list[str] = []
    for row in table.rows:
        # 横向合并的单元格会被 python-docx 重复返回（同一 cell 出现多次），
        # 不做去重会把一个值刷成 "值 | 值 | 值"，切片又脏又浪费 token
        cells: list[str] = []
        for cell in row.cells:
            value = cell.text.strip()
            if value and (not cells or cells[-1] != value):
                cells.append(value)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _load_docx(path: Path) -> LoadResult:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - 依赖缺失属环境问题
        raise ParseError("缺少 python-docx 依赖，无法解析 DOCX") from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ParseError(f"DOCX 解析失败：{path.name}（{exc}）") from exc

    # 表格也是正文的一部分。校园通知里大量信息（时间表、办理流程、审批权限）
    # 以表格承载，若只取段落会静默丢内容，故一并抽取（设计文档未明确，已登记）。
    #
    # 必须按 body 子元素顺序还原：python-docx 把 paragraphs 与 tables 分成两个列表，
    # 「先全部段落、再全部表格」会把表格搬到文末、粘到最后一节，标题（如「二、审批权限」）
    # 下空无一物，检索命中标题却取不到表格内容（FB-3.5 评测实测：q19 因此答不出）。
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parts: list[str] = []
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            text = Paragraph(child, document).text
            if text and text.strip():
                parts.append(text)
        elif child.tag == qn("w:tbl"):
            table_text = _docx_table_text(Table(child, document))
            if table_text:
                parts.append(table_text)

    text = clean_text("\n\n".join(parts))
    # python-docx 无法可靠还原分页，page 统一为 None
    sections = [RawSection(text=text)] if text else []
    return LoadResult(sections=sections)


def load_document(path: str | Path, *, settings: Settings | None = None) -> LoadResult:
    """解析文档，返回文本段落与告警。

    Raises:
        UnsupportedFileType: 后缀不在白名单（FR-05），或图片遇上 OCR 被关闭。
        ParseError: 文件不存在、损坏、加密或无法解码。
    """
    target = Path(path)
    suffix = target.suffix.lower()
    cfg = settings or get_settings()

    if suffix not in ALLOWED_SUFFIXES:
        supported = "、".join(sorted(ALLOWED_SUFFIXES))
        raise UnsupportedFileType(
            f"不支持的文件类型：{suffix or target.name}（支持：{supported}）"
        )

    if not target.exists():
        raise ParseError(f"文件不存在：{target}")
    if not target.is_file():
        raise ParseError(f"不是文件：{target}")

    if suffix == ".pdf":
        result = _load_pdf(target, settings=cfg)
    elif suffix == ".docx":
        result = _load_docx(target)
    elif suffix in IMAGE_SUFFIXES:
        result = _load_image(target, settings=cfg)
    else:
        result = _load_plain(target)

    # 空文本告警；若已给出更具体的 OCR 告警则不覆盖
    if result.is_empty and not result.warning:
        result.warning = "未能从文件中提取到文本"

    return result
