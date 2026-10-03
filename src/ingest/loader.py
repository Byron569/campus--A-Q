"""文档解析：PDF / DOCX / TXT / MD → 原始文本段落。

设计依据：docs/02-架构设计.md §4.2
异常约定：docs/02 §11 —— 后缀不支持抛 UnsupportedFileType；解析失败抛 ParseError。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import ALLOWED_SUFFIXES, OCR_CHAR_THRESHOLD_PER_PAGE
from src.errors import ParseError, UnsupportedFileType

logger = logging.getLogger(__name__)


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
    """清洗：去行首尾空白，连续空行压缩为一个。"""
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

    return "\n".join(lines).strip()


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


def _load_pdf(path: Path) -> LoadResult:
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
            result.needs_ocr = True
            result.warning = (
                f"该文件疑似扫描件（平均每页仅 {average:.1f} 个可提取字符），"
                "内容可能未能入库，建议后续启用 OCR。"
            )

    return result


def _load_docx(path: Path) -> LoadResult:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - 依赖缺失属环境问题
        raise ParseError("缺少 python-docx 依赖，无法解析 DOCX") from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ParseError(f"DOCX 解析失败：{path.name}（{exc}）") from exc

    parts: list[str] = [p.text for p in document.paragraphs]

    # 表格也是正文的一部分。校园通知里大量信息（时间表、办理流程）以表格承载，
    # 若只取段落会静默丢内容，故一并抽取（设计文档未明确，已在自审清单登记）。
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    text = clean_text("\n\n".join(part for part in parts if part and part.strip()))
    # python-docx 无法可靠还原分页，page 统一为 None
    sections = [RawSection(text=text)] if text else []
    return LoadResult(sections=sections)


def load_document(path: str | Path) -> LoadResult:
    """解析文档，返回文本段落与告警。

    Raises:
        UnsupportedFileType: 后缀不在白名单（FR-05）。
        ParseError: 文件不存在、损坏、加密或无法解码。
    """
    target = Path(path)
    suffix = target.suffix.lower()

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
        result = _load_pdf(target)
    elif suffix == ".docx":
        result = _load_docx(target)
    else:
        result = _load_plain(target)

    # 空文本告警；若已给出更具体的 OCR 告警则不覆盖
    if result.is_empty and not result.warning:
        result.warning = "未能从文件中提取到文本"

    return result
