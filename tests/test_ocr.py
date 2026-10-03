"""OCR 解析单元测试（二期 M6）。

覆盖图片识别、扫描件 PDF 逐页识别的接线，以及 OCR 关闭时的拒绝 / 降级。
OCR 引擎用替身注入：真实引擎首次加载要约 10 秒，且识别结果依赖字体与图像质量，
不适合放进单测。真实识别效果由隔离环境下的端到端冒烟验证。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import Settings
from src.errors import UnsupportedFileType
from src.ingest.loader import load_document
from src.ui.documents import validate_upload

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000100ffff03000006000557bfabd400"
    "00000049454e44ae426082"
)


def make_image(tmp_path: Path, name: str = "通知.png") -> Path:
    target = tmp_path / name
    target.write_bytes(PNG)
    return target


def make_blank_pdf(tmp_path: Path, name: str = "扫描件.pdf") -> Path:
    """无任何文本的 PDF，用于触发「疑似扫描件」分支。"""
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    target = tmp_path / name
    with target.open("wb") as handle:
        writer.write(handle)
    return target


# ==================== 图片 ====================


def test_image_is_recognized(tmp_path: Path, settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.ingest.ocr.extract_image_text", lambda path: "宿舍搬迁需提前三个工作日申请"
    )

    result = load_document(make_image(tmp_path), settings=settings)

    assert not result.is_empty
    assert "三个工作日" in result.sections[0].text


def test_image_without_recognizable_text_warns(tmp_path: Path, settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.ocr.extract_image_text", lambda path: "")

    result = load_document(make_image(tmp_path), settings=settings)

    assert result.is_empty
    assert "未能从图片中识别到文字" in result.warning


def test_image_rejected_when_ocr_disabled(tmp_path: Path, settings: Settings) -> None:
    """OCR 关闭时图片必须被拒绝，不能入库一份空文档。"""
    disabled = settings.model_copy(update={"ocr_enabled": False})

    with pytest.raises(UnsupportedFileType, match="未启用 OCR"):
        load_document(make_image(tmp_path), settings=disabled)


def test_validate_upload_blocks_image_when_ocr_disabled(settings: Settings) -> None:
    disabled = settings.model_copy(update={"ocr_enabled": False})

    assert validate_upload("通知.png", 1024, settings=settings) == ""
    assert "未启用 OCR" in validate_upload("通知.png", 1024, settings=disabled)
    assert validate_upload("通知.png", 1024, settings=settings) == ""


def test_corrupt_image_surfaces_parse_error(tmp_path: Path, monkeypatch) -> None:
    """引擎抛内部异常时要转成面向用户的中文报错，不泄漏堆栈（docs/02 §11）。"""
    from src.errors import ParseError
    from src.ingest import ocr

    def boom(_source):
        raise ValueError("cannot identify image file")

    monkeypatch.setattr(ocr, "get_engine", lambda: boom)

    with pytest.raises(ParseError, match="图片识别失败"):
        ocr.recognize(make_image(tmp_path))


# ==================== 扫描件 PDF ====================


def test_scanned_pdf_falls_back_to_ocr(tmp_path: Path, settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.ingest.ocr.extract_pdf_text", lambda path, page_count: [(1, "扫描件第一页正文")]
    )

    result = load_document(make_blank_pdf(tmp_path), settings=settings)

    assert "扫描件第一页正文" in result.sections[0].text
    assert result.needs_ocr is True
    assert "OCR" in result.warning  # 如实告知是识别结果


def test_scanned_pdf_without_ocr_keeps_warning(tmp_path: Path, settings: Settings) -> None:
    disabled = settings.model_copy(update={"ocr_enabled": False})

    result = load_document(make_blank_pdf(tmp_path), settings=disabled)

    assert result.is_empty
    assert result.needs_ocr is True
    assert "疑似扫描件" in result.warning


def test_scanned_pdf_ocr_yields_nothing_keeps_warning(
    tmp_path: Path, settings: Settings, monkeypatch
) -> None:
    monkeypatch.setattr("src.ingest.ocr.extract_pdf_text", lambda path, page_count: [])

    result = load_document(make_blank_pdf(tmp_path), settings=settings)

    assert result.is_empty
    assert "疑似扫描件" in result.warning
