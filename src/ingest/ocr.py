"""图片与扫描件 OCR（二期 M6）。

设计依据：docs/02-架构设计.md §4.2 ——「图片 OCR 入库」为二期（高风险可提前项），
原设计指定 PaddleOCR，`src/ingest/ocr.py` 的位置也沿用。

**实现偏离（客户 2026-10-04 裁决）**：改用 RapidOCR（ONNX）。
原因：PaddleOCR 的依赖树过大（`paddleocr` + `paddlepaddle` + `paddlex[ocr-core]`
+ `modelscope` + `cryptography` + `opencv-contrib`），且会把 numpy 升到 2.x，
对现有 torch / chromadb / sentence-transformers 有破坏风险；RapidOCR 复用已安装的
`onnxruntime`，只新增 4 个小依赖，模型随包内置、无需联网下载。已登记 docs/09 §12。

引擎是**进程级单例**：初始化要加载检测 / 方向分类 / 识别三个 ONNX 模型（实测约 10 秒），
每次调用都新建不可接受。首次调用时惰性加载，避免不用 OCR 的场景白付这份代价。
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.errors import ParseError

logger = logging.getLogger(__name__)

# 扫描件 PDF 逐页识别时，单页栅格化的缩放比。2.0 约等于 150~200 DPI，
# 实测中文识别置信度 0.98+，再高收益有限、耗时与内存翻倍
PDF_RENDER_SCALE = 2.0

_engine = None


def reset_engine() -> None:
    """丢弃已加载的引擎。仅供测试使用。"""
    global _engine
    _engine = None


def get_engine():
    """惰性加载并复用 OCR 引擎。缺少依赖时给出明确的中文报错。"""
    global _engine
    if _engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover - 依赖缺失属环境问题
            raise ParseError(
                "缺少 rapidocr-onnxruntime 依赖，无法识别图片；请先 pip install -r requirements.txt"
            ) from exc
        logger.info("加载 OCR 引擎（首次约 10 秒）…")
        _engine = RapidOCR()
    return _engine


def _to_lines(result) -> list[str]:
    """把引擎返回值（`[box, text, score]` 列表）转成文本行。"""
    if not result:
        return []
    lines: list[str] = []
    for item in result:
        if not item or len(item) < 2:
            continue
        text = str(item[1]).strip()
        if text:
            lines.append(text)
    return lines


def recognize(source) -> list[str]:
    """识别一张图片，返回按识别顺序排列的文本行。

    Args:
        source: 图片路径、`bytes`，或已读入的数组。

    Returns:
        文本行列表；识别不到内容时返回空列表，不抛异常。
    """
    if isinstance(source, str):
        source = Path(source)
    if isinstance(source, Path) and not source.exists():
        raise ParseError(f"图片文件不存在：{source}")

    try:
        result, _elapsed = get_engine()(source)
    except ParseError:
        raise
    except Exception as exc:
        # 图片损坏、格式不支持等都由引擎抛内部异常；对外统一成清晰的中文报错
        # （docs/02 §11：不把内部堆栈抛给用户）
        raise ParseError(f"图片识别失败（文件可能损坏或不是有效图片）：{exc}") from exc
    return _to_lines(result)


def extract_image_text(path: str | Path) -> str:
    """识别单张图片，返回以换行拼接的文本（空串表示未识别到文字）。"""
    return "\n".join(recognize(path))


def extract_pdf_text(path: str | Path, page_count: int) -> list[tuple[int, str]]:
    """逐页栅格化并识别扫描件 PDF，返回 `[(页码, 文本)]`。

    栅格化用 pypdfium2（纯 wheel，无系统依赖）。单页识别失败只跳过该页并记日志，
    不中断整篇——一份扫描件里个别页识别不出来，不该让整份资料入不了库。
    """
    target = Path(path)
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - 依赖缺失属环境问题
        raise ParseError(
            "缺少 pypdfium2 依赖，无法识别扫描件 PDF；请先 pip install -r requirements.txt"
        ) from exc

    try:
        document = pdfium.PdfDocument(str(target))
    except Exception as exc:
        raise ParseError(f"扫描件 PDF 打开失败：{target.name}（{exc}）") from exc

    pages: list[tuple[int, str]] = []
    limit = min(page_count or len(document), len(document))
    try:
        for index in range(limit):
            page = document[index]
            try:
                image = page.render(scale=PDF_RENDER_SCALE).to_pil()
                text = "\n".join(recognize(image))
            except Exception as exc:  # 单页失败不影响其余页
                logger.warning("扫描件第 %d 页识别失败：%s（%s）", index + 1, target.name, exc)
                continue
            finally:
                # 显式关闭：pypdfium2 的页面/位图不主动释放会在进程退出时刷告警
                page.close()
            if text:
                pages.append((index + 1, text))
    finally:
        document.close()
    return pages
