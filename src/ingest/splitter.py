"""中文两阶段切片。

第一阶段：按结构切分——标题（第X章 / 一、 / 1. / Markdown #）与空行段落，
           标题与其后正文保持在同一块，避免把一条规定的标题和内容切断。
第二阶段：递归兜底——超长块用递归字符切分，分隔符优先取语义边界。

设计依据：docs/02-架构设计.md §4.2
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE

# 递归兜底的分隔符优先级：段落 → 换行 → 句号 → 分号 → 逗号
SEPARATORS = ["\n\n", "\n", "。", "；", "，"]

# 单个结构块的上限：超过则先切成多块再递归，避免超大块拖慢切分
MAX_BLOCK_SIZE = CHUNK_SIZE * 2

# 标题识别：仅匹配短行，避免把正文里带编号的长句误判成标题
MAX_HEADING_LENGTH = 40

_HEADING_PATTERNS = (
    re.compile(r"^#{1,6}\s"),                                  # Markdown 标题
    re.compile(r"^第[一二三四五六七八九十百千零〇\d]+[章节条编部分]"),  # 第二章 / 第3节
    re.compile(r"^[一二三四五六七八九十]+[、.．]"),                    # 一、 / 二.
    re.compile(r"^[（(][一二三四五六七八九十]+[)）]"),                # （一）
    re.compile(r"^\d+[、.．]\s*\S"),                                 # 1. / 2、
    re.compile(r"^[（(]\d+[)）]"),                                   # (1)
)


@dataclass
class Chunk:
    """切片。chunk_index 在文档内连续，用于引用溯源与进度计数。"""

    text: str
    chunk_index: int


def is_heading(line: str) -> bool:
    """判断一行是否为标题。"""
    candidate = (line or "").strip()
    if not candidate or len(candidate) > MAX_HEADING_LENGTH:
        return False
    return any(pattern.match(candidate) for pattern in _HEADING_PATTERNS)


def split_structural(text: str, max_block_size: int = MAX_BLOCK_SIZE) -> list[str]:
    """第一阶段：按标题与空行切成语义块。

    规则：
    - 遇到标题行 → 开启新块（标题留在新块开头，与其后正文同块）
    - 块长超过 max_block_size → 开启新块
    """
    normalized = (text or "").strip()
    if not normalized:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]

    # PDF 抽取常只有单换行、没有空行，此时退化为按行切分
    if len(paragraphs) <= 1:
        paragraphs = [line.strip() for line in normalized.splitlines() if line.strip()]

    blocks: list[str] = []
    buffer: list[str] = []
    size = 0

    for paragraph in paragraphs:
        lines = paragraph.splitlines()
        starts_new_section = is_heading(lines[0]) if lines else False

        if buffer and (starts_new_section or size + len(paragraph) > max_block_size):
            blocks.append("\n\n".join(buffer))
            buffer, size = [], 0

        buffer.append(paragraph)
        size += len(paragraph)

    if buffer:
        blocks.append("\n\n".join(buffer))

    return blocks


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """两阶段切片入口。

    空输入或全空白输入返回空列表，不抛异常（docs/02 §11）。
    """
    normalized = (text or "").strip()
    if not normalized:
        return []

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        # "end"：把分隔符留在前一片末尾。默认的 True 会挂到下一片开头，
        # 使切片以「秩序」这类断句处收尾、下一片以「。」开头，引用展示很别扭
        keep_separator="end",
        length_function=len,
    )

    pieces: list[str] = []
    for block in split_structural(normalized, max_block_size=chunk_size * 2):
        if len(block) <= chunk_size:
            pieces.append(block)
        else:
            pieces.extend(recursive.split_text(block))

    cleaned = [piece.strip() for piece in pieces if piece and piece.strip()]
    return [Chunk(text=piece, chunk_index=index) for index, piece in enumerate(cleaned)]
