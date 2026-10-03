"""文档接入层：解析、切片、入库流水线。

设计依据：docs/02-架构设计.md §4.2、docs/06-接口文档.md §1.3
"""

from src.ingest.loader import LoadResult, RawSection, clean_text, load_document
from src.ingest.splitter import Chunk, is_heading, split_structural, split_text

__all__ = [
    "LoadResult",
    "RawSection",
    "clean_text",
    "load_document",
    "Chunk",
    "is_heading",
    "split_structural",
    "split_text",
]
