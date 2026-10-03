"""向量库封装 —— 全项目唯一的向量访问入口。

设计依据：docs/02-架构设计.md §4.3 / §4.8 / §5.2、docs/06-接口文档.md §1.4

安全红线（FR-31）：
- 所有向量查询必须经由本模块，上层禁止直接操作 Chroma
- `user_id` 是**必传关键字参数**，漏传在 Python 层即 TypeError
- `user_id=None` 表示「无登录身份」（如命令行工具早期验证），只检索公共文档，
  比带身份更严格，**绝不会退化成无过滤**
- 任何情况下过滤条件都不为空
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document as LCDocument

from config.settings import (
    CHROMA_DISTANCE,
    COLLECTION_NAME,
    PUBLIC_USER_ID,
    Settings,
    get_settings,
)
from src.errors import PermissionFilterError
from src.providers.embedding import get_embeddings

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """一条检索命中。

    score 为余弦相似度（0~1，越大越相关）。仅 BM25 命中、向量路未命中的切片
    没有相似度可言，此时 score 为 None，由 matched_by 说明来源（DR-15）。
    """

    text: str
    score: float | None
    filename: str
    doc_id: int
    chunk_index: int
    category: str
    matched_by: str = "vector"


class VectorStore:
    """Chroma 封装。构造时会加载 Embedding 模型（本地 BGE 首次加载较慢）。"""

    def __init__(self, settings: Settings | None = None, embeddings: Any | None = None) -> None:
        self.settings = settings or get_settings()

        # embeddings 允许注入，便于测试用确定性向量验证过滤逻辑，避免下载模型
        embedding_function = embeddings or get_embeddings(self.settings)

        from langchain_chroma import Chroma

        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            persist_directory=str(self.settings.chroma_path),
            collection_metadata={"hnsw:space": CHROMA_DISTANCE},
        )

    # ---------------- 过滤条件（安全红线核心）----------------

    def _build_filter(self, user_id: int | None, category: str | None) -> dict[str, Any]:
        """构造 Chroma 过滤条件。

        公共文档在元数据中 user_id 固定为 -1、is_public 为 1；
        个人资料 user_id 为本人、is_public 为 0。
        """
        if user_id is None:
            # 无身份 → 只给公共文档
            base: dict[str, Any] = {"is_public": {"$eq": 1}}
        else:
            base = {
                "$or": [
                    {"is_public": {"$eq": 1}},
                    {"user_id": {"$eq": int(user_id)}},
                ]
            }

        if category:
            combined: dict[str, Any] = {"$and": [base, {"category": {"$eq": category}}]}
            return combined

        # 兜底自检：过滤条件绝不能为空，否则等于全量返回
        if not base:  # pragma: no cover - 防御性断言
            raise PermissionFilterError("检索过滤条件为空，拒绝执行查询")
        return base

    # ---------------- 写入 ----------------

    def add_chunks(self, chunks: list[tuple[str, dict[str, Any]]]) -> int:
        """批量写入切片。chunks 为 (文本, 元数据) 列表，返回写入条数。"""
        documents = [
            LCDocument(page_content=text, metadata=metadata)
            for text, metadata in chunks
            if text and text.strip()
        ]
        if not documents:
            return 0

        self._store.add_documents(documents)
        return len(documents)

    # ---------------- 检索 ----------------

    def search(
        self,
        query: str,
        *,
        user_id: int | None,
        category: str | None = None,
        k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        """混合检索的向量路。

        相似度阈值**只作用于本方法**（DR-01）：BM25 分数无上界、量纲不同，
        套同一个阈值会得出错误结果。

        Args:
            query: 查询文本。
            user_id: 用户身份。必传；None 表示无身份（只检索公共文档）。
            category: 可选的分类限定。
            k: 返回条数，默认取配置。
            score_threshold: 相似度阈值，默认取配置。
        """
        if not query or not query.strip():
            return []

        top_k = k if k is not None else self.settings.retrieve_top_k
        threshold = (
            self.settings.retrieve_score_threshold
            if score_threshold is None
            else score_threshold
        )
        where = self._build_filter(user_id, category)

        try:
            pairs = self._store.similarity_search_with_score(query, k=top_k, filter=where)
        except Exception as exc:
            logger.error("向量检索失败：%s", exc)
            raise

        hits: list[SearchHit] = []
        for document, distance in pairs:
            # 集合以 cosine 建、向量已归一化，故「1 - 距离」即余弦相似度
            similarity = 1.0 - float(distance)
            if similarity < threshold:
                continue

            metadata = document.metadata or {}
            hits.append(
                SearchHit(
                    text=document.page_content,
                    score=round(similarity, 4),
                    filename=str(metadata.get("filename", "")),
                    doc_id=int(metadata.get("doc_id", -1)),
                    chunk_index=int(metadata.get("chunk_index", -1)),
                    category=str(metadata.get("category", "")),
                    matched_by="vector",
                )
            )

        return hits

    def list_chunks(
        self,
        *,
        user_id: int | None,
        category: str | None = None,
    ) -> list[SearchHit]:
        """列出该身份可访问的**全部**切片，供 BM25 路构建关键词索引（docs/02 §4.3）。

        与 `search` 走同一套权限过滤（FR-31），`user_id` 同样必传：这里没有相似度
        可言，故 `score=None`；不做阈值过滤（阈值只作用于向量路，DR-01）。
        上层不得绕过本方法去读 Chroma。
        """
        where = self._build_filter(user_id, category)

        try:
            data = self._store.get(where=where)
        except Exception as exc:
            logger.error("读取检索语料失败：%s", exc)
            raise

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        hits: list[SearchHit] = []
        for text, metadata in zip(documents, metadatas):
            info = metadata or {}
            hits.append(
                SearchHit(
                    text=text,
                    score=None,
                    filename=str(info.get("filename", "")),
                    doc_id=int(info.get("doc_id", -1)),
                    chunk_index=int(info.get("chunk_index", -1)),
                    category=str(info.get("category", "")),
                )
            )
        # 固定顺序，保证 BM25 打分与融合结果可复现
        hits.sort(key=lambda hit: (hit.doc_id, hit.chunk_index))
        return hits

    # ---------------- 删除 ----------------

    def delete_by_doc_id(self, doc_id: int) -> int:
        """按文档删除其全部向量。删除操作的唯一入口之一。"""
        return self._delete_where({"doc_id": {"$eq": int(doc_id)}})

    def delete_by_user_id(self, user_id: int) -> int:
        """删除某用户的**个人**向量。公共文档（user_id=-1）不受影响。"""
        return self._delete_where(
            {"$and": [{"user_id": {"$eq": int(user_id)}}, {"is_public": {"$eq": 0}}]}
        )

    def _delete_where(self, where: dict[str, Any]) -> int:
        data = self._store.get(where=where)
        ids = data.get("ids") or []
        if not ids:
            return 0
        self._store.delete(ids=ids)
        return len(ids)

    # ---------------- 统计 ----------------

    def count(self) -> int:
        """集合内的切片总数。用于验证与排查。"""
        return int(self._store._collection.count())  # noqa: SLF001 - langchain 封装未暴露计数


def build_metadata(
    *,
    doc_id: int,
    user_id: int | None,
    is_public: bool,
    category: str,
    filename: str,
    chunk_index: int,
    source_type: str = "text",
) -> dict[str, Any]:
    """构造切片元数据。字段约定见 docs/02 §5.2。"""
    return {
        "doc_id": int(doc_id),
        "user_id": PUBLIC_USER_ID if is_public or user_id is None else int(user_id),
        "is_public": 1 if is_public else 0,
        "category": category or "",
        "filename": filename,
        "chunk_index": int(chunk_index),
        "source_type": source_type,
    }
