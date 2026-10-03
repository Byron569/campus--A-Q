"""混合检索器：向量路 + BM25 路，按配置权重融合。

设计依据：
- docs/02-架构设计.md §4.3（混合检索；**相似度阈值只作用于向量路**；「命中」判定）
- docs/06-接口文档.md §1.6（`build_retriever(user_id, category=None)`）、§3.1（`SearchHit`）
- docs/09-迭代开发计划.md §7 C-05（阈值只作用于向量路；`matched_by` 正确；分类过滤生效）

两条路的分工与约束：

| 路 | 数据来源 | 阈值 | 命中含义 |
| --- | --- | --- | --- |
| 向量 | `VectorStore.search()`（唯一入口，强制传 `user_id`） | **生效**（`RETRIEVE_SCORE_THRESHOLD`） | 余弦相似度 ≥ 阈值 |
| BM25 | `VectorStore.list_chunks()`（同一套权限过滤） | **不生效**（无上界、量纲不同） | query 分词后有关键词命中 |

只要有一路给出候选即为「命中」；两路皆空才拒答（FR-10）。该判定由上层
`chain.py` 依据本模块返回的列表是否为空来完成。

**与设计的一处偏离（已在自审清单登记）**：docs/02 §1.1/§4.3 与 docs/06 §1.6 提到用
LangChain 的 `EnsembleRetriever` 融合。实测其 `reciprocal_rank_fusion` 以
`(page_content, metadata)` 为去重键，而两条路给同一切片打的 `matched_by` / `score`
不同，会被判为两个文档而重复出现；且 RRF 只输出 Documents，拿不到 DR-15 要求的
`matched_by` 与仅向量路可得的余弦分数。故这里自行实现加权 RRF，返回 `SearchHit`，
语义与设计一致。`build_retriever` 的返回类型随之由 `EnsembleRetriever` 变为本模块的
`HybridRetriever`（提供 `search` / `invoke`）。
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache

from config.settings import Settings, get_settings
from src.store.chroma import SearchHit, VectorStore

logger = logging.getLogger(__name__)

# RRF 平滑常数，沿用信息检索中的常用取值 60
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """jieba 分词，去掉纯空白 token。"""
    import jieba

    return [token for token in jieba.lcut(text or "") if token.strip()]


class HybridRetriever:
    """向量 + BM25 融合检索器（docs/02 §4.3）。"""

    def __init__(
        self,
        *,
        user_id: int | None,
        category: str | None = None,
        store: VectorStore,
        settings: Settings,
    ) -> None:
        self.user_id = user_id
        self.category = category
        self.store = store
        self.settings = settings

    # ---------------- 两条路 ----------------

    def _vector_route(self, query: str) -> list[SearchHit]:
        """向量路：阈值在此生效（DR-01）。"""
        return self.store.search(
            query,
            user_id=self.user_id,
            category=self.category,
            k=self.settings.retrieve_top_k,
            score_threshold=self.settings.retrieve_score_threshold,
        )

    def _bm25_route(self, query: str) -> list[SearchHit]:
        """关键词路：**不套用相似度阈值**，只要有词命中即可作为候选。"""
        corpus = self.store.list_chunks(user_id=self.user_id, category=self.category)
        if not corpus:
            return []

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        from rank_bm25 import BM25Okapi

        tokenized = [_tokenize(hit.text) for hit in corpus]
        # 语料全是标点等无法分词的文本时，BM25Okapi 的平均长度会取到 0 而在打分时除零
        if not any(tokenized):
            return []

        scores = BM25Okapi(tokenized).get_scores(list(query_tokens))

        # 命中判定用「分词集合是否有交集」，而不是「BM25 分数 > 0」：
        # 某词出现在过半文档时 BM25 的 idf 可能为负，按分数过滤会漏掉真正命中的切片。
        candidates = [
            (float(score), hit)
            for score, tokens, hit in zip(scores, tokenized, corpus)
            if query_tokens.intersection(tokens)
        ]
        candidates.sort(key=lambda pair: (-pair[0], pair[1].doc_id, pair[1].chunk_index))

        return [
            replace(hit, score=None, matched_by="bm25")
            for _, hit in candidates[: self.settings.retrieve_top_k]
        ]

    # ---------------- 融合 ----------------

    @staticmethod
    def _key(hit: SearchHit) -> tuple[int, int]:
        """同一份文档内的切片序号唯一，可作为切片身份。"""
        return (hit.doc_id, hit.chunk_index)

    def _fuse(
        self, vector_hits: list[SearchHit], bm25_hits: list[SearchHit]
    ) -> list[SearchHit]:
        """加权 RRF 融合，并标出每条候选来自哪一路（DR-15）。"""
        fused: dict[tuple[int, int], float] = {}
        routes: dict[tuple[int, int], set[str]] = {}
        chosen: dict[tuple[int, int], SearchHit] = {}

        def accumulate(hits: list[SearchHit], weight: float, route: str) -> None:
            for rank, hit in enumerate(hits):
                key = self._key(hit)
                fused[key] = fused.get(key, 0.0) + weight / (RRF_K + rank + 1)
                routes.setdefault(key, set()).add(route)
                # 两条路都有时保留向量路的命中——只有它带余弦分数
                if key not in chosen or route == "vector":
                    chosen[key] = hit

        accumulate(vector_hits, self.settings.vector_weight, "vector")
        accumulate(bm25_hits, self.settings.bm25_weight, "bm25")

        ordered = sorted(fused, key=lambda key: (-fused[key], key))
        results: list[SearchHit] = []
        for key in ordered[: self.settings.retrieve_top_k]:
            tags = routes[key]
            matched_by = "both" if len(tags) == 2 else next(iter(tags))
            results.append(replace(chosen[key], matched_by=matched_by))
        return results

    # ---------------- 对外 ----------------

    def search(self, query: str) -> list[SearchHit]:
        """返回融合后的 Top-K 命中；空列表表示两路皆未命中（触发拒答）。"""
        if not query or not query.strip():
            return []
        return self._fuse(self._vector_route(query), self._bm25_route(query))

    def invoke(self, query: str) -> list[SearchHit]:
        """`search` 的别名，便于上层按 LangChain 习惯调用。"""
        return self.search(query)


@lru_cache(maxsize=1)
def _default_store() -> VectorStore:
    """进程内复用的向量库（本地 BGE 加载一次较慢，不可每次提问都新建）。"""
    return VectorStore(get_settings())


def build_retriever(
    user_id: int | None,
    category: str | None = None,
    *,
    store: VectorStore | None = None,
    settings: Settings | None = None,
) -> HybridRetriever:
    """构造混合检索器。

    Args:
        user_id: 用户身份，必传。None 表示无身份，只检索公共文档（比带身份更严格）。
        category: 可选的分类限定（FR-14）。
        store: 向量库实例，默认取进程内单例；测试可注入假向量库。
        settings: 配置，默认取全局单例。
    """
    resolved = settings or get_settings()
    return HybridRetriever(
        user_id=user_id,
        category=category,
        store=store or _default_store(),
        settings=resolved,
    )
