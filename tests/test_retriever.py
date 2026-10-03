"""混合检索器单元测试（M2-02）。

覆盖 docs/09 §7 C-05：**相似度阈值只作用于向量路**、`matched_by` 正确、分类过滤生效；
以及 docs/03 §6.1 TC-U10（分类过滤）与 FR-31 权限隔离在检索层的落地。
"""

from __future__ import annotations

from config.settings import Settings
from src.rag.retriever import build_retriever
from src.store.chroma import VectorStore, build_metadata


def add(
    store: VectorStore,
    text: str,
    *,
    doc_id: int,
    user_id: int | None = None,
    is_public: bool = False,
    filename: str = "doc.pdf",
    chunk_index: int = 0,
    category: str = "freshman",
) -> None:
    store.add_chunks(
        [
            (
                text,
                build_metadata(
                    doc_id=doc_id, user_id=user_id, is_public=is_public,
                    category=category, filename=filename, chunk_index=chunk_index,
                ),
            )
        ]
    )


def make(store: VectorStore, settings: Settings, *, user_id=None, category=None):
    return build_retriever(user_id, category, store=store, settings=settings)


# ==================== 融合与 matched_by（DR-15）====================


def test_hit_from_both_routes_is_tagged_both(store: VectorStore, settings: Settings) -> None:
    text = "宿舍搬迁需要提前三个工作日向辅导员提交申请。"
    add(store, text, doc_id=1, is_public=True, filename="学生手册.pdf", chunk_index=3)

    hits = make(store, settings, user_id=None).search(text)

    assert hits
    assert hits[0].doc_id == 1
    assert hits[0].chunk_index == 3
    assert hits[0].matched_by == "both"
    assert hits[0].score is not None


def test_same_chunk_is_not_duplicated_across_routes(
    store: VectorStore, settings: Settings
) -> None:
    """同一切片被两路命中时只出现一次，不因元数据不同而重复（自审清单登记的偏离点）。"""
    text = "校园卡遗失应及时到一卡通中心挂失补办。"
    add(store, text, doc_id=1, is_public=True)

    hits = make(store, settings, user_id=None).search(text)

    assert len(hits) == 1


# ==================== 阈值只作用于向量路（C-05 / DR-01）====================


def test_threshold_only_applies_to_vector_route(
    store: VectorStore, settings: Settings
) -> None:
    """阈值拉到极高清空向量路后，BM25 路仍应给出候选——证明阈值没有误伤关键词路。"""
    add(
        store,
        "宿舍搬迁需要提前三个工作日向辅导员提交申请，逾期不再受理。",
        doc_id=1,
        is_public=True,
    )
    strict = settings.model_copy(update={"retrieve_score_threshold": 0.99})

    hits = make(store, strict, user_id=None).search("宿舍申请")

    assert hits, "向量路被阈值清空后，BM25 路不应被同一阈值误杀"
    assert all(hit.matched_by == "bm25" for hit in hits)
    assert all(hit.score is None for hit in hits)


def test_bm25_route_drops_chunks_without_keyword_hit(
    store: VectorStore, settings: Settings
) -> None:
    """没有任何关键词命中时，「命中」判定必须为否，触发上层拒答。"""
    add(store, "宿舍搬迁需要提前申请。", doc_id=1, is_public=True)
    strict = settings.model_copy(update={"retrieve_score_threshold": 0.99})

    assert make(store, strict, user_id=None).search("火星采矿条例") == []


# ==================== 权限过滤（FR-31）====================


def test_other_users_private_chunks_are_excluded(
    store: VectorStore, settings: Settings
) -> None:
    add(store, "B 同学的私人笔记：编译原理重点整理。", doc_id=100, user_id=2,
        filename="B的笔记.pdf")

    assert make(store, settings, user_id=1).search("编译原理重点") == []

    own = make(store, settings, user_id=2).search("编译原理重点")
    assert own and own[0].doc_id == 100


def test_public_chunks_are_visible_to_every_identity(
    store: VectorStore, settings: Settings
) -> None:
    add(store, "学校公共通知：机房开放时间调整。", doc_id=200, is_public=True, category="admin")

    for identity in (1, 2, None):
        hits = make(store, settings, user_id=identity).search("机房开放时间")
        assert hits, f"user_id={identity} 检索不到公共文档"
        assert hits[0].doc_id == 200


# ==================== 分类过滤（TC-U10 / FR-14）====================


def test_category_filter_limits_both_routes(store: VectorStore, settings: Settings) -> None:
    add(store, "宿舍搬迁申请流程说明", doc_id=1, is_public=True, category="freshman")
    add(store, "请假申请流程说明", doc_id=2, is_public=True, category="admin")

    hits = make(store, settings, user_id=None, category="admin").search("申请流程说明")

    assert {hit.doc_id for hit in hits} == {2}
    assert all(hit.category == "admin" for hit in hits)


# ==================== 边界 ====================


def test_empty_query_returns_nothing(store: VectorStore, settings: Settings) -> None:
    add(store, "任意内容", doc_id=1, is_public=True)

    assert make(store, settings, user_id=None).search("") == []
    assert make(store, settings, user_id=None).search("   ") == []


def test_empty_corpus_returns_nothing(store: VectorStore, settings: Settings) -> None:
    assert make(store, settings, user_id=None).search("任意问题") == []


def test_results_are_capped_by_top_k(store: VectorStore, settings: Settings) -> None:
    for index in range(6):
        add(
            store,
            f"宿舍搬迁流程第 {index} 条说明。",
            doc_id=1,
            is_public=True,
            chunk_index=index,
        )

    hits = make(store, settings, user_id=None).search("宿舍搬迁流程")

    assert len(hits) == settings.retrieve_top_k == 5
