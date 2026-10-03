"""向量库封装单元测试。

重点覆盖 **FR-31 数据隔离（安全红线）**，对应 docs/03 §6.1 的 TC-U16 / TC-U17 / TC-U21。

测试注入确定性假向量，不下载真实模型，因此可在离线环境快速验证过滤逻辑。
"""

from __future__ import annotations

import pytest

from config.settings import CHROMA_DISTANCE, PUBLIC_USER_ID
from src.store.chroma import VectorStore, build_metadata


def add(store: VectorStore, text: str, *, doc_id: int, user_id=None, is_public=False,
        filename: str = "doc.pdf", chunk_index: int = 0, category: str = "freshman") -> None:
    metadata = build_metadata(
        doc_id=doc_id, user_id=user_id, is_public=is_public,
        category=category, filename=filename, chunk_index=chunk_index,
    )
    store.add_chunks([(text, metadata)])


# ==================== 元数据约定 ====================


def test_build_metadata_for_personal_document() -> None:
    metadata = build_metadata(
        doc_id=12, user_id=3, is_public=False, category="freshman",
        filename="笔记.pdf", chunk_index=8,
    )

    assert metadata["doc_id"] == 12
    assert metadata["user_id"] == 3
    assert metadata["is_public"] == 0
    assert metadata["chunk_index"] == 8
    assert metadata["source_type"] == "text"


def test_build_metadata_for_public_document_uses_sentinel_user_id() -> None:
    """公共文档的 user_id 固定为 -1（docs/02 §5.2）。"""
    metadata = build_metadata(
        doc_id=1, user_id=None, is_public=True, category="admin",
        filename="通知.pdf", chunk_index=0,
    )

    assert metadata["user_id"] == PUBLIC_USER_ID == -1
    assert metadata["is_public"] == 1


# ==================== 基础读写 ====================


def test_collection_uses_cosine_distance(store: VectorStore) -> None:
    """C-04：集合必须以 cosine 建。

    否则 `score = 1 - 距离` 就不再是余弦相似度，阈值与 Top-K 的语义全部失准。
    集合的度量创建后不可改，这条必须固定住。
    """
    metadata = store._store._collection.metadata  # noqa: SLF001
    assert metadata["hnsw:space"] == CHROMA_DISTANCE == "cosine"


def test_add_and_count(store: VectorStore) -> None:
    add(store, "宿舍搬迁需要提前三个工作日申请。", doc_id=1, user_id=1)
    add(store, "校园卡遗失应及时挂失。", doc_id=2, user_id=1)

    assert store.count() == 2


def test_search_returns_hits_with_score(store: VectorStore) -> None:
    add(store, "宿舍搬迁需要提前三个工作日向辅导员提交申请。", doc_id=1, user_id=1,
        filename="学生手册.pdf", chunk_index=3)

    hits = store.search("宿舍搬迁需要提前申请吗", user_id=1, score_threshold=0.0)

    assert hits
    top = hits[0]
    assert top.filename == "学生手册.pdf"
    assert top.doc_id == 1
    assert top.chunk_index == 3
    assert top.matched_by == "vector"
    assert top.score is not None and 0.0 <= top.score <= 1.0


def test_search_empty_query_returns_nothing(store: VectorStore) -> None:
    add(store, "任何内容", doc_id=1, user_id=1)

    assert store.search("", user_id=1) == []
    assert store.search("   ", user_id=1) == []


def test_add_chunks_skips_blank_text(store: VectorStore) -> None:
    written = store.add_chunks([("", {}), ("   ", {}), ("有效内容", {"doc_id": 1})])

    assert written == 1
    assert store.count() == 1


def test_score_threshold_filters_weak_matches(store: VectorStore) -> None:
    add(store, "宿舍搬迁需要提前三个工作日申请。", doc_id=1, user_id=1)

    strict = store.search("完全无关的火星采矿条例", user_id=1, score_threshold=0.99)
    loose = store.search("完全无关的火星采矿条例", user_id=1, score_threshold=0.0)

    assert strict == []
    assert len(loose) == 1


# ==================== FR-31 安全红线 ====================


def test_tc_u16_user_cannot_see_others_private_documents(store: VectorStore) -> None:
    """TC-U16：用户 A 用任意查询都检索不到 B 的个人资料。"""
    add(store, "B 同学的私人复习笔记：编译原理重点整理。", doc_id=100, user_id=2,
        filename="B的笔记.pdf")

    # A 用尽各种问法
    for query in ["编译原理重点", "复习笔记", "B的笔记", "私人", "编译原理重点整理"]:
        hits = store.search(query, user_id=1, score_threshold=0.0)
        assert hits == [], f"用户 A 越权检索到了 B 的资料，query={query}"


def test_tc_u17_everyone_can_see_public_documents(store: VectorStore) -> None:
    """TC-U17：公共文档任何人（含无身份）都能检索到。"""
    add(store, "学校公共通知：机房开放时间调整。", doc_id=200, is_public=True,
        filename="机房通知.pdf", category="admin")

    for identity in (1, 2, None):
        hits = store.search("机房开放时间", user_id=identity, score_threshold=0.0)
        assert hits, f"user_id={identity} 检索不到公共文档"
        assert hits[0].filename == "机房通知.pdf"


def test_user_sees_own_and_public_but_not_others(store: VectorStore) -> None:
    add(store, "公共通知：图书馆规则", doc_id=1, is_public=True)
    add(store, "我的私人笔记：图书馆复习重点", doc_id=2, user_id=1)
    add(store, "别人的私人笔记：图书馆复习重点", doc_id=3, user_id=2)

    hits = store.search("图书馆", user_id=1, score_threshold=0.0)
    doc_ids = {h.doc_id for h in hits}

    assert doc_ids == {1, 2}
    assert 3 not in doc_ids


def test_anonymous_identity_only_sees_public(store: VectorStore) -> None:
    """user_id=None 表示无身份，只能看到公共文档——比带身份更严格。"""
    add(store, "公共通知：机房开放", doc_id=1, is_public=True)
    add(store, "个人笔记：机房实验记录", doc_id=2, user_id=1)

    hits = store.search("机房", user_id=None, score_threshold=0.0)

    assert {h.doc_id for h in hits} == {1}


def test_tc_u21_missing_user_id_raises_type_error(store: VectorStore) -> None:
    """TC-U21：漏传 user_id 必须在调用层就失败，不允许降级为无过滤。"""
    add(store, "任意内容", doc_id=1, user_id=1)

    with pytest.raises(TypeError):
        store.search("任意内容")  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        store.search("任意内容", category="freshman")  # type: ignore[call-arg]


def test_filter_is_never_empty(store: VectorStore) -> None:
    """兜底断言：无论何种入参，过滤条件都不为空。"""
    assert store._build_filter(1, None)
    assert store._build_filter(None, None)
    assert store._build_filter(1, "freshman")


def test_filter_shape_for_known_identity(store: VectorStore) -> None:
    condition = store._build_filter(7, None)
    assert condition == {
        "$or": [{"is_public": {"$eq": 1}}, {"user_id": {"$eq": 7}}]
    }


def test_filter_shape_for_anonymous(store: VectorStore) -> None:
    assert store._build_filter(None, None) == {"is_public": {"$eq": 1}}


def test_filter_shape_with_category(store: VectorStore) -> None:
    condition = store._build_filter(None, "admin")
    assert condition == {
        "$and": [{"is_public": {"$eq": 1}}, {"category": {"$eq": "admin"}}]
    }


# ==================== 分类过滤 ====================


def test_search_category_filter_limits_results(store: VectorStore) -> None:
    add(store, "宿舍搬迁申请流程说明", doc_id=1, user_id=1, category="freshman")
    add(store, "请假申请流程说明", doc_id=2, user_id=1, category="admin")

    hits = store.search("申请流程说明", user_id=1, category="admin", score_threshold=0.0)

    assert {h.doc_id for h in hits} == {2}
    assert all(h.category == "admin" for h in hits)


# ==================== 删除 ====================


def test_delete_by_doc_id_removes_all_its_chunks(store: VectorStore) -> None:
    for index in range(3):
        add(store, f"第 {index} 段内容：宿舍搬迁流程。", doc_id=1, user_id=1, chunk_index=index)
    add(store, "另一份文档的内容。", doc_id=2, user_id=1)

    removed = store.delete_by_doc_id(1)

    assert removed == 3
    assert store.count() == 1
    # 被删文档的向量不再出现；文档 2 仍在（score_threshold=0.0 表示不过滤）
    remaining = store.search("宿舍搬迁流程", user_id=1, score_threshold=0.0)
    assert {hit.doc_id for hit in remaining} == {2}


def test_delete_by_user_id_keeps_public_documents(store: VectorStore) -> None:
    add(store, "公共通知：图书馆规则。", doc_id=1, is_public=True)
    add(store, "我的私人笔记：图书馆复习。", doc_id=2, user_id=1)

    removed = store.delete_by_user_id(1)

    assert removed == 1
    remaining = store.search("图书馆", user_id=1, score_threshold=0.0)
    assert {h.doc_id for h in remaining} == {1}


def test_delete_by_user_id_does_not_touch_other_users(store: VectorStore) -> None:
    add(store, "用户2的笔记。", doc_id=1, user_id=2)

    assert store.delete_by_user_id(1) == 0
    assert store.count() == 1


def test_delete_missing_doc_returns_zero(store: VectorStore) -> None:
    assert store.delete_by_doc_id(999) == 0
