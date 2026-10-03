"""入库流水线与异步入库任务测试。

覆盖 docs/09 §7 的两条联调缝：
- C-01 `ingest.splitter` ↔ `ingest.pipeline` ↔ `chroma.add_chunks`：
  切片数 = 写入条数 = `documents.chunk_count`，元数据字段齐全
- C-02 `ingest.tasks` ↔ `repository` ↔ 前端进度：
  进度计数单调推进不跳变，失败写 `error`

依赖 `conftest` 中的 `store`（假向量）与 `db`（隔离 SQLite）夹具，离线可跑。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import PUBLIC_USER_ID
from src.errors import IngestError, UnsupportedFileType
from src.ingest.pipeline import ingest_file
from src.ingest.tasks import recover_stale_tasks, submit_ingest
from src.repository import (
    DOC_ACTIVE,
    TASK_DONE,
    TASK_FAILED,
    TASK_RUNNING,
    create_document,
    create_task,
    get_document,
    get_task,
    update_task,
)
from src.store.chroma import VectorStore

SENTENCE = "校园卡遗失后应及时挂失并补办，补办地点为行政楼一楼服务大厅。"


def write_text_file(tmp_path: Path, *, name: str = "校园卡须知.txt", repeats: int = 40) -> Path:
    """造一个必然被切成多片的纯文本文件。"""
    target = tmp_path / name
    target.write_text(SENTENCE * repeats, encoding="utf-8")
    return target


def prepare_doc_and_task(db: Path, filename: str, user_id: int | None = 1) -> tuple[int, int]:
    doc_id = create_document(
        filename=filename, filetype="txt", category="uncategorized",
        user_id=user_id, db_path=db,
    )
    return doc_id, create_task(doc_id, user_id=user_id, db_path=db)


class RecordingStore:
    """包一层真实 store，记录每批写入的切片条数，用于核对分批粒度。"""

    def __init__(self, inner: VectorStore) -> None:
        self._inner = inner
        self.batch_sizes: list[int] = []

    def add_chunks(self, chunks):
        written = self._inner.add_chunks(chunks)
        self.batch_sizes.append(len(chunks))
        return written


# ==================== C-01：切片 ↔ 写入 ↔ 元数据 ====================


def test_chunk_count_matches_vector_count_and_db(db: Path, store: VectorStore, tmp_path: Path) -> None:
    path = write_text_file(tmp_path)

    result = ingest_file(path, user_id=1, store=store, db_path=db)

    document = get_document(result.doc_id, db_path=db)
    assert result.chunk_count > 1
    assert document.chunk_count == result.chunk_count
    assert store.count() == result.chunk_count
    assert document.status == DOC_ACTIVE
    assert document.filename == "校园卡须知.txt"
    assert document.filetype == "txt"
    assert document.category == "uncategorized"
    assert result.elapsed_ms >= 0


def test_personal_metadata_is_correct(db: Path, store: VectorStore, tmp_path: Path) -> None:
    path = write_text_file(tmp_path)

    result = ingest_file(path, user_id=2, category="course", store=store, db_path=db)

    raw = store._store.get(where={"doc_id": {"$eq": result.doc_id}})  # noqa: SLF001
    metadatas = raw["metadatas"]
    assert len(metadatas) == result.chunk_count
    assert {m["user_id"] for m in metadatas} == {2}
    assert {m["is_public"] for m in metadatas} == {0}
    assert {m["category"] for m in metadatas} == {"course"}
    assert {m["filename"] for m in metadatas} == {"校园卡须知.txt"}
    assert {m["source_type"] for m in metadatas} == {"text"}
    assert sorted(m["chunk_index"] for m in metadatas) == list(range(result.chunk_count))


def test_public_metadata_uses_sentinel_and_is_searchable_by_anyone(
    db: Path, store: VectorStore, tmp_path: Path
) -> None:
    path = write_text_file(tmp_path)

    result = ingest_file(
        path, user_id=None, is_public=True, category="admin", store=store, db_path=db
    )

    raw = store._store.get(where={"doc_id": {"$eq": result.doc_id}})  # noqa: SLF001
    assert {m["user_id"] for m in raw["metadatas"]} == {PUBLIC_USER_ID}
    assert {m["is_public"] for m in raw["metadatas"]} == {1}
    # 公共文档对任意身份（含有身份的用户）都可见
    assert store.search("校园卡挂失", user_id=1, score_threshold=0.0)


def test_existing_doc_id_is_reused(db: Path, store: VectorStore, tmp_path: Path) -> None:
    path = write_text_file(tmp_path)
    doc_id, _ = prepare_doc_and_task(db, path.name)

    result = ingest_file(path, user_id=1, doc_id=doc_id, store=store, db_path=db)

    assert result.doc_id == doc_id
    assert get_document(doc_id, db_path=db).chunk_count == result.chunk_count


# ==================== C-02：进度不跳变、不回退 ====================


def capture_progress_writes(monkeypatch) -> list[dict]:
    """拦截 pipeline 对 ingest_tasks 的写入，直接观察进度序列。"""
    from src.ingest import pipeline

    writes: list[dict] = []
    real_update_task = pipeline.update_task

    def spy(task_id: int, db_path=None, **fields):
        writes.append(dict(fields))
        return real_update_task(task_id, db_path=db_path, **fields)

    monkeypatch.setattr(pipeline, "update_task", spy)
    return writes


def test_progress_is_monotonic_across_batches(
    db: Path, store: VectorStore, tmp_path: Path, monkeypatch
) -> None:
    path = write_text_file(tmp_path)
    doc_id, task_id = prepare_doc_and_task(db, path.name)
    writes = capture_progress_writes(monkeypatch)

    result = ingest_file(
        path, user_id=1, doc_id=doc_id, task_id=task_id,
        store=store, batch_size=1, db_path=db,
    )

    # 分母先落库且 done 归零，前端进度条才不会出现分母为 0 的跳变
    assert writes[0]["total_chunks"] == result.chunk_count
    assert writes[0]["done_chunks"] == 0

    # batch_size=1 → 先归零，再每批 +1，严格递增、不跳变、不回退
    done_sequence = [w["done_chunks"] for w in writes if "done_chunks" in w]
    assert done_sequence == list(range(0, result.chunk_count + 1))

    task = get_task(task_id, db_path=db)
    assert task.total_chunks == result.chunk_count
    assert task.done_chunks == result.chunk_count
    assert task.progress == 1.0
    assert task.error is None


def test_batches_respect_batch_size(db: Path, store: VectorStore, tmp_path: Path) -> None:
    path = write_text_file(tmp_path)
    doc_id, task_id = prepare_doc_and_task(db, path.name)

    recorder = RecordingStore(store)
    result = ingest_file(
        path, user_id=1, doc_id=doc_id, task_id=task_id,
        store=recorder, batch_size=2, db_path=db,
    )

    assert sum(recorder.batch_sizes) == result.chunk_count
    assert all(size <= 2 for size in recorder.batch_sizes)
    assert len(recorder.batch_sizes) == -(-result.chunk_count // 2)


def test_ingest_without_task_id_still_writes_vectors(db: Path, store: VectorStore, tmp_path: Path) -> None:
    """命令行同步入库不传 task_id，不应因为缺少任务记录而失败。"""
    path = write_text_file(tmp_path)

    result = ingest_file(path, user_id=1, store=store, db_path=db)

    assert store.count() == result.chunk_count


# ==================== 异常路径（docs/02 §11） ====================


def test_empty_file_raises_ingest_error(db: Path, store: VectorStore, tmp_path: Path) -> None:
    path = tmp_path / "空文件.txt"
    path.write_text("   \n\n  \t ", encoding="utf-8")

    with pytest.raises(IngestError, match="未能从文件中提取到文本"):
        ingest_file(path, user_id=1, store=store, db_path=db)


def test_personal_document_without_user_id_is_rejected(
    db: Path, store: VectorStore, tmp_path: Path
) -> None:
    """没有归属的个人资料既非公共、又匹配不到任何用户，等于静默丢数据。"""
    path = write_text_file(tmp_path)

    with pytest.raises(IngestError, match="必须指定 user_id"):
        ingest_file(path, is_public=False, store=store, db_path=db)


def test_unsupported_suffix_propagates(db: Path, store: VectorStore, tmp_path: Path) -> None:
    path = tmp_path / "可疑文件.exe"
    path.write_bytes(b"MZ")

    with pytest.raises(UnsupportedFileType):
        ingest_file(path, user_id=1, store=store, db_path=db)


def test_vectorization_failure_raises_ingest_error(db: Path, tmp_path: Path) -> None:
    path = write_text_file(tmp_path)
    doc_id, task_id = prepare_doc_and_task(db, path.name)

    class BrokenStore:
        def add_chunks(self, chunks):
            raise RuntimeError("模型加载失败")

    with pytest.raises(IngestError, match="向量化失败"):
        ingest_file(
            path, user_id=1, doc_id=doc_id, task_id=task_id,
            store=BrokenStore(), db_path=db,
        )


# ==================== 异步入库任务 ====================


def test_submit_ingest_marks_task_done(db: Path, store: VectorStore, tmp_path: Path) -> None:
    path = write_text_file(tmp_path)
    doc_id, task_id = prepare_doc_and_task(db, path.name)

    thread = submit_ingest(
        path=path, doc_id=doc_id, task_id=task_id, user_id=1,
        store=store, batch_size=4, db_path=db,
    )

    assert thread.daemon is True
    thread.join(timeout=30)
    assert not thread.is_alive()

    task = get_task(task_id, db_path=db)
    assert task.status == TASK_DONE
    assert task.progress == 1.0
    assert task.error is None
    assert get_document(doc_id, db_path=db).chunk_count == task.total_chunks


def test_submit_ingest_failure_writes_error_without_raising(
    db: Path, store: VectorStore, tmp_path: Path
) -> None:
    path = tmp_path / "空文件.txt"
    path.write_text("", encoding="utf-8")
    doc_id, task_id = prepare_doc_and_task(db, path.name)

    thread = submit_ingest(
        path=path, doc_id=doc_id, task_id=task_id, user_id=1, store=store, db_path=db
    )
    thread.join(timeout=30)

    task = get_task(task_id, db_path=db)
    assert task.status == TASK_FAILED
    assert "未能从文件中提取到文本" in task.error
    # 失败时不得留下任何向量
    assert store.count() == 0


def test_recover_stale_tasks_delegates_to_repository(db: Path) -> None:
    doc_id, task_id = prepare_doc_and_task(db, "a.txt")
    update_task(task_id, db_path=db, status=TASK_RUNNING, total_chunks=10, done_chunks=3)

    assert recover_stale_tasks(db_path=db) == 1
    assert get_task(task_id, db_path=db).status == TASK_FAILED
