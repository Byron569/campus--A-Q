"""文档管理页纯逻辑测试。

覆盖 docs/05 §PG-03 的交互要求与 docs/03 §1.7 M1-18 的完成标准：
多选上传、校验（类型/大小）、进度、列表、删除级联、失败重试（DR-05）、防重复提交（DR-07）。

渲染函数不测（st.* 编排），但页面用到的全部判断逻辑都在纯函数里，此处覆盖。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from config.settings import Settings
from src.errors import IngestError
from src.ingest.pipeline import ingest_file
from src.repository import (
    DOC_DELETED,
    TASK_DONE,
    TASK_FAILED,
    TASK_PENDING,
    TASK_RUNNING,
    create_task,
    get_document,
    get_task,
    get_task_by_doc,
    update_task,
)
from src.files import sanitize_filename, stored_path, uploads_dir
from src.store.chroma import VectorStore
from src.ui.documents import (
    delete_document,
    enqueue_new,
    retry_document,
    stage_upload,
    validate_upload,
)

CONTENT = ("校园卡遗失后应及时挂失并补办。补办地点为行政楼一楼服务大厅。" * 30).encode("utf-8")


def wait_for_task(task_id: int, settings: Settings, timeout: float = 20.0):
    """等后台入库线程结束，返回最终任务状态。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = get_task(task_id, db_path=settings.database_path)
        if task.status not in (TASK_PENDING, TASK_RUNNING):
            return task
        time.sleep(0.05)
    raise AssertionError(f"任务 {task_id} 在 {timeout}s 内未结束")


def stage_with_task(settings: Settings, *, name: str = "通知.txt", content: bytes = CONTENT):
    """落盘并建一条 pending 任务，返回 (staged, task_id)。"""
    staged = stage_upload(
        name, content, category="admin", is_public=True, user_id=None, settings=settings
    )
    task_id = create_task(staged.doc_id, user_id=None, db_path=settings.database_path)
    return staged, task_id


# ==================== 文件名清洗 ====================


def test_sanitize_filename_strips_directory_traversal() -> None:
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("/tmp/通知.pdf") == "通知.pdf"
    assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"


def test_sanitize_filename_replaces_illegal_chars() -> None:
    assert sanitize_filename('a:b*c?d"e<f>g|h.pdf') == "a_b_c_d_e_f_g_h.pdf"


def test_sanitize_filename_rejects_hidden_and_empty() -> None:
    assert sanitize_filename(".env") == "env"
    assert sanitize_filename("   ") == "未命名文件"
    assert sanitize_filename("") == "未命名文件"


# ==================== 上传校验（PG-03） ====================


def test_validate_upload_accepts_supported_suffix(settings: Settings) -> None:
    assert validate_upload("学生手册.pdf", 1024, settings=settings) == ""
    assert validate_upload("通知.DOCX", 1024, settings=settings) == ""


def test_validate_upload_rejects_unknown_suffix(settings: Settings) -> None:
    reason = validate_upload("木马.exe", 1024, settings=settings)
    assert "不支持的文件类型" in reason
    assert ".exe" in reason


def test_validate_upload_rejects_oversized_file(settings: Settings) -> None:
    over = settings.max_upload_bytes + 1
    reason = validate_upload("学生手册.pdf", over, settings=settings)
    assert "超过 20MB 上限" in reason

    assert validate_upload("学生手册.pdf", settings.max_upload_bytes, settings=settings) == ""


# ==================== 落盘与建记录 ====================


def test_stage_upload_creates_record_and_file(settings: Settings, db: Path) -> None:
    staged = stage_upload(
        "校园卡须知.txt", CONTENT, category="freshman", is_public=True, user_id=None,
        settings=settings,
    )

    document = get_document(staged.doc_id, db_path=settings.database_path)
    assert document.filename == "校园卡须知.txt"
    assert document.filetype == "txt"
    assert document.category == "freshman"
    assert document.is_public is True
    assert document.user_id is None
    assert document.size_bytes == len(CONTENT)

    assert staged.path.exists()
    assert staged.path.read_bytes() == CONTENT
    assert staged.path.name == f"{staged.doc_id}_校园卡须知.txt"

    # 任务由 enqueue_new 负责创建：若这里就建好 pending 任务，
    # enqueue_new 的 DR-07 幂等校验会把自己刚建的任务当成重复提交
    assert get_task_by_doc(staged.doc_id, db_path=settings.database_path) is None


def test_public_document_lands_in_sentinel_directory(settings: Settings, db: Path) -> None:
    staged = stage_upload(
        "通知.txt", CONTENT, category="admin", is_public=True, user_id=None, settings=settings
    )

    assert staged.path.parent == settings.uploads_path / "-1"
    assert staged.path.parent == uploads_dir(None, settings=settings)


def test_same_filename_does_not_overwrite(settings: Settings, db: Path) -> None:
    """同名文件必须各存各的，否则删除一个会连带删掉另一个的原文件。"""
    first = stage_upload(
        "通知.txt", CONTENT, category="admin", is_public=True, user_id=None, settings=settings
    )
    second = stage_upload(
        "通知.txt", b"second", category="admin", is_public=True, user_id=None, settings=settings
    )

    assert first.path != second.path
    assert first.path.read_bytes() == CONTENT
    assert second.path.read_bytes() == b"second"


def test_stored_path_is_reconstructible(settings: Settings, db: Path) -> None:
    """重试要靠文档记录反推原始文件路径，因此路径必须可重建。"""
    staged = stage_upload(
        "通知.txt", CONTENT, category="admin", is_public=True, user_id=None, settings=settings
    )
    document = get_document(staged.doc_id, db_path=settings.database_path)

    rebuilt = stored_path(
        owner_id=document.user_id,
        doc_id=document.id,
        filename=document.filename,
        settings=settings,
    )
    assert rebuilt == staged.path


# ==================== 防重复提交（DR-07） ====================


def test_enqueue_new_rejects_duplicate_active_task(
    settings: Settings, db: Path, store: VectorStore
) -> None:
    """同一文档已有未完成任务时，不得再建一条（DR-07）。"""
    staged = stage_upload(
        "通知.txt", CONTENT, category="admin", is_public=True, user_id=None, settings=settings
    )
    create_task(staged.doc_id, user_id=None, db_path=settings.database_path)

    with pytest.raises(IngestError, match="请勿重复提交"):
        enqueue_new(
            staged, category="admin", is_public=True, user_id=None, store=store, settings=settings
        )


def test_enqueue_new_starts_fresh_task(settings: Settings, db: Path, store: VectorStore) -> None:
    """正常路径：新建任务并真正启动后台入库。"""
    staged = stage_upload(
        "通知.txt", CONTENT, category="admin", is_public=True, user_id=None, settings=settings
    )

    task_id = enqueue_new(
        staged, category="admin", is_public=True, user_id=None, store=store, settings=settings
    )
    task = wait_for_task(task_id, settings)

    assert task.status == TASK_DONE
    assert task.doc_id == staged.doc_id
    assert store.count() == task.total_chunks > 0
    assert get_document(staged.doc_id, db_path=settings.database_path).chunk_count == task.total_chunks


# ==================== 删除级联（M1-A5） ====================


def test_delete_document_clears_record_vectors_and_file(
    settings: Settings, db: Path, store: VectorStore
) -> None:
    staged, task_id = stage_with_task(settings)
    result = ingest_file(
        staged.path, category="admin", is_public=True, user_id=None,
        doc_id=staged.doc_id, task_id=task_id, store=store,
        db_path=settings.database_path,
    )
    assert store.count() > 0

    delete_document(staged.doc_id, store=store, settings=settings)

    assert get_document(staged.doc_id, db_path=settings.database_path).status == DOC_DELETED
    assert store.count() == 0
    assert not staged.path.exists()
    # 删除后不应再检索到该文档
    assert store.search("校园卡挂失", user_id=None, score_threshold=0.0) == []
    assert result.chunk_count > 0


def test_delete_document_is_idempotent(settings: Settings, db: Path, store: VectorStore) -> None:
    staged = stage_upload(
        "通知.txt", CONTENT, category="admin", is_public=True, user_id=None, settings=settings
    )

    delete_document(staged.doc_id, store=store, settings=settings)
    delete_document(staged.doc_id, store=store, settings=settings)  # 重复删除不报错

    assert get_document(staged.doc_id, db_path=settings.database_path).status == DOC_DELETED


# ==================== 失败重试（DR-05：复用原任务） ====================


def test_retry_reuses_task_and_clears_stale_vectors(
    settings: Settings, db: Path, store: VectorStore
) -> None:
    staged, task_id = stage_with_task(settings)
    original = ingest_file(
        staged.path, category="admin", is_public=True, user_id=None,
        doc_id=staged.doc_id, task_id=task_id, store=store,
        db_path=settings.database_path,
    )
    update_task(
        task_id, db_path=settings.database_path,
        status=TASK_FAILED, error="向量化失败（模拟）",
    )

    retry_document(staged.doc_id, store=store, settings=settings)
    task = wait_for_task(task_id, settings)

    assert task.id == task_id                     # 复用原任务，不新建记录
    assert task.status == TASK_DONE
    assert task.error is None
    assert get_task_by_doc(staged.doc_id, db_path=settings.database_path).id == task_id
    # 旧向量已清理，重试后不会翻倍
    assert store.count() == original.chunk_count
    assert store.count() == task.total_chunks


def test_retry_rejects_while_task_is_running(
    settings: Settings, db: Path, store: VectorStore
) -> None:
    staged, task_id = stage_with_task(settings)
    update_task(task_id, db_path=settings.database_path, status=TASK_RUNNING)

    with pytest.raises(IngestError, match="正在入库"):
        retry_document(staged.doc_id, store=store, settings=settings)


def test_retry_reports_missing_source_file(settings: Settings, db: Path, store: VectorStore) -> None:
    staged, task_id = stage_with_task(settings)
    update_task(task_id, db_path=settings.database_path, status=TASK_FAILED, error="x")
    staged.path.unlink()

    with pytest.raises(IngestError, match="原始文件已丢失"):
        retry_document(staged.doc_id, store=store, settings=settings)


def test_retry_reports_deleted_document(settings: Settings, db: Path, store: VectorStore) -> None:
    staged = stage_upload(
        "通知.txt", CONTENT, category="admin", is_public=True, user_id=None, settings=settings
    )
    delete_document(staged.doc_id, store=store, settings=settings)

    with pytest.raises(IngestError, match="文档不存在或已删除"):
        retry_document(staged.doc_id, store=store, settings=settings)
