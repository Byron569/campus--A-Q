"""备份与恢复单元测试（FR-24，二期）。

覆盖往返一致性：备份 → 在空的隔离目录恢复 → 记录、原始文件、向量三处都回来，
且检索能命中；以及恢复的保护分支（目标库非空、备份损坏、版本不符）。
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from config.settings import Settings
from src.backup import BACKUP_TABLES, create_backup, restore_backup
from src.errors import BackupError
from src.ingest.pipeline import ingest_file
from src.repository import (
    add_feedback,
    add_message,
    add_sources,
    create_conversation,
    create_document,
    import_tables,
)
from src.store.chroma import SearchHit, VectorStore
from src.store.db import get_conn, init_db
from tests.conftest import FakeEmbeddings

CONTENT = "宿舍搬迁需要提前三个工作日向辅导员提交申请。"


def _seed_source(settings: Settings, db: Path) -> int:
    """造一份带原始文件、会话、引用与反馈的库，返回 doc_id。"""
    settings.ensure_dirs()
    doc_id = create_document(
        filename="搬迁须知.txt",
        filetype="txt",
        category="freshman",
        size_bytes=len(CONTENT.encode("utf-8")),
        user_id=1,
        is_public=False,
        db_path=db,
    )
    target = settings.uploads_path / "1" / f"{doc_id}_搬迁须知.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(CONTENT, encoding="utf-8")

    conversation_id = create_conversation(1, title="搬迁问题", db_path=db)
    user_message = add_message(conversation_id, "user", "搬宿舍要提前几天？", db_path=db)
    assistant_message = add_message(conversation_id, "assistant", CONTENT, db_path=db)
    add_sources(
        assistant_message,
        [
            SearchHit(
                text=CONTENT,
                score=0.81,
                filename="搬迁须知.txt",
                doc_id=doc_id,
                chunk_index=0,
                category="freshman",
                matched_by="both",
            )
        ],
        db_path=db,
    )
    add_feedback(assistant_message, 1, "useful", db_path=db)
    assert user_message < assistant_message
    return doc_id


def _target_settings(tmp_path: Path) -> Settings:
    """模拟「另一台机器 / 空库」：独立的数据目录与向量目录。"""
    data_dir = tmp_path / "target"
    return Settings(_env_file=None, data_dir=data_dir, chroma_dir=data_dir / "chroma")


# ==================== 往返一致性 ====================


def test_backup_then_restore_round_trip(tmp_path: Path, db: Path, settings: Settings) -> None:
    doc_id = _seed_source(settings, db)
    archive = tmp_path / "backup.zip"

    create_backup(settings=settings, out_path=archive)
    assert archive.exists()

    target = _target_settings(tmp_path)
    target_store = VectorStore(settings=target, embeddings=FakeEmbeddings())
    result = restore_backup(
        settings=target, zip_path=archive, store=target_store, force=True
    )

    # 记录：用户、文档、会话、消息、引用、反馈都回来了
    assert result.table_counts["users"] == 2
    assert result.table_counts["documents"] == 1
    assert result.table_counts["conversations"] == 1
    assert result.table_counts["messages"] == 2
    assert result.table_counts["message_sources"] == 1
    assert result.table_counts["feedback"] == 1

    # 原始文件按原相对路径还原
    restored_file = target.uploads_path / "1" / f"{doc_id}_搬迁须知.txt"
    assert restored_file.read_text(encoding="utf-8") == CONTENT

    # 向量重建：能检索到，且权限过滤照旧
    hits = target_store.search("宿舍搬迁", user_id=1, score_threshold=0.0)
    assert hits and hits[0].filename == "搬迁须知.txt"
    assert target_store.search("宿舍搬迁", user_id=2, score_threshold=0.0) == []
    assert result.reindexed_docs == 1
    assert result.chunk_total >= 1
    assert result.missing_files == []


def test_round_trip_covers_cli_ingested_document(
    tmp_path: Path, db: Path, settings: Settings, store: VectorStore
) -> None:
    """批量入库（CLI）的文档也必须能被备份、恢复并重建向量。

    这类文档原先不落盘原始文件，备份里只剩网页上传的文件，恢复时整批
    「缺原始文件」无法重建（真实数据往返实测暴露）。
    """
    settings.ensure_dirs()
    source = tmp_path / "搬迁须知.txt"
    source.write_text(CONTENT, encoding="utf-8")
    ingested = ingest_file(
        source, is_public=True, user_id=None, store=store, db_path=db, settings=settings
    )

    # 入库即落盘：公共文档归属目录为 -1
    stored = settings.uploads_path / "-1" / f"{ingested.doc_id}_搬迁须知.txt"
    assert stored.read_text(encoding="utf-8") == CONTENT

    archive = tmp_path / "backup.zip"
    create_backup(settings=settings, out_path=archive)

    target = _target_settings(tmp_path)
    target_store = VectorStore(settings=target, embeddings=FakeEmbeddings())
    restored = restore_backup(settings=target, zip_path=archive, store=target_store, force=True)

    assert restored.reindexed_docs == 1
    assert restored.missing_files == []
    hits = target_store.search("宿舍搬迁", user_id=None, score_threshold=0.0)
    assert hits and hits[0].filename == "搬迁须知.txt"


def test_backup_excludes_env_and_metrics(tmp_path: Path, db: Path, settings: Settings) -> None:
    """备份只带业务表：不含 .env、不含脱敏日志 qa_metrics。"""
    _seed_source(settings, db)
    with get_conn(db) as conn:
        conn.execute("INSERT INTO qa_metrics (question_len, answerable) VALUES (5, 1)")

    archive = tmp_path / "backup.zip"
    manifest = create_backup(settings=settings, out_path=archive)

    assert set(manifest.table_counts) == set(BACKUP_TABLES)
    assert "qa_metrics" not in manifest.table_counts

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert "metadata/qa_metrics.json" not in names
    assert all(".env" not in name for name in names)
    assert "manifest.json" in names
    assert "knowledge_base.yaml" in names


def test_restore_can_skip_reindex(tmp_path: Path, db: Path, settings: Settings) -> None:
    _seed_source(settings, db)
    archive = tmp_path / "backup.zip"
    create_backup(settings=settings, out_path=archive)

    target = _target_settings(tmp_path)
    result = restore_backup(
        settings=target, zip_path=archive, store=None, force=True, reindex=False
    )

    assert result.reindexed_docs == 0
    assert result.table_counts["documents"] == 1


# ==================== 保护分支 ====================


def test_restore_refuses_non_empty_target_without_force(
    tmp_path: Path, db: Path, settings: Settings
) -> None:
    """目标库已有数据时默认拒绝，避免一次误操作把现有数据整库替换掉。"""
    _seed_source(settings, db)
    archive = tmp_path / "backup.zip"
    create_backup(settings=settings, out_path=archive)

    # 仍恢复到原库（已有 users 1/2）
    with pytest.raises(BackupError, match="--force"):
        restore_backup(settings=settings, zip_path=archive, store=None)


def test_restore_rejects_missing_file(tmp_path: Path, settings: Settings) -> None:
    with pytest.raises(BackupError, match="不存在"):
        restore_backup(
            settings=settings, zip_path=tmp_path / "nope.zip", store=None, force=True
        )


def test_restore_rejects_broken_zip(tmp_path: Path, settings: Settings) -> None:
    broken = tmp_path / "broken.zip"
    broken.write_text("这不是一个 zip", encoding="utf-8")

    with pytest.raises(BackupError, match="损坏"):
        restore_backup(settings=settings, zip_path=broken, store=None, force=True)


def test_restore_rejects_unsupported_version(tmp_path: Path, settings: Settings) -> None:
    archive = tmp_path / "future.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"version": 999, "created_at": "", "table_counts": {}, "file_count": 0}),
        )

    with pytest.raises(BackupError, match="版本不支持"):
        restore_backup(settings=settings, zip_path=archive, store=None, force=True)


def test_import_tables_rejects_unknown_column(db: Path) -> None:
    """备份文件里的列名属外部输入，出现表里没有的列必须直接拒绝。"""
    with pytest.raises(ValueError, match="不存在的列"):
        import_tables({"users": [{"id": 9, "username": "x", "evil": "1"}]}, db_path=db)


def test_import_tables_is_whole_table_replacement(db: Path) -> None:
    """恢复是整库替换：备份里没有的行必须被清掉，不能与旧数据混在一起。"""
    init_db(db)
    import_tables({"users": [{"id": 7, "username": "solo", "password_hash": "x"}]}, db_path=db)

    with get_conn(db) as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM users")]

    assert [row["username"] for row in rows] == ["solo"]
