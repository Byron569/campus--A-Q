"""备份与恢复（FR-24，二期）。

设计依据：docs/02-架构设计.md §4.9（一期保留设计，二期实现）。

备份产物是一个 zip：

    manifest.json               版本、导出时间、各表条数、文件数
    knowledge_base.yaml         知识库配置快照
    metadata/<表名>.json         users / documents / conversations /
                                messages / message_sources / feedback
    uploads/<原始相对路径>       data/uploads 下的原始文件

**不含 `.env`**（密钥绝不外带）；**不含 qa_metrics**（脱敏日志，90 天自动清理，
可丢）；**向量不导出**——恢复时复用入库流水线重新向量化，避免把与大模型版本
绑定的向量数据当作文档数据一起搬运。

恢复语义是**整库替换**：先清空相关表，再写回备份记录，最后按仍为 active 的文档
重建向量。目标库非空时必须显式授权（`force=True` / `--force`），避免误覆盖。
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_BATCH_SIZE, KB_CONFIG_PATH, UNCATEGORIZED_KEY
from src.errors import BackupError
from src.ingest.loader import load_document
from src.ingest.splitter import split_text
from src.repository import BACKUP_TABLES, export_tables, import_tables
from src.store.chroma import VectorStore, build_metadata
from src.store.db import get_conn, init_db

logger = logging.getLogger(__name__)

BACKUP_VERSION = 1
MANIFEST_NAME = "manifest.json"
KB_SNAPSHOT_NAME = "knowledge_base.yaml"
METADATA_DIR = "metadata"
UPLOADS_DIR = "uploads"

DOC_DELETED = "deleted"


@dataclass
class BackupManifest:
    version: int
    created_at: str
    table_counts: dict[str, int] = field(default_factory=dict)
    file_count: int = 0


@dataclass
class RestoreResult:
    table_counts: dict[str, int] = field(default_factory=dict)
    file_count: int = 0
    reindexed_docs: int = 0
    chunk_total: int = 0
    missing_files: list[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def create_backup(*, settings, out_path: str | Path) -> BackupManifest:
    """导出整库备份到 zip，返回清单。

    Raises:
        BackupError: 输出路径不可写。
    """
    target = Path(out_path)
    tables = export_tables(settings.database_path)

    uploads_root = settings.uploads_path
    files = (
        sorted(p for p in uploads_root.rglob("*") if p.is_file())
        if uploads_root.exists()
        else []
    )

    manifest = BackupManifest(
        version=BACKUP_VERSION,
        created_at=_now(),
        table_counts={name: len(rows) for name, rows in tables.items()},
        file_count=len(files),
    )

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                MANIFEST_NAME, json.dumps(asdict(manifest), ensure_ascii=False, indent=2)
            )
            if KB_CONFIG_PATH.exists():
                archive.write(KB_CONFIG_PATH, KB_SNAPSHOT_NAME)
            for name, rows in tables.items():
                archive.writestr(
                    f"{METADATA_DIR}/{name}.json",
                    json.dumps(rows, ensure_ascii=False, indent=2),
                )
            for path in files:
                archive.write(path, f"{UPLOADS_DIR}/{path.relative_to(uploads_root)}")
    except OSError as exc:
        raise BackupError(f"写入备份文件失败：{target}（{exc}）") from exc

    logger.info(
        "备份完成 %s：%d 张表、%d 个原始文件",
        target,
        len(BACKUP_TABLES),
        manifest.file_count,
    )
    return manifest


def restore_backup(
    *,
    settings,
    zip_path: str | Path,
    store: VectorStore,
    force: bool = False,
    reindex: bool = True,
) -> RestoreResult:
    """从备份 zip 恢复整库，并按 active 文档重建向量。

    Args:
        settings: 运行配置（决定数据库、uploads、data 目录位置）。
        zip_path: 备份文件路径。
        store: 向量库实例；重新向量化写这里。
        force: 目标库非空时是否允许覆盖。默认拒绝，避免误操作毁掉现有数据。
        reindex: 是否重建向量。关闭则只恢复记录与文件（用于排查）。

    Raises:
        BackupError: 备份文件缺失 / 损坏、版本不支持、目标库非空且未授权。
    """
    source = Path(zip_path)
    if not source.exists():
        raise BackupError(f"备份文件不存在：{source}")

    settings.ensure_dirs()
    init_db(settings.database_path)

    try:
        archive = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise BackupError(f"备份文件已损坏或不是 zip：{source}") from exc

    with archive:
        manifest = _read_manifest(archive)
        if manifest.version != BACKUP_VERSION:
            raise BackupError(
                f"备份版本不支持：文件为 v{manifest.version}，当前程序支持 v{BACKUP_VERSION}"
            )
        if not force and _has_any_user(settings.database_path):
            raise BackupError(
                "目标库已有数据，恢复会整库替换。确认无误请加 --force 重新执行。"
            )

        file_count = _extract_uploads(archive, settings.uploads_path)
        tables = {name: _read_table(archive, name) for name in BACKUP_TABLES}
        counts = import_tables(tables, db_path=settings.database_path)

        # 配置快照只落到 data/ 下留档，不覆盖仓库里的 config/knowledge_base.yaml：
        # 恢复数据不该顺带改运行配置，交由管理员比对后手动替换
        if KB_SNAPSHOT_NAME in archive.namelist():
            (settings.data_dir / f"{KB_SNAPSHOT_NAME}.restored").write_bytes(
                archive.read(KB_SNAPSHOT_NAME)
            )

    result = RestoreResult(table_counts=counts, file_count=file_count)
    if reindex:
        _reindex_documents(settings, store, result)

    logger.info(
        "恢复完成 %s：记录 %s，文件 %d，重向量化 %d 篇 / %d 片",
        source,
        counts,
        file_count,
        result.reindexed_docs,
        result.chunk_total,
    )
    return result


# ==================== 内部实现 ====================


def _has_any_user(db_path) -> bool:
    with get_conn(db_path) as conn:
        return conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def _read_manifest(archive: zipfile.ZipFile) -> BackupManifest:
    try:
        raw = archive.read(MANIFEST_NAME).decode("utf-8")
        data = json.loads(raw)
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupError("备份缺少可读的 manifest.json，文件可能不完整") from exc
    try:
        return BackupManifest(
            version=int(data["version"]),
            created_at=str(data.get("created_at", "")),
            table_counts=dict(data.get("table_counts") or {}),
            file_count=int(data.get("file_count") or 0),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BackupError("备份 manifest.json 字段不完整或类型错误") from exc


def _read_table(archive: zipfile.ZipFile, name: str) -> list[dict]:
    member = f"{METADATA_DIR}/{name}.json"
    if member not in archive.namelist():
        # 旧备份可能缺表；按空表处理，让恢复继续
        logger.warning("备份中缺少 %s，按空表处理", member)
        return []
    try:
        data = json.loads(archive.read(member).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupError(f"备份中的 {member} 无法解析") from exc
    if not isinstance(data, list):
        raise BackupError(f"备份中的 {member} 不是列表")
    return data


def _extract_uploads(archive: zipfile.ZipFile, uploads_root: Path) -> int:
    """把 uploads/ 下的文件还原到数据目录，返回文件数。"""
    count = 0
    uploads_root.mkdir(parents=True, exist_ok=True)
    for member in archive.namelist():
        if not member.startswith(f"{UPLOADS_DIR}/") or member.endswith("/"):
            continue
        relative = member[len(UPLOADS_DIR) + 1 :]
        # 防止 zip 内路径穿越（../ 逃出 uploads 目录）
        destination = (uploads_root / relative).resolve()
        if uploads_root.resolve() not in destination.parents:
            raise BackupError(f"备份包含越界的文件路径：{member}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(member))
        count += 1
    return count


def _index_uploaded_files(uploads_root: Path) -> dict[int, Path]:
    """把 uploads 下的文件按「文件名前缀 = doc_id」建索引。

    文件名约定为 `{doc_id}_{原文件名}`（见 `ui.documents.stored_path`），
    这里只解析前缀，不复制那套命名规则，避免两处实现漂移。
    """
    index: dict[int, Path] = {}
    if not uploads_root.exists():
        return index
    for path in uploads_root.rglob("*"):
        if not path.is_file():
            continue
        head = path.name.split("_", 1)[0]
        if head.isdigit():
            index[int(head)] = path
    return index


def _reindex_documents(settings, store: VectorStore, result: RestoreResult) -> None:
    """按恢复出来的 documents 记录重建向量（仅 active 且有原始文件的）。"""
    rows = export_tables(settings.database_path)["documents"]
    file_index = _index_uploaded_files(settings.uploads_path)

    for row in rows:
        if row.get("status") == DOC_DELETED:
            continue
        doc_id = int(row["id"])
        filename = row.get("filename") or ""
        path = file_index.get(doc_id)
        if path is None:
            result.missing_files.append(filename or f"doc-{doc_id}")
            logger.warning("文档 %s（%s）缺少原始文件，跳过重向量化", doc_id, filename)
            continue

        try:
            loaded = load_document(path)
        except Exception as exc:  # 解析失败不该中断整库恢复
            result.missing_files.append(filename or f"doc-{doc_id}")
            logger.warning("文档 %s（%s）解析失败，跳过：%s", doc_id, filename, exc)
            continue

        text = "\n\n".join(section.text for section in loaded.sections)
        chunks = split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        if not chunks:
            result.missing_files.append(filename or f"doc-{doc_id}")
            continue

        # 幂等：先清掉该文档的旧向量，再写入
        store.delete_by_doc_id(doc_id)
        pairs = [
            (
                chunk.text,
                build_metadata(
                    doc_id=doc_id,
                    user_id=row.get("user_id"),
                    is_public=bool(row.get("is_public")),
                    category=row.get("category") or UNCATEGORIZED_KEY,
                    filename=filename,
                    chunk_index=chunk.chunk_index,
                ),
            )
            for chunk in chunks
        ]
        for start in range(0, len(pairs), EMBED_BATCH_SIZE):
            store.add_chunks(pairs[start : start + EMBED_BATCH_SIZE])

        result.reindexed_docs += 1
        result.chunk_total += len(pairs)
        _update_chunk_count(settings.database_path, doc_id, len(pairs))


def _update_chunk_count(db_path, doc_id: int, count: int) -> None:
    with get_conn(db_path) as conn:
        conn.execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (count, doc_id))
