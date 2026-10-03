"""备份恢复 CLI（FR-24，二期）。

设计依据：docs/02-架构设计.md §4.9。

恢复是**整库替换**：会清空 users / documents / conversations / messages /
message_sources / feedback 六张表，再写回备份内容，最后按仍为 active 的文档
重新向量化。目标库非空时必须加 `--force`。

用法：
    python scripts/restore.py backups/campus-qa-20261004-120000.zip --force
    python scripts/restore.py backup.zip --no-reindex      # 只恢复记录与文件
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

# 直接以 `python scripts/xxx.py` 运行时，sys.path[0] 是 scripts/ 而非项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings  # noqa: E402
from src.backup import restore_backup  # noqa: E402
from src.errors import CampusQAError  # noqa: E402
from src.store.chroma import VectorStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校答（campus-qa）备份恢复")
    parser.add_argument("zip_path", help="备份 zip 路径")
    parser.add_argument(
        "--force", action="store_true", help="目标库非空时允许覆盖（整库替换，谨慎）"
    )
    parser.add_argument(
        "--no-reindex", action="store_true", help="只恢复记录与文件，不重建向量"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    # 预检放在建 VectorStore 之前：本地 BGE 加载要数秒，备份文件本身就有问题
    # 时不该白加载一遍模型
    if not Path(args.zip_path).exists():
        print(f"恢复失败：备份文件不存在：{args.zip_path}")
        return 1
    if not zipfile.is_zipfile(args.zip_path):
        print(f"恢复失败：备份文件已损坏或不是 zip：{args.zip_path}")
        return 1

    settings = get_settings()
    reindex = not args.no_reindex
    store = VectorStore(settings) if reindex else _NullStore()

    try:
        result = restore_backup(
            settings=settings,
            zip_path=args.zip_path,
            store=store,
            force=args.force,
            reindex=reindex,
        )
    except CampusQAError as exc:
        print(f"恢复失败：{exc}")
        return 1

    print("恢复完成：")
    for name, count in result.table_counts.items():
        print(f"  {name}: {count} 行")
    print(f"  原始文件：{result.file_count} 个")
    if reindex:
        print(f"  重新向量化：{result.reindexed_docs} 篇 / {result.chunk_total} 片")
    if result.missing_files:
        print(f"  跳过（缺原始文件或解析失败）：{len(result.missing_files)} 篇")
        for name in result.missing_files:
            print(f"    - {name}")
    print(f"\n知识库配置快照已存到 {settings.data_dir / 'knowledge_base.yaml.restored'}，")
    print("请与 config/knowledge_base.yaml 比对后手动替换。")
    return 0


class _NullStore:
    """`--no-reindex` 时的占位：任何写入都直接拒绝。"""

    def delete_by_doc_id(self, doc_id: int) -> int:  # pragma: no cover - 不应被调用
        raise AssertionError("--no-reindex 时不应触碰向量库")

    def add_chunks(self, chunks) -> int:  # pragma: no cover - 不应被调用
        raise AssertionError("--no-reindex 时不应触碰向量库")


if __name__ == "__main__":
    raise SystemExit(main())
