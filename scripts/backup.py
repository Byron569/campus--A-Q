"""备份导出 CLI（FR-24，二期）。

设计依据：docs/02-架构设计.md §4.9。

用法：
    python scripts/backup.py                                   # 输出到 backups/campus-qa-<时间>.zip
    python scripts/backup.py --out /path/to/backup.zip
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 直接以 `python scripts/xxx.py` 运行时，sys.path[0] 是 scripts/ 而非项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings  # noqa: E402
from src.backup import create_backup  # noqa: E402
from src.errors import CampusQAError  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校答（campus-qa）备份导出")
    parser.add_argument(
        "--out", default=None, help="输出 zip 路径，默认为 backups/campus-qa-<时间>.zip"
    )
    return parser


def default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "backups" / f"campus-qa-{stamp}.zip"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()
    out_path = Path(args.out) if args.out else default_output()

    try:
        manifest = create_backup(settings=settings, out_path=out_path)
    except CampusQAError as exc:
        print(f"备份失败：{exc}")
        return 1

    print(f"备份已生成：{out_path}")
    print(f"  导出时间：{manifest.created_at}")
    print(f"  原始文件：{manifest.file_count} 个")
    for name, count in manifest.table_counts.items():
        print(f"  {name}: {count} 行")
    print("\n提示：备份不含 .env 与向量库；恢复后会自动重新向量化。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
