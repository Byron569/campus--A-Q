"""批量入库 + 检索验证 CLI。

设计依据：docs/03-开发任务清单.md §1.6（M1-17）

定位（RV-07 裁决）：这是 **M1 的最小检索验证工具**。
面向管理员的完整批量导入脚本（参数校验、进度输出、失败汇总）为 Should 级别，
归 M3 的 FR-30 完善，此处只做「能入库、能检索到」所需的最小实现。

检索只走**向量路**（`VectorStore.search`）：M1 尚未实现 BM25 混合检索（M2 的
`rag.retriever`），此处用于验证「切片 → 向量 → 召回」这条主链是否通。

用法：
    python scripts/ingest_cli.py data/samples/freshman --public --category freshman
    python scripts/ingest_cli.py data/samples --public --category admin --user-id 1
    python scripts/ingest_cli.py --query "搬宿舍需要提前申请吗"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 直接以 `python scripts/xxx.py` 运行时，sys.path[0] 是 scripts/ 而非项目根，
# 因此需要手动把项目根加进来才能 import config / src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (  # noqa: E402
    ALLOWED_SUFFIXES,
    UNCATEGORIZED_KEY,
    category_options,
    get_settings,
)
from src.errors import CampusQAError  # noqa: E402
from src.ingest.pipeline import ingest_file  # noqa: E402
from src.store.chroma import VectorStore  # noqa: E402
from src.store.db import init_db  # noqa: E402

logger = logging.getLogger("scripts.ingest_cli")

SNIPPET_LENGTH = 120


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="校答（campus-qa）批量入库与检索验证工具（M1 最小版）",
    )
    parser.add_argument(
        "paths", nargs="*", help="待入库的文件或目录（目录会递归收集白名单内的文件）",
    )
    parser.add_argument("--public", action="store_true", help="作为公共文档入库，全员可见")
    parser.add_argument(
        "--category",
        default=None,
        help=f"分类 key；入库时不填归入「{UNCATEGORIZED_KEY}」，检索时不填表示不限分类",
    )
    parser.add_argument(
        "--user-id", type=int, default=None, help="个人资料的归属用户 id；入库个人资料时必填",
    )
    parser.add_argument("--query", default=None, help="检索问句；只给该参数时不入库，仅检索")
    parser.add_argument("--top-k", type=int, default=None, help="检索返回条数，默认取配置")
    return parser


def collect_files(targets: list[str], skipped: list[tuple[Path, str]]) -> list[Path]:
    """收集待入库文件。目录递归展开，非白名单后缀与不存在的路径记入 skipped。"""
    files: list[Path] = []
    for raw in targets:
        path = Path(raw)
        if not path.exists():
            skipped.append((path, "路径不存在"))
            continue

        candidates = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.suffix.lower() in ALLOWED_SUFFIXES:
                files.append(candidate)
            else:
                skipped.append(
                    (candidate, f"不支持的文件类型：{candidate.suffix or candidate.name}")
                )
    return files


def ingest_all(
    files: list[Path], *, category: str, is_public: bool, user_id: int | None, store: VectorStore
) -> tuple[int, int]:
    """逐个同步入库。返回 (成功数, 失败数)。

    单个文件失败只打印原因并继续，不中断整批（docs/02 §11 统一原则 1）。
    """
    succeeded = failed = 0
    for path in files:
        try:
            result = ingest_file(
                path, category=category, is_public=is_public, user_id=user_id, store=store,
            )
        except CampusQAError as exc:
            print(f"[失败] {path}：{exc}")
            failed += 1
            continue

        succeeded += 1
        suffix = f"，告警：{result.warning}" if result.warning else ""
        print(
            f"[完成] {path} → {result.chunk_count} 个切片，文档 id={result.doc_id}，"
            f"耗时 {result.elapsed_ms}ms{suffix}"
        )
    return succeeded, failed


def run_query(
    store: VectorStore,
    question: str,
    *,
    user_id: int | None,
    category: str | None,
    top_k: int | None,
) -> int:
    """检索并打印片段与分数。返回命中条数。"""
    hits = store.search(question, user_id=user_id, category=category, k=top_k)
    if not hits:
        print("未检索到任何片段（可能相关文档还没入库，或相似度低于阈值）。")
        return 0

    print(f"检索：{question}")
    for index, hit in enumerate(hits, start=1):
        snippet = " ".join(hit.text.split())
        if len(snippet) > SNIPPET_LENGTH:
            snippet = snippet[:SNIPPET_LENGTH] + "…"
        score = f"{hit.score:.4f}" if hit.score is not None else "—"
        print(
            f"【来源{index}】{hit.filename}（第 {hit.chunk_index} 片，分类 {hit.category}，"
            f"相似度 {score}）"
        )
        print(f"         {snippet}")
    print("\n注：M1 只验证向量路；BM25 混合检索在 M2 的 rag.retriever 实现。")
    return len(hits)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("src").setLevel(logging.INFO)

    args = build_parser().parse_args(argv)
    settings = get_settings()

    if not args.paths and not args.query:
        print("参数错误：至少给出一个待入库路径，或用 --query 指定检索问句。")
        return 2

    valid_categories = {option["key"] for option in category_options()}
    if args.category is not None and args.category not in valid_categories:
        print(
            f"参数错误：未知分类 `{args.category}`。可选：{'、'.join(sorted(valid_categories))}"
        )
        return 2

    if args.paths and not args.public and args.user_id is None:
        print("参数错误：入库个人资料必须用 --user-id 指定归属用户；公共文档请加 --public。")
        return 2

    ingest_category = args.category or UNCATEGORIZED_KEY

    settings.ensure_dirs()
    init_db()

    # 整批复用同一个 VectorStore：本地 BGE 加载一次要数秒，逐文件重建代价很高
    store = VectorStore(settings)

    failed = 0
    if args.paths:
        skipped: list[tuple[Path, str]] = []
        files = collect_files(args.paths, skipped)
        for path, reason in skipped:
            print(f"[跳过] {path}：{reason}")

        if files:
            print(f"待入库 {len(files)} 个文件，分类 {ingest_category}，"
                  f"{'公共文档' if args.public else f'用户 {args.user_id} 的个人资料'}。\n")
            succeeded, failed = ingest_all(
                files,
                category=ingest_category,
                is_public=args.public,
                user_id=None if args.public else args.user_id,
                store=store,
            )
            print(f"\n汇总：成功 {succeeded} 个，失败 {failed} 个，跳过 {len(skipped)} 个。")
        elif not args.query:
            print("没有可入库的文件。")
            return 1

    if args.query:
        print()
        run_query(store, args.query, user_id=args.user_id, category=args.category, top_k=args.top_k)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
