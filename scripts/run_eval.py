"""评测脚本：跑 30 题评测集，输出四项指标（FR-32 / M3-11）。

设计依据：docs/01 §3.1 FR-32、docs/03 §3 M3-10 / M3-11、docs/02 §9 可观测性。

评测集（`eval/campus_qa_eval.jsonl`）每行一题，三种类型：

- `answerable`  可回答：知识库内有明确答案，字段 `expected_sources`（期望来源文件名）
  与 `expected_keywords`（期望答案要点）。
- `unanswerable` 不可回答：知识库外问题，应被拒答。
- `multi_turn`  多轮追问：`turns` 数组按顺序在同一会话内提问，后一轮带指代，
  用来检验 Query 改写是否生效。

四项指标的定义（docs/01 只给了名称，未给公式，此处明确，见自审清单「需客户决策」）：

| 指标 | 定义 |
| --- | --- |
| 准确率 | 答对题数 / 总题数。可回答题（含多轮末轮）＝未拒答且未降级且答案含全部期望要点；不可回答题＝正确拒答 |
| 引用命中率 | 可回答单选 20 题中，回答引用的来源文件命中期望文档的比例 |
| 拒答正确率 | 5 道库外题中，`refused=True` 的比例 |
| 追问改写成功率 | 多轮题的后一轮中，引用来源命中期望文档的比例（改写失效则检索会跑偏） |

**运行会真的调用大模型**（改写与生成），需要 `.env` 中配置有效的 `LLM_API_KEY`。
会话与指标写入一个临时库，跑完即删，**不碰真实 `data/app.db`**。

用法：
    HF_ENDPOINT=https://hf-mirror.com python scripts/run_eval.py
    python scripts/run_eval.py --eval-file eval/campus_qa_eval.jsonl --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 直接以 `python scripts/xxx.py` 运行时，sys.path[0] 是 scripts/ 而非项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import Settings, get_settings  # noqa: E402
from src.rag.chain import answer  # noqa: E402
from src.rag.retriever import build_retriever  # noqa: E402
from src.repository import create_conversation, create_user, get_user_by_name  # noqa: E402
from src.store.chroma import VectorStore  # noqa: E402
from src.store.db import init_db  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_FILE = PROJECT_ROOT / "eval" / "campus_qa_eval.jsonl"
EVAL_USERNAME = "eval-runner"


# ==================== 数据结构 ====================


@dataclass
class TurnRecord:
    question: str
    expected_sources: list[str]
    refused: bool
    degraded: bool
    cited: list[str]
    keyword_ok: bool

    @property
    def source_hit(self) -> bool:
        return any(name in self.cited for name in self.expected_sources)


@dataclass
class ItemRecord:
    item_id: str
    kind: str
    turns: list[TurnRecord] = field(default_factory=list)

    @property
    def last(self) -> TurnRecord:
        return self.turns[-1]


@dataclass
class EvalReport:
    records: list[ItemRecord]

    def _by_kind(self, kind: str) -> list[ItemRecord]:
        return [r for r in self.records if r.kind == kind]

    # ---- 四项指标 ----

    @property
    def accuracy(self) -> float:
        return _ratio(self._correct_count(), len(self.records))

    @property
    def citation_hit_rate(self) -> float:
        items = self._by_kind("answerable")
        return _ratio(sum(r.last.source_hit for r in items), len(items))

    @property
    def refusal_rate(self) -> float:
        items = self._by_kind("unanswerable")
        return _ratio(sum(r.last.refused for r in items), len(items))

    @property
    def rewrite_success_rate(self) -> float:
        follow_ups = [turn for r in self._by_kind("multi_turn") for turn in r.turns[1:]]
        return _ratio(sum(t.source_hit for t in follow_ups), len(follow_ups))

    # ---- 明细统计 ----

    def _correct_count(self) -> int:
        correct = 0
        for record in self.records:
            if record.kind == "unanswerable":
                correct += int(record.last.refused)
            else:
                last = record.last
                correct += int(not last.refused and not last.degraded and last.keyword_ok)
        return correct

    def summary(self) -> dict[str, object]:
        total = len(self.records)
        return {
            "total": total,
            "correct": self._correct_count(),
            "accuracy": self.accuracy,
            "citation_hit_rate": self.citation_hit_rate,
            "refusal_rate": self.refusal_rate,
            "rewrite_success_rate": self.rewrite_success_rate,
        }

    def failures(self) -> list[str]:
        """列出未达预期的题目，便于定位。"""
        problems: list[str] = []
        for record in self.records:
            if record.kind == "unanswerable":
                if not record.last.refused:
                    problems.append(f"{record.item_id} 未拒答：引用 {record.last.cited}")
                continue
            last = record.last
            if last.refused:
                problems.append(f"{record.item_id} 被误拒答")
            elif last.degraded:
                problems.append(f"{record.item_id} 发生降级（模型未调用成功）")
            elif not last.keyword_ok:
                problems.append(f"{record.item_id} 答案缺少期望要点：{last.expected_sources}")
            elif not last.source_hit:
                problems.append(f"{record.item_id} 引用未命中期望来源：{last.cited}")
        return problems


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


# ==================== 执行 ====================


def load_items(path: Path) -> list[dict]:
    """读取 JSONL 评测集，逐行解析并做基本校验。"""
    items: list[dict] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"评测集第 {line_number} 行不是合法 JSON：{exc}") from exc
        if item.get("type") not in {"answerable", "unanswerable", "multi_turn"}:
            raise ValueError(f"评测集第 {line_number} 行 type 非法：{item.get('type')!r}")
        items.append(item)
    return items


def _contains_all(text: str, keywords: list[str]) -> bool:
    """忽略空白后逐个做子串匹配——模型可能写成「9 月 1 日」或「9月1日」。"""
    normalized = "".join(text.split())
    return all(keyword.replace(" ", "") in normalized for keyword in keywords)


def _ensure_eval_user(db_path) -> int:
    existing = get_user_by_name(EVAL_USERNAME, db_path=db_path)
    if existing is not None:
        return existing.id
    return create_user(
        username=EVAL_USERNAME, password_hash="*eval*", display_name="评测", db_path=db_path
    )


def evaluate(
    items: list[dict],
    *,
    db_path,
    store: VectorStore | None = None,
    retriever=None,
    llm=None,
    settings: Settings | None = None,
) -> EvalReport:
    """在同一临时库里按题执行问答，汇总为 EvalReport。

    `retriever` / `llm` 可注入替身，测试时无需真实模型；生产环境按配置构造。
    """
    s = settings or get_settings()
    init_db(db_path)
    user_id = _ensure_eval_user(db_path)
    if retriever is None:
        retriever = build_retriever(user_id, store=store, settings=s)

    records: list[ItemRecord] = []
    for item in items:
        conversation_id = create_conversation(user_id, db_path=db_path)
        turns = item["turns"] if item["type"] == "multi_turn" else [item]
        record = ItemRecord(item_id=item["id"], kind=item["type"])
        for turn in turns:
            result = answer(
                turn["question"],
                user_id,
                conversation_id,
                retriever=retriever,
                llm=llm,
                settings=s,
                db_path=db_path,
            )
            record.turns.append(
                TurnRecord(
                    question=turn["question"],
                    expected_sources=turn.get("expected_sources", []),
                    refused=result.refused,
                    degraded=result.degraded,
                    cited=[hit.filename for hit in result.sources],
                    keyword_ok=_contains_all(result.text, turn.get("expected_keywords", [])),
                )
            )
        records.append(record)
    return EvalReport(records=records)


def format_report(report: EvalReport) -> str:
    """把指标渲染成固定格式的文本，四个百分比各一行。"""
    summary = report.summary()
    lines = [
        "===== 校答评测结果 =====",
        f"题目总数    ：{summary['total']}",
        f"准确率      ：{summary['accuracy'] * 100:.1f}%  "
        f"（答对 {summary['correct']}/{summary['total']}）",
        f"引用命中率  ：{summary['citation_hit_rate'] * 100:.1f}%",
        f"拒答正确率  ：{summary['refusal_rate'] * 100:.1f}%",
        f"追问改写成功率：{summary['rewrite_success_rate'] * 100:.1f}%",
    ]
    return "\n".join(lines)


# ==================== CLI ====================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校答（campus-qa）30 题评测脚本（FR-32）")
    parser.add_argument(
        "--eval-file", default=str(DEFAULT_EVAL_FILE), help="评测集 JSONL 路径"
    )
    parser.add_argument("--verbose", action="store_true", help="逐条打印未达预期的题目")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出汇总指标")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    settings.ensure_dirs()

    eval_file = Path(args.eval_file)
    if not eval_file.exists():
        print(f"评测集不存在：{eval_file}")
        return 2
    items = load_items(eval_file)

    # 评测写入独立临时库，跑完删除，真实 data/app.db 不受影响
    eval_db = settings.data_dir / "eval_run.db"
    for suffix in ("", "-wal", "-shm"):
        Path(str(eval_db) + suffix).unlink(missing_ok=True)

    store = VectorStore(settings)
    try:
        report = evaluate(items, db_path=eval_db, store=store, settings=settings)
    finally:
        for suffix in ("", "-wal", "-shm"):
            Path(str(eval_db) + suffix).unlink(missing_ok=True)

    if args.json:
        print(json.dumps(report.summary(), ensure_ascii=False, indent=2))
    else:
        print(format_report(report))

    problems = report.failures()
    if problems:
        print(f"\n未达预期 {len(problems)} 条：")
        for problem in problems:
            print(f"  - {problem}")
        if not args.verbose:
            print("（加 --verbose 查看逐题明细；以上已是全部未达预期项）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
