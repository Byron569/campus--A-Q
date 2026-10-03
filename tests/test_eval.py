"""评测脚本单元测试（M3-11）。

覆盖 docs/03 §6.1 TC-U22：评测脚本能跑通并输出四项指标，格式正确。
测试用替身检索器与替身模型驱动，不联网、不调真实大模型。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from config.settings import Settings
from scripts.run_eval import EvalReport, ItemRecord, TurnRecord, evaluate, format_report
from src.store.chroma import SearchHit


class RoutingRetriever:
    """按问题里的关键词返回预设来源；都不匹配则返回空（触发拒答）。"""

    def __init__(self, routes: dict[str, str]) -> None:
        self.routes = routes
        self.queries: list[str] = []

    def search(self, query: str) -> list[SearchHit]:
        self.queries.append(query)
        for keyword, filename in self.routes.items():
            if keyword in query:
                return [
                    SearchHit(
                        text=f"{filename} 的片段",
                        score=0.9,
                        filename=filename,
                        doc_id=1,
                        chunk_index=0,
                        category="freshman",
                        matched_by="vector",
                    )
                ]
        return []


class StubLLM:
    """按队列返回内容，模拟改写与生成两类调用。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def invoke(self, messages):
        return SimpleNamespace(content=self.responses.pop(0) if self.responses else "")


def test_tc_u22_eval_script_runs_and_outputs_four_metrics(db: Path, settings: Settings) -> None:
    """TC-U22：脚本跑通，四项指标均为百分比，格式正确。"""
    items = [
        {
            "id": "a1",
            "type": "answerable",
            "question": "图书馆开放时间",
            "expected_sources": ["图书馆借阅规则.md"],
            "expected_keywords": ["8:00"],
        },
        {"id": "r1", "type": "unanswerable", "question": "宇宙黑洞是什么"},
        {
            "id": "m1",
            "type": "multi_turn",
            "turns": [
                {
                    "question": "图书馆的规则",
                    "expected_sources": ["图书馆借阅规则.md"],
                    "expected_keywords": ["8:00"],
                },
                {
                    "question": "那能借几本？",
                    "expected_sources": ["图书馆借阅规则.md"],
                    "expected_keywords": ["10册"],
                },
            ],
        },
    ]
    # 调用顺序：a1 生成 → m1 首轮生成 → m1 追问改写 → m1 追问生成
    llm = StubLLM(
        [
            "开放时间为 8:00 至 22:00【来源1】",
            "开放时间为 8:00 至 22:00【来源1】",
            "本科生一次能借几本书 图书馆",
            "本科生可借 10 册【来源1】",
        ]
    )
    retriever = RoutingRetriever({"图书馆": "图书馆借阅规则.md"})

    report = evaluate(
        items, db_path=db, retriever=retriever, llm=llm, settings=settings
    )
    text = format_report(report)

    assert report.accuracy == 1.0
    assert report.citation_hit_rate == 1.0
    assert report.refusal_rate == 1.0
    assert report.rewrite_success_rate == 1.0
    for label in ("准确率", "引用命中率", "拒答正确率", "追问改写成功率"):
        assert label in text
    assert text.count("%") == 4


def test_eval_report_counts_misses_correctly() -> None:
    """错答 / 引用跑偏 / 漏拒答 / 改写失败都要如实反映在分母与分子上。"""
    report = EvalReport(
        records=[
            ItemRecord(
                "a1",
                "answerable",
                [TurnRecord("q", ["甲.md"], False, False, ["甲.md"], True)],
            ),
            ItemRecord(
                "a2",
                "answerable",
                [TurnRecord("q", ["乙.md"], False, False, ["丙.md"], False)],
            ),
            ItemRecord(
                "r1", "unanswerable", [TurnRecord("q", [], False, False, ["丁.md"], True)]
            ),
            ItemRecord(
                "m1",
                "multi_turn",
                [
                    TurnRecord("q1", ["甲.md"], False, False, ["甲.md"], True),
                    TurnRecord("q2", ["乙.md"], False, False, ["丙.md"], True),
                ],
            ),
        ]
    )

    # 答对：a1、m1（末轮要点命中）→ 2/4
    assert report.accuracy == 0.5
    # 可回答题 2 道，命中 1 道
    assert report.citation_hit_rate == 0.5
    # 库外题 1 道，正确拒答 0 道
    assert report.refusal_rate == 0.0
    # 追问 1 轮，命中 0 轮
    assert report.rewrite_success_rate == 0.0
    assert len(report.failures()) == 3
