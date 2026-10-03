"""Query 改写单元测试（M2-03）。

覆盖 docs/02 §4.4 的三条退化路径：无历史 / 调用失败 / 返回空，都必须回到原问题。
"""

from __future__ import annotations

from types import SimpleNamespace

from config.settings import Settings
from src.rag.rewrite import format_history, rewrite_query
from src.repository import Message


class StubLLM:
    """记录调用并按队列返回内容；exc 非空时模拟调用失败。"""

    def __init__(self, responses: list[str] | None = None, exc: Exception | None = None) -> None:
        self.responses = list(responses or [])
        self.exc = exc
        self.calls: list[object] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(content=self.responses.pop(0) if self.responses else "")


def msg(role: str, content: str, *, msg_id: int = 1, conversation_id: int = 1) -> Message:
    return Message(
        id=msg_id, conversation_id=conversation_id, role=role,
        content=content, created_at="2026-10-03 10:00:00",
    )


HISTORY = [msg("user", "搬宿舍要怎么弄", msg_id=1), msg("assistant", "需提前三个工作日申请", msg_id=2)]


def test_no_history_returns_original_without_calling_llm(settings: Settings) -> None:
    llm = StubLLM(["不该被使用"])

    assert rewrite_query("它怎么申请", [], llm=llm, settings=settings) == "它怎么申请"
    assert llm.calls == []


def test_blank_question_returns_blank(settings: Settings) -> None:
    assert rewrite_query("   ", HISTORY, llm=StubLLM(["x"]), settings=settings) == ""


def test_history_rewrites_question(settings: Settings) -> None:
    llm = StubLLM(["  搬宿舍需要提前几天申请？  "])

    result = rewrite_query("提前几天", HISTORY, llm=llm, settings=settings)

    assert result == "搬宿舍需要提前几天申请？"
    assert len(llm.calls) == 1
    assert "搬宿舍要怎么弄" in llm.calls[0]  # 历史被拼进了提示词


def test_llm_failure_falls_back_to_original(settings: Settings) -> None:
    llm = StubLLM(exc=RuntimeError("timeout"))

    assert rewrite_query("提前几天", HISTORY, llm=llm, settings=settings) == "提前几天"


def test_empty_rewrite_falls_back_to_original(settings: Settings) -> None:
    assert rewrite_query("提前几天", HISTORY, llm=StubLLM(["   "]), settings=settings) == "提前几天"


def test_format_history_labels_speakers_and_keeps_order() -> None:
    text = format_history(HISTORY, limit=10)

    assert text.splitlines() == ["用户：搬宿舍要怎么弄", "助手：需提前三个工作日申请"]


def test_format_history_keeps_only_recent() -> None:
    text = format_history(HISTORY, limit=1)

    assert text == "助手：需提前三个工作日申请"
