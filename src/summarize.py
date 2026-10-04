"""课件总结 / 复习提纲生成（二期 2.4）。

设计依据：docs/02 v1.15 §4.15；客户 2026-10-04 裁决：
① 入口在「我的文档」页的「课件助手」区块；
② 产物落库保存（`summaries` 表，同一文档同一产物只保留最新一份）；
③ 课件总结与复习提纲两种都做。

长文处理（map-reduce）：课件可能远超单次上下文，因此先把该文档的切片分批生成
「分段摘要」，再归并成最终产物。只有一批时直接一次生成，省掉一次归并调用。

LLM 可注入；分批、提示词拼装、归并均为纯逻辑，可直接单元测试。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import Settings, get_settings
from src.errors import SummarizeError
from src.providers.llm import get_llm, response_text
from src.store.chroma import SearchHit

logger = logging.getLogger(__name__)

KIND_SUMMARY = "summary"
KIND_OUTLINE = "outline"
KIND_LABELS = {KIND_SUMMARY: "课件总结", KIND_OUTLINE: "复习提纲"}

SYSTEM_PROMPT = (
    "你是校园课件整理助手。只依据给定的课件内容提炼与归纳，"
    "不补充原文没有的信息，不编造公式、数据或结论。"
)

BATCH_TEMPLATE = """以下是课件《{filename}》的第 {index}/{total} 部分内容。请提炼这一部分的核心要点，\
条目化，保留关键概念、定义、公式与结论；不要添加原文没有的内容。

{body}"""

DIRECT_TEMPLATES = {
    KIND_SUMMARY: """请阅读课件《{filename}》的完整内容，写一份**内容总结**：先用 2~3 句话总述这份课件讲什么，\
再分点列出主要章节与核心内容。条目清晰、不啰嗦。

{body}""",
    KIND_OUTLINE: """请阅读课件《{filename}》的完整内容，写一份**复习提纲**：按知识点组织成层级大纲\
（用「一、二、」与「-」缩进表示层级），标出重点与易考点，条目简洁、便于背诵。

{body}""",
}

REDUCE_TEMPLATES = {
    KIND_SUMMARY: """以下是课件《{filename}》各部分的分段摘要。请把它们整合成一份**内容总结**：\
先总述，再分点覆盖主要内容；去掉重复、保持逻辑连贯，不要遗漏分段摘要中的要点。

{digests}""",
    KIND_OUTLINE: """以下是课件《{filename}》各部分的分段摘要。请把它们整合成一份**复习提纲**：\
按知识点组织成层级大纲，标出重点；去掉重复、保持层级清晰。

{digests}""",
}


def usable_chunks(hits: Iterable[SearchHit]) -> list[str]:
    """取出有实际文字的切片正文（空切片不参与生成）。"""
    return [hit.text.strip() for hit in hits if hit.text and hit.text.strip()]


def chunk_batches(texts: Sequence[str], size: int) -> list[list[str]]:
    """把切片正文按 `size` 分批。`size` 小于 1 时按 1 处理。"""
    step = max(int(size), 1)
    return [list(texts[index : index + step]) for index in range(0, len(texts), step)]


def _batch_prompt(filename: str, index: int, total: int, body: str) -> str:
    return (
        BATCH_TEMPLATE.replace("{filename}", filename)
        .replace("{index}", str(index))
        .replace("{total}", str(total))
        .replace("{body}", body)
    )


def _direct_prompt(filename: str, kind: str, body: str) -> str:
    return DIRECT_TEMPLATES[kind].replace("{filename}", filename).replace("{body}", body)


def _reduce_prompt(filename: str, kind: str, digests: str) -> str:
    return REDUCE_TEMPLATES[kind].replace("{filename}", filename).replace("{digests}", digests)


def _invoke(model, prompt: str) -> str:
    return response_text(
        model.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
    )


def generate(
    hits: Iterable[SearchHit],
    *,
    filename: str,
    kind: str,
    llm=None,
    settings: Settings | None = None,
) -> str:
    """生成一份课件总结或复习提纲，返回 Markdown 正文。

    Raises:
        SummarizeError: 产物类型未知、没有可用文本，或模型调用失败 / 返回空。
    """
    s = settings or get_settings()
    if kind not in KIND_LABELS:
        raise SummarizeError(f"未知的产物类型：{kind}")

    texts = usable_chunks(hits)
    if not texts:
        raise SummarizeError("该文档没有可用于生成的文本内容")

    batches = chunk_batches(texts, s.summarize_batch_chunks)
    model = llm or get_llm(s, streaming=False)

    try:
        if len(batches) == 1:
            final = _invoke(model, _direct_prompt(filename, kind, "\n\n".join(batches[0])))
        else:
            digests = [
                _invoke(
                    model,
                    _batch_prompt(filename, index, len(batches), "\n\n".join(batch)),
                )
                for index, batch in enumerate(batches, start=1)
            ]
            final = _invoke(model, _reduce_prompt(filename, kind, "\n\n---\n\n".join(digests)))
    except Exception as exc:
        # 不把内部异常细节透给用户（docs/02 §11），只记日志
        logger.exception("课件生成失败：file=%s kind=%s", filename, kind)
        raise SummarizeError("生成失败，请稍后重试") from exc

    text = (final or "").strip()
    if not text:
        raise SummarizeError("生成失败：模型没有返回内容，请重试")
    return text
