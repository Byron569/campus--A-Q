"""防幻觉提示词、引用格式与拒答文案。

设计依据：
- docs/02-架构设计.md §4.5（生成与防幻觉：三条硬约束 + `【来源N】` + 越界剔除 DR-11）
- docs/02-架构设计.md §4.10（拒答文案中的联系方式来自 knowledge_base.yaml 的 school.contact）
- docs/06-接口文档.md §1.6（`SYSTEM_PROMPT` / `build_context`）

三条硬约束是本模块的核心，改动前必须对照 docs/02 §4.5：
1. 只能依据检索到的资料回答，不得用自身知识补充学校政策；
2. 每个结论后标注来源，格式 `【来源N】`；
3. 资料与问题无关时，直接输出固定拒答文案，禁止猜测。
"""

from __future__ import annotations

import re
from typing import Iterable

from config.settings import load_kb_config
from src.errors import CampusQAError
from src.store.chroma import SearchHit

# 学校联系方式取不到时的兜底称谓
DEFAULT_CONTACT = "学校相关部门"

# 检索结果与问题无关时的固定文案（FR-10）。`{contact}` 由 build_refusal_text 填充。
REFUSAL_TEMPLATE = "知识库中暂无相关资料，建议咨询{contact}。"

# 引用标记格式：`【来源N】`，N 为本次上下文中的资料编号（从 1 开始）。
# 越界编号（超出实际提供的资料条数）直接剔除，绝不伪装成有效来源（DR-11）。
SOURCE_PATTERN = re.compile(r"【来源(\d+)】")

# `{contact}` 用 replace 填充而非 str.format：提示词正文里出现花括号不会引发格式化错误。
SYSTEM_PROMPT = """你是校园知识库问答助手。你必须严格依据下方【资料】回答用户问题，遵守三条硬约束：

1. 只能依据【资料】回答，不得使用你自己的知识补充或推测学校政策、流程、时间、地点与联系方式。资料中没有的，一律视为没有。
2. 每个结论后必须标注来源编号，格式为「【来源N】」，N 为对应资料的编号；一句话可由多个来源支撑时并列标注，如「【来源1】【来源3】」。禁止标注资料中不存在的编号。
3. 若【资料】与问题无关或不足以回答，必须只回答「知识库中暂无相关资料，建议咨询{contact}」，禁止猜测、禁止编造、禁止用常识补充。

回答要求：使用简体中文，先给结论再给依据，条目清晰、简洁。除上述拒答情形外，不要输出与问题无关的内容。"""


def _resolve_contact(contact: str | None = None) -> str:
    """取拒答文案里的咨询对象：入参优先，其次配置，最后兜底称谓。"""
    if contact and contact.strip():
        return contact.strip()
    try:
        value = str(load_kb_config()["school"].get("contact", "")).strip()
    except CampusQAError:  # 配置缺失不应阻断拒答，降级为兜底称谓
        value = ""
    return value or DEFAULT_CONTACT


def build_system_prompt(contact: str | None = None) -> str:
    """生成系统提示词：把 `{contact}` 替换为实际联系方式。"""
    return SYSTEM_PROMPT.replace("{contact}", _resolve_contact(contact))


def build_refusal_text(contact: str | None = None) -> str:
    """未命中检索时直接输出的固定文案（FR-10，不调用 LLM）。"""
    return REFUSAL_TEMPLATE.replace("{contact}", _resolve_contact(contact))


def build_context(hits: Iterable[SearchHit]) -> str:
    """把命中的切片按 `【来源N】` 编号拼成模型上下文。

    编号顺序即 `hits` 顺序，且与上层用于渲染引用卡片的顺序一致——
    两处顺序必须相同，否则 `【来源N】` 会指向错误的文档。
    """
    blocks = [
        f"【来源{index}】{hit.filename}\n{hit.text}"
        for index, hit in enumerate(hits, start=1)
    ]
    return "\n\n".join(blocks)


def parse_source_numbers(text: str) -> list[int]:
    """按出现顺序提取 `【来源N】` 中的编号（去重）。"""
    numbers: list[int] = []
    for match in SOURCE_PATTERN.finditer(text or ""):
        number = int(match.group(1))
        if number not in numbers:
            numbers.append(number)
    return numbers


def strip_invalid_sources(text: str, *, max_index: int) -> str:
    """剔除超出实际来源范围的引用编号（DR-11）。

    模型可能写出「【来源7】」而本次只提供了 5 条资料。此时**直接删掉该标记**，
    既不渲染引用卡片、也不写入 `message_sources`，绝不因模型编造编号而伪造来源。
    """
    return SOURCE_PATTERN.sub(
        lambda match: match.group(0) if int(match.group(1)) <= max_index else "",
        text or "",
    )
