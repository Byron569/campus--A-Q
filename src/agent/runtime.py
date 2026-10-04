"""Agent 运行时：规则 + LLM 双层路由，执行只读工具。

设计依据：docs/02 v1.14 §4.14；客户 2026-10-04 裁决（规则优先、LLM 兜底）。

路由顺序（不能换，换了就会给每次提问白加一次模型调用）：

1. 规则层命中 → 直接返回该工具，**不调用 LLM**（确定性、零成本、不拖慢首字节）；
2. 规则未命中、且问题里没有任何工具信号 → 直接走知识库检索，**同样不调用 LLM**；
3. 只有「疑似工具意图、但关键词不明确」时才调一次 LLM 做选择；
4. LLM 调用失败 / 选不出来 → 回退知识库检索，行为与二期前完全一致（不退化）。

规则层刻意收窄：`截止` / `作业` / `考试` / `安排` 这类词会出现在知识库问题里
（如「奖学金申请截止时间」），单独命中不足以判定为日程意图，必须**同时**出现
「我的 / 有什么 / 哪些」等本人指代才路由到日程；否则交给 LLM 层兜底。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from config.settings import Settings, get_settings
from src.agent.tools import (
    RANGE_ALL,
    RANGE_OVERDUE,
    RANGE_PENDING,
    RANGE_TODAY,
    RANGE_WEEK,
    TOOL_KNOWLEDGE,
    TOOL_SCHEDULES,
    TOOL_TIME,
    current_time_text,
    list_schedule_text,
)
from src.providers.llm import get_llm, response_text
from src.repository import ROLE_ASSISTANT, ROLE_USER, add_message, add_metric

logger = logging.getLogger(__name__)

# 规则层关键词
# 强词：几乎只与日程有关，单独命中即可路由
_STRONG_SCHEDULE_TERMS = ("日程", "待办", "ddl", "deadline")
# 弱词：可能出现在知识库问题里，需与本人指代同时出现
_WEAK_SCHEDULE_TERMS = (
    "提醒", "截止", "作业", "考试", "安排", "任务", "要交", "要做",
    "课表", "时间表", "待完成", "没完成", "没做",
)
_MINE_WORDS = ("我的", "我有", "有什么", "有哪些", "哪些", "最近", "这周", "本周")
_TIME_TERMS = ("几号", "星期几", "几点", "今天日期", "当前时间", "现在时间", "今天是星期")
# 出现这些词才值得多花一次 LLM 路由；否则直接走知识库检索。
# 必须与上面的两组关键词保持覆盖：漏掉的词会让问题**根本进不了**调度
# （既不被规则命中、也不触发 LLM 兜底），实测「我有什么要交的」就是这样漏的。
_TOOL_SIGNAL_TERMS = (
    "日程", "待办", "提醒", "截止", "ddl", "deadline",
    "作业", "考试", "安排", "任务", "要交", "要做", "课表", "时间表",
    "待完成", "没完成", "没做", "几号", "几点", "星期",
)

ROUTER_PROMPT = """你在为「校答」做工具路由，可选工具只有三个：
- knowledge_search：查校园知识库（通知、政策、流程、规定等公共资料）
- list_schedules：查用户**自己**的作业 / 考试日程
- current_time：查当前日期 / 时间

规则：只输出一个工具名，不要解释、不要标点。若问题与用户个人日程无关，一律输出 knowledge_search。

用户问题：{question}"""


@dataclass(frozen=True)
class ToolChoice:
    """路由结果：选中的工具名 + 参数。"""

    name: str
    range_key: str = RANGE_PENDING


@dataclass
class ToolResult:
    """工具直答结果：文本 + 落库后的助手消息 id（供反馈按钮使用）。"""

    text: str
    message_id: int


# ==================== 规则层 ====================


def parse_range(text: str) -> str:
    """从问题里解析日程查询范围，默认「未完成」。"""
    if "今天" in text:
        return RANGE_TODAY
    if any(word in text for word in ("本周", "这周", "最近", "未来", "接下来", "这几天", "下周")):
        return RANGE_WEEK
    if any(word in text for word in ("逾期", "过期", "没做", "没完成", "欠交", "拖欠")):
        return RANGE_OVERDUE
    if any(word in text for word in ("全部", "所有", "历史")):
        return RANGE_ALL
    return RANGE_PENDING


def rule_route(question: str) -> ToolChoice | None:
    """规则路由：命中返回工具，未命中返回 None（交给下一层）。"""
    text = (question or "").strip().lower()
    if not text:
        return None
    if any(term in text for term in _TIME_TERMS):
        return ToolChoice(TOOL_TIME)
    if any(term in text for term in _STRONG_SCHEDULE_TERMS):
        return ToolChoice(TOOL_SCHEDULES, range_key=parse_range(text))
    if any(word in text for word in _MINE_WORDS) and any(
        term in text for term in _WEAK_SCHEDULE_TERMS
    ):
        return ToolChoice(TOOL_SCHEDULES, range_key=parse_range(text))
    return None


def has_tool_signal(question: str) -> bool:
    """问题里是否出现了工具相关信号词（决定要不要再多花一次 LLM 路由）。"""
    text = (question or "").lower()
    return any(term in text for term in _TOOL_SIGNAL_TERMS)


# ==================== LLM 层 ====================


def parse_tool_name(response: str) -> str | None:
    """从模型回复里解析工具名。识别不了返回 None。"""
    text = (response or "").strip()
    for name in (TOOL_SCHEDULES, TOOL_TIME, TOOL_KNOWLEDGE):
        if name in text:
            return name
    # 兜底：模型只回了中文
    if "日程" in text:
        return TOOL_SCHEDULES
    if "时间" in text:
        return TOOL_TIME
    return None


def llm_route(
    question: str, *, llm=None, settings: Settings | None = None
) -> ToolChoice | None:
    """LLM 路由。任何异常都返回 None，由上层回退知识库检索。"""
    try:
        model = llm or get_llm(settings, streaming=False)
        response = model.invoke(
            [HumanMessage(content=ROUTER_PROMPT.replace("{question}", question))]
        )
        name = parse_tool_name(response_text(response))
    except Exception:
        logger.exception("工具路由失败，回退知识库检索")
        return None

    if name == TOOL_SCHEDULES:
        return ToolChoice(TOOL_SCHEDULES, range_key=parse_range(question))
    if name == TOOL_TIME:
        return ToolChoice(TOOL_TIME)
    if name == TOOL_KNOWLEDGE:
        return ToolChoice(TOOL_KNOWLEDGE)
    return None


def decide(
    question: str, *, llm=None, settings: Settings | None = None
) -> ToolChoice:
    """路由主入口：规则优先，LLM 兜底，最终默认知识库检索。"""
    rule = rule_route(question)
    if rule is not None:
        return rule
    if not has_tool_signal(question):
        return ToolChoice(TOOL_KNOWLEDGE)
    return llm_route(question, llm=llm, settings=settings) or ToolChoice(TOOL_KNOWLEDGE)


# ==================== 执行 ====================


def execute(
    choice: ToolChoice,
    question: str,
    *,
    user_id: int,
    conversation_id: int,
    settings: Settings | None = None,
    db_path=None,
    now=None,
) -> ToolResult:
    """执行直答工具并落库（用户消息 → 助手消息 → 指标）。

    只处理 `list_schedules` / `current_time`；`knowledge_search` 走既有 RAG 链路，
    不经过这里（传进来即视为调用错误）。
    """
    s = settings or get_settings()
    resolved_db = db_path or s.database_path
    started = time.perf_counter()

    if choice.name == TOOL_SCHEDULES:
        text = list_schedule_text(
            user_id, range_key=choice.range_key, db_path=resolved_db, now=now
        )
    elif choice.name == TOOL_TIME:
        text = current_time_text(now=now)
    else:
        raise ValueError(f"execute 只处理直答工具，收到：{choice.name}")

    latency_ms = int((time.perf_counter() - started) * 1000)
    message_id = _persist(
        question, text, conversation_id=conversation_id,
        db_path=resolved_db, latency_ms=latency_ms,
    )
    logger.info("Agent 工具直答：tool=%s message_id=%s", choice.name, message_id)
    return ToolResult(text=text, message_id=message_id)


def _persist(
    question: str, text: str, *, conversation_id: int, db_path, latency_ms: int
) -> int:
    """落库：用户消息 → 助手消息 → 指标（工具回答无引用来源）。"""
    add_message(conversation_id, ROLE_USER, question, db_path=db_path)
    assistant_id = add_message(conversation_id, ROLE_ASSISTANT, text, db_path=db_path)
    add_metric(
        question_len=len(question),
        answerable=True,
        latency_ms=latency_ms,
        db_path=db_path,
    )
    return assistant_id
