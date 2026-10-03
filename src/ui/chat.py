"""问答页（PG-02）：多轮提问、流式回答、引用溯源、反馈、复制与导出。

设计依据：
- docs/05-产品原型与交互说明.md §PG-02（页面元素与交互）
- docs/02-架构设计.md §4.5（防幻觉与引用）、§4.6（降级）、§4.10（分类过滤）
- docs/07-设计令牌.md §8.2（引用卡片用 `app-card`；分类过滤放页面顶部工具栏，DR-09）
- docs/03-开发任务清单.md M2-05 / M2-07

**当前身份说明**：认证在 M3-03 才到位，本页暂时以匿名身份运行——
检索只覆盖公共文档（`user_id=None`），会话归属用内置占位 `-1`。
个人资料检索与多会话列表分别在 M3 与 M2-06（FB-2.4）补齐。

纯逻辑函数（问题校验、导出 Markdown、相似度文案）不依赖 Streamlit，可直接单测。
"""

from __future__ import annotations

import json
import logging

import streamlit as st
import streamlit.components.v1 as components

from config.settings import PUBLIC_USER_ID, Settings, category_options
from src.rag.chain import stream_answer
from src.rag.prompts import parse_source_numbers
from src.repository import (
    RATING_USEFUL,
    RATING_USELESS,
    ROLE_ASSISTANT,
    ROLE_USER,
    add_feedback,
    create_conversation,
    get_feedback,
    list_messages,
    list_sources_by_message,
)
from src.store.chroma import VectorStore

logger = logging.getLogger(__name__)

# 单条问题长度上限（docs/05 PG-02「输入校验」）
MAX_QUESTION_LEN = 500
# 会话标题取首问前 15 字（docs/06 §1.5 会话接口约定）
TITLE_MAX_LEN = 15
# 认证到位前，会话归属使用与公共文档一致的占位身份
ANONYMOUS_OWNER_ID = PUBLIC_USER_ID

STATE_CONVERSATION = "chat_conversation_id"
STATE_CATEGORY = "chat_category"
STATE_NOTICE = "chat_notice"


# ==================== 纯逻辑（不依赖 Streamlit） ====================


def validate_question(text: str) -> str:
    """校验提问。返回空串表示通过，否则返回面向用户的提示文案。"""
    question = (text or "").strip()
    if not question:
        return "请输入问题内容。"
    if len(question) > MAX_QUESTION_LEN:
        return f"问题过长，请精简到 {MAX_QUESTION_LEN} 字以内。"
    return ""


def source_labels(content: str, sources: list[dict]) -> list[tuple[int, dict]]:
    """给每条引用配回它**在正文里的编号**，保证正文 `【来源N】` 与卡片一一对应。

    模型可能只引用第 1、5 条，此时卡片若按位置重编成 1、2，正文的 `【来源5】`
    就找不到对应卡片了。这里用正文里出现的编号顺序与 sources 顺序对齐
    （chain 正是按这个顺序产出 sources 的）；数量对不上（例如降级分支没有任何
    引用标记）时退回按顺序编号。
    """
    numbers = parse_source_numbers(content or "")
    if len(numbers) != len(sources):
        numbers = list(range(1, len(sources) + 1))
    return list(zip(numbers, sources))


def score_text(score: float | None) -> str:
    """引用卡片的相似度文案。

    只有向量路能给余弦相似度；纯关键词命中的切片没有分数，如实说明来源方式，
    不编造一个 0 分或空值（DR-15）。
    """
    return f"相似度 {score:.2f}" if score is not None else "关键词命中"


def conversation_title(messages) -> str:
    """标题取首问前 15 字（空会话显示「新会话」）。"""
    for message in messages:
        content = (message.content or "").strip()
        if message.role == ROLE_USER and content:
            return content[:TITLE_MAX_LEN]
    return "新会话"


def answer_clipboard_text(content: str, sources: list[dict]) -> str:
    """复制到剪贴板的文本：答案正文 + 引用来源（docs/05 PG-02「复制答案」）。"""
    if not sources:
        return content
    refs = "\n".join(
        f"{number}. {row.get('filename', '')}（{score_text(row.get('score'))}）"
        for number, row in source_labels(content, sources)
    )
    return f"{content}\n\n引用来源：\n{refs}"


def export_markdown(*, title: str, messages, sources: dict[int, list[dict]]) -> str:
    """把整个会话导出为 Markdown（FR-20）。"""
    lines = [f"# {title or '校答会话'}", ""]
    for message in messages:
        lines += [f"## {'我' if message.role == ROLE_USER else '校答'}", "", message.content, ""]
        rows = sources.get(message.id) or []
        if rows:
            lines.append("引用来源：")
            lines += [
                f"{number}. {row.get('filename', '')}（{score_text(row.get('score'))}）"
                for number, row in source_labels(message.content, rows)
            ]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def category_filter_options() -> list[tuple[str, str | None]]:
    """分类过滤下拉的选项：(显示名, 分类 key)。「全部」不加过滤条件。"""
    return [("全部", None)] + [
        (item["name"], item["key"]) for item in category_options()
    ]


# ==================== 页面渲染 ====================


def render(*, settings: Settings, store: VectorStore) -> None:
    """渲染整个问答页。"""
    st.markdown('<div class="cqa-page-title">问答</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cqa-page-desc">只依据知识库资料作答，结论标注【来源N】并附引用卡片；'
        "资料无法回答时明确拒答，不做猜测。</div>",
        unsafe_allow_html=True,
    )

    conversation_id = _ensure_conversation(settings)
    messages = list_messages(conversation_id, db_path=settings.database_path)
    sources = {
        message.id: list_sources_by_message(message.id, db_path=settings.database_path)
        for message in messages
        if message.role == ROLE_ASSISTANT
    }

    category = _render_toolbar(settings=settings, messages=messages, sources=sources)
    _render_notice()
    _render_history(messages, sources=sources, settings=settings)

    question = st.chat_input("输入你的问题，例如：搬宿舍需要提前申请吗")
    if question is not None:
        _handle_submit(
            question,
            conversation_id=conversation_id,
            category=category,
            settings=settings,
            store=store,
        )


def _ensure_conversation(settings: Settings) -> int:
    """取当前会话；首次进入时建一条空会话并记住（M2-06 接手多会话管理）。"""
    conversation_id = st.session_state.get(STATE_CONVERSATION)
    if conversation_id is None:
        conversation_id = create_conversation(ANONYMOUS_OWNER_ID, db_path=settings.database_path)
        st.session_state[STATE_CONVERSATION] = conversation_id
    return int(conversation_id)


def _render_toolbar(*, settings: Settings, messages, sources: dict[int, list[dict]]) -> str | None:
    """页面顶部工具栏：资料分类过滤（DR-09）+ 导出会话（FR-20）。"""
    options = category_filter_options()
    labels = [label for label, _ in options]
    title = conversation_title(messages)

    left, right = st.columns([3, 1])
    with left:
        picked = st.selectbox("资料分类", labels, index=0, key=STATE_CATEGORY)
    with right:
        st.download_button(
            "导出会话",
            data=export_markdown(title=title, messages=messages, sources=sources),
            file_name=f"{title}.md",
            mime="text/markdown",
            disabled=not messages,
            help="把当前会话下载为 Markdown（含引用来源）",
        )
    return dict(options)[picked]


def _render_notice() -> None:
    notice = st.session_state.pop(STATE_NOTICE, "")
    if notice:
        st.warning(notice)


def _render_history(messages, *, sources: dict[int, list[dict]], settings: Settings) -> None:
    """渲染历史消息。刷新后从数据库重放，因此不会丢。"""
    if not messages:
        st.markdown(
            '<div class="cqa-panel"><div class="cqa-empty">'
            "还没有对话。先上传资料，再在下方提问试试。</div></div>",
            unsafe_allow_html=True,
        )
        return

    for message in messages:
        with st.chat_message("user" if message.role == ROLE_USER else "assistant"):
            st.markdown(message.content)
            if message.role != ROLE_ASSISTANT:
                continue
            rows = sources.get(message.id, [])
            _render_sources(message.content, rows)
            _render_answer_actions(
                message_id=message.id,
                content=message.content,
                sources=rows,
                settings=settings,
            )


def _render_sources(content: str, sources: list[dict]) -> None:
    """引用卡片：文档名 + 片段 + 相似度，可展开原文（FR-09）。

    卡片编号取自正文的 `【来源N】`，与正文一一对应；拒答时列表为空，不渲染。
    """
    for number, row in source_labels(content, sources):
        label = f"{number}. {row.get('filename', '')} · {score_text(row.get('score'))}"
        with st.expander(label):
            st.markdown(row.get("snippet") or "")


def _render_answer_actions(
    *, message_id: int, content: str, sources: list[dict], settings: Settings
) -> None:
    """反馈按钮（FR-18）与复制答案（FR-19）。"""
    selected = get_feedback(message_id, ANONYMOUS_OWNER_ID, db_path=settings.database_path)

    up_col, down_col, copy_col, _ = st.columns([1, 1, 1.4, 4.6])
    if up_col.button(
        "有用",
        key=f"fb-up-{message_id}",
        disabled=selected is not None,
        type="primary" if selected == RATING_USEFUL else "secondary",
    ):
        _save_feedback(message_id, RATING_USEFUL, settings)
    if down_col.button(
        "没用",
        key=f"fb-down-{message_id}",
        disabled=selected is not None,
        type="primary" if selected == RATING_USELESS else "secondary",
    ):
        _save_feedback(message_id, RATING_USELESS, settings)
    with copy_col:
        _copy_button(answer_clipboard_text(content, sources), key=str(message_id))


def _save_feedback(message_id: int, rating: str, settings: Settings) -> None:
    add_feedback(message_id, ANONYMOUS_OWNER_ID, rating, db_path=settings.database_path)
    st.rerun()


def _copy_button(text: str, *, key: str) -> None:
    """复制答案到剪贴板。

    Streamlit 没有原生复制按钮，只能用 `components.html` 里的脚本；
    iframe 不能继承页面 CSS 变量，因此颜色在这里写死为设计令牌的取值。
    局域网走 http 时 `navigator.clipboard` 不可用（非安全上下文），
    故保留 execCommand 兜底。
    """
    payload = json.dumps(text)
    components.html(
        f"""
        <div style="font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif">
          <button id="cqa-copy-{key}" type="button" style="
              height:38px;padding:0 16px;cursor:pointer;
              border:1px solid #e7eaef;border-radius:999px;background:#ffffff;
              color:#0e1115;font-size:13px;font-weight:500">复制答案</button>
          <span id="cqa-copied-{key}" style="margin-left:8px;font-size:12px;color:#16a34a"></span>
        </div>
        <script>
          const btn = document.getElementById("cqa-copy-{key}");
          const tip = document.getElementById("cqa-copied-{key}");
          const text = {payload};
          function done() {{
            tip.textContent = "已复制";
            setTimeout(() => {{ tip.textContent = ""; }}, 1500);
          }}
          function fallback() {{
            const area = document.createElement("textarea");
            area.value = text;
            area.style.position = "fixed";
            area.style.opacity = "0";
            document.body.appendChild(area);
            area.select();
            try {{ document.execCommand("copy"); done(); }} catch (err) {{}}
            document.body.removeChild(area);
          }}
          btn.addEventListener("click", () => {{
            if (navigator.clipboard && window.isSecureContext) {{
              navigator.clipboard.writeText(text).then(done).catch(fallback);
            }} else {{
              fallback();
            }}
          }});
        </script>
        """,
        height=46,
    )


def _handle_submit(
    question: str,
    *,
    conversation_id: int,
    category: str | None,
    settings: Settings,
    store: VectorStore,
) -> None:
    """处理一次提问：校验 → 上屏用户消息 → 流式输出 → 引用卡片 → 操作行。

    这里**不做 st.rerun()**：答案在流式结束时已落库，直接留在屏幕上即可；
    若在此 rerun，刚渲染的答案会被脚本重跑冲掉（Sprint 1 踩过的坑）。
    """
    reason = validate_question(question)
    if reason:
        st.session_state[STATE_NOTICE] = reason
        st.rerun()
        return
    question = question.strip()

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        hint = st.empty()
        hint.markdown(
            '<div class="cqa-hint">正在检索资料…</div>', unsafe_allow_html=True
        )
        turn = stream_answer(
            question,
            None,
            conversation_id,
            category=category,
            store=store,
            settings=settings,
            db_path=settings.database_path,
        )
        hint.empty()  # 检索已完成，进入生成阶段
        st.write_stream(turn.tokens)

        result = turn.result
        if result is None:  # pragma: no cover - 生成器必然回填
            return

        # 引用读库回显，与落库内容天然一致（C-06）
        latest = list_messages(conversation_id, limit=1, db_path=settings.database_path)
        if not latest or latest[0].role != ROLE_ASSISTANT:
            return
        message = latest[0]
        rows = list_sources_by_message(message.id, db_path=settings.database_path)
        _render_sources(message.content, rows)
        _render_answer_actions(
            message_id=message.id,
            content=message.content,
            sources=rows,
            settings=settings,
        )
