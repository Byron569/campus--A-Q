"""文档管理页：多选上传、入库进度、列表、删除、重试。

设计依据：
- docs/05-产品原型与交互说明.md §PG-03（页面元素与交互）
- docs/02-架构设计.md §4.7（异步入库与进度、DR-05/DR-07）、§4.9（删除）、§6.1（入库流）
- docs/03-开发任务清单.md §1.7 M1-18

模块归属说明：设计文档没有指定「上传校验与落盘」「删除级联」的归属模块。
这里放在本模块内（docs/02 §3 给 documents.py 的职责就是多选上传 / 进度 / 重试），
不新增设计目录之外的模块。

**M1 范围说明**：本期尚无登录（认证在 M3-03），页面无法确定上传者身份，
因此按**公共文档**入库（`is_public=1`、`user_id=NULL`），分类必选，
与 docs/02 §4.10「管理员上传公共文档必须指定分类」一致。
个人文档页随 M3 的登录能力一起到位；M3-04 管理员页接管公共文档上传。

纯逻辑函数（校验 / 落盘 / 级联删除 / 重试）不依赖 Streamlit，可直接单元测试；
渲染函数只做 st.* 编排。
"""

from __future__ import annotations

import html
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from config.settings import (
    ALLOWED_SUFFIXES,
    PUBLIC_USER_ID,
    UNCATEGORIZED_KEY,
    Settings,
    category_options,
    get_settings,
)
from src.errors import CampusQAError, IngestError
from src.ingest.tasks import submit_ingest
from src.repository import (
    DOC_ACTIVE,
    TASK_DONE,
    TASK_FAILED,
    TASK_PENDING,
    TASK_RUNNING,
    create_document,
    create_task,
    exists_active_task,
    get_document,
    get_task_by_doc,
    list_documents,
    list_tasks_by_user,
    reset_task,
    soft_delete_document,
)
from src.store.chroma import VectorStore

logger = logging.getLogger(__name__)

# 未完成的入库任务状态
ACTIVE_TASK_STATUSES = (TASK_PENDING, TASK_RUNNING)

# session_state 键：已提交过的文件指纹（DR-07 防重复提交）
STATE_SUBMITTED = "documents_submitted"
# session_state 键：列表页码
STATE_PAGE = "documents_page"
# session_state 键：已放弃等待的入库任务 id，避免每次刷新都白等一轮
STATE_ABANDONED = "documents_abandoned"
# session_state 键：提交结果。必须暂存后渲染——提交完立刻 st.rerun()，
# 直接打印在这轮的消息会被 rerun 冲掉，用户什么都看不到
STATE_RESULTS = "documents_results"

STATUS_LABELS = {
    TASK_PENDING: "等待中",
    TASK_RUNNING: "入库中",
    TASK_DONE: "已完成",
    TASK_FAILED: "失败",
}

_ILLEGAL_NAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

POLL_INTERVAL_SECONDS = 0.5
# 进度轮询的总上限，防止个别卡死的任务把页面挂住
MAX_POLL_SECONDS = 120


# ==================== 纯逻辑（不依赖 Streamlit） ====================


def sanitize_filename(name: str) -> str:
    """清洗文件名：去掉目录部分与非法字符，避免目录穿越与路径注入。

    Windows 风格的反斜杠在 POSIX 上不是分隔符，但上传者可能来自 Windows，
    因此先把两种分隔符统一，再取最后一段。
    """
    normalized = (name or "").strip().replace("\\", "/")
    base = _ILLEGAL_NAME_CHARS.sub("_", Path(normalized).name).strip()
    # 去掉开头的点，避免落成隐藏文件或 "." / ".."
    base = base.lstrip(".")
    return base or "未命名文件"


def validate_upload(filename: str, size_bytes: int, *, settings: Settings | None = None) -> str:
    """校验上传文件。

    Returns:
        空串表示通过；否则返回面向用户的拒绝原因（文案对齐 docs/02 §11）。
        用返回值而非异常，是为了让「单文件报错不影响同批其他文件」写起来直接。
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        supported = "、".join(sorted(ALLOWED_SUFFIXES))
        return f"不支持的文件类型：{suffix or filename}（支持：{supported}）"

    cfg = settings or get_settings()
    if size_bytes > cfg.max_upload_bytes:
        return f"文件超过 {cfg.max_upload_mb}MB 上限"
    return ""


def uploads_dir(owner_id: int | None, *, settings: Settings) -> Path:
    """该归属者的上传目录：data/uploads/<user_id>/。

    公共文档的 user_id 用占位值 -1，与 Chroma 元数据里的约定保持一致。
    """
    directory = settings.uploads_path / str(PUBLIC_USER_ID if owner_id is None else owner_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def stored_path(
    *, owner_id: int | None, doc_id: int, filename: str, settings: Settings
) -> Path:
    """落盘路径。带 doc_id 前缀，保证同名文件互不覆盖，且可由文档记录反推出来。"""
    return uploads_dir(owner_id, settings=settings) / f"{doc_id}_{sanitize_filename(filename)}"


@dataclass
class StagedUpload:
    doc_id: int
    path: Path


def stage_upload(
    filename: str,
    data: bytes,
    *,
    category: str,
    is_public: bool,
    user_id: int | None,
    settings: Settings,
) -> StagedUpload:
    """落盘并建记录：documents(active) → 原始文件。

    入库任务由 `enqueue_new` 负责创建——两者必须分开，否则 DR-07 的幂等校验
    会撞上自己刚建出来的 pending 任务，把正常提交误判成重复提交。

    与 docs/02 §6.1 的顺序略有出入：这里**先写 documents 拿到 doc_id，再落盘**。
    原因是落盘文件名需要 doc_id 才能保证同名文件不互相覆盖——若先落盘，
    同名文件会写到同一路径，删除其中一个会误删另一个的文件，且重试时无法
    反推出该文档的文件路径。此调整已登记在自审清单中。

    Raises:
        IngestError: 落盘失败（已把刚建的文档记录软删除，不留半条记录）。
    """
    doc_id = create_document(
        filename=filename,
        filetype=Path(filename).suffix.lower().lstrip("."),
        category=category,
        size_bytes=len(data),
        user_id=user_id,
        is_public=is_public,
        db_path=settings.database_path,
    )

    target = stored_path(
        owner_id=user_id, doc_id=doc_id, filename=filename, settings=settings
    )
    try:
        target.write_bytes(data)
    except OSError as exc:
        soft_delete_document(doc_id, db_path=settings.database_path)
        raise IngestError(f"文件保存失败：{filename}（{exc}）") from exc

    return StagedUpload(doc_id=doc_id, path=target)


def enqueue_new(
    staged: StagedUpload,
    *,
    category: str,
    is_public: bool,
    user_id: int | None,
    store: VectorStore,
    settings: Settings,
) -> int:
    """为该文档新建入库任务并启动（DR-07：写入任务前先做幂等校验）。返回 task_id。

    幂等校验必须在 `create_task` **之前**：若某文档已有 pending/running 任务，
    说明它正在入库，此时再建一条任务会让同一文档出现两条记录。
    """
    if exists_active_task(staged.doc_id, db_path=settings.database_path):
        raise IngestError("该文档已有未完成的入库任务，请勿重复提交")

    task_id = create_task(staged.doc_id, user_id=user_id, db_path=settings.database_path)

    submit_ingest(
        path=staged.path,
        doc_id=staged.doc_id,
        task_id=task_id,
        user_id=user_id,
        is_public=is_public,
        category=category,
        store=store,
        db_path=settings.database_path,
    )
    return task_id


def retry_document(doc_id: int, *, store: VectorStore, settings: Settings) -> None:
    """失败重试（DR-05）：**复用原任务**，不新建记录。

    重试前先清掉该文档已写入的部分向量，否则残留脏数据会在检索时重复命中。
    """
    document = get_document(doc_id, db_path=settings.database_path)
    if document is None or document.status != DOC_ACTIVE:
        raise IngestError("文档不存在或已删除")

    task = get_task_by_doc(doc_id, db_path=settings.database_path)
    if task is None:
        raise IngestError("该文档没有入库任务记录，无法重试")
    if task.status in ACTIVE_TASK_STATUSES:
        raise IngestError("该文档正在入库，请稍候")

    store.delete_by_doc_id(doc_id)
    reset_task(task.id, db_path=settings.database_path)

    path = stored_path(
        owner_id=document.user_id,
        doc_id=document.id,
        filename=document.filename,
        settings=settings,
    )
    if not path.exists():
        raise IngestError(f"原始文件已丢失，无法重试：{path.name}")

    submit_ingest(
        path=path,
        doc_id=document.id,
        task_id=task.id,
        user_id=document.user_id,
        is_public=document.is_public,
        category=document.category,
        store=store,
        db_path=settings.database_path,
    )
    logger.info("已重新入队 doc_id=%s task_id=%s", document.id, task.id)


def delete_document(doc_id: int, *, store: VectorStore, settings: Settings) -> None:
    """删除文档（docs/02 §4.9）：soft delete → 清向量 → 删原始文件。"""
    document = get_document(doc_id, db_path=settings.database_path)
    if document is None or document.status != DOC_ACTIVE:
        return

    soft_delete_document(doc_id, db_path=settings.database_path)
    removed = store.delete_by_doc_id(doc_id)

    path = stored_path(
        owner_id=document.user_id,
        doc_id=document.id,
        filename=document.filename,
        settings=settings,
    )
    if path.exists():
        path.unlink()

    logger.info("已删除文档 doc_id=%s，清理向量 %d 条", document.id, removed)


# ==================== 页面渲染 ====================
#
# 视觉按 docs/07-设计令牌.md 落地：面板 + 1px 描边 + 14px 圆角 + 无阴影，
# 状态用胶囊标签（模板 .status-chip），按钮为胶囊形。样式本体在 src/ui/theme.py。


def _panel_head(title: str, subtitle: str = "") -> None:
    """面板标题行，对应模板 data-table 的 .panel .head。"""
    sub = f'<span class="cqa-sub">{subtitle}</span>' if subtitle else ""
    st.markdown(
        f'<div class="cqa-panel-head" style="border:0;padding:0 0 10px">'
        f"<h3>{title}</h3>{sub}</div>",
        unsafe_allow_html=True,
    )


def render(*, settings: Settings, store: VectorStore) -> None:
    """渲染整个文档管理页。"""
    st.markdown('<div class="cqa-page-title">文档管理</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cqa-page-desc">当前版本尚未提供登录（认证在 M3），'
        "本页上传的文件按公共文档入库，所有用户均可检索到。</div>",
        unsafe_allow_html=True,
    )

    _render_upload(settings=settings, store=store)
    _render_progress(settings=settings)
    _render_list(settings=settings, store=store)

    # 必须放在最后：进度轮询结束时还会 st.rerun() 一次，若在它之前渲染，
    # 提交结果会只闪一下就没了
    _drain_results()


def _render_upload(*, settings: Settings, store: VectorStore) -> None:
    suffix_hint = " / ".join(sorted(s.lstrip(".").upper() for s in ALLOWED_SUFFIXES))
    _panel_head("上传文档", f"支持 {suffix_hint}，单文件不超过 {settings.max_upload_mb}MB，可多选")

    # 公共文档必须指定分类（docs/02 §4.10），因此不提供「未分类」选项
    options = [item for item in category_options() if item["key"] != UNCATEGORIZED_KEY]
    label_to_key = {item["name"]: item["key"] for item in options}

    left, right = st.columns([3, 2])
    with left:
        files = st.file_uploader(
            "选择要入库的文件（可多选）",
            type=sorted(suffix.lstrip(".") for suffix in ALLOWED_SUFFIXES),
            accept_multiple_files=True,
        )
    with right:
        picked = st.selectbox(
            "分类（必选）", list(label_to_key), index=None, placeholder="请选择分类"
        )
        start = st.button("开始上传", type="primary")

    if not start:
        return

    if not files:
        st.warning("请先选择文件。")
        return
    if not picked:
        st.warning("公共文档必须指定分类（docs/02 §4.10）。")
        return

    category = label_to_key[picked]
    submitted: set[str] = st.session_state.setdefault(STATE_SUBMITTED, set())

    results: list[tuple[str, str]] = []
    for uploaded in files:
        results.append(
            _submit_one(
                uploaded, category=category, submitted=submitted, settings=settings, store=store
            )
        )

    st.session_state[STATE_RESULTS] = results
    st.rerun()


def _drain_results() -> None:
    """渲染并清空上一轮暂存的提交结果。"""
    results: list[tuple[str, str]] = st.session_state.pop(STATE_RESULTS, [])
    for level, text in results:
        if level == "success":
            st.success(text)
        elif level == "error":
            st.error(text)
        else:
            st.info(text)


def _submit_one(
    uploaded, *, category: str, submitted: set[str], settings: Settings, store: VectorStore
) -> tuple[str, str]:
    """处理单个上传文件，返回 (消息级别, 文案)。

    失败只影响本文件，不中断同批（docs/02 §11 统一原则 1）。
    """
    fingerprint = f"{uploaded.name}|{uploaded.size}"
    if fingerprint in submitted:
        # Streamlit 每次交互都会重跑脚本，不加这道闸同一个文件会被提交两次（DR-07）
        return "info", f"已跳过（本次会话已提交过）：{uploaded.name}"

    reason = validate_upload(uploaded.name, uploaded.size, settings=settings)
    if reason:
        return "error", reason

    try:
        staged = stage_upload(
            uploaded.name,
            uploaded.getvalue(),
            category=category,
            is_public=True,
            user_id=None,
            settings=settings,
        )
        enqueue_new(
            staged,
            category=category,
            is_public=True,
            user_id=None,
            store=store,
            settings=settings,
        )
    except CampusQAError as exc:
        return "error", f"{uploaded.name}：{exc}"

    submitted.add(fingerprint)
    return "success", f"{uploaded.name} 已提交入库"


def _render_progress(*, settings: Settings) -> None:
    """对未完成的入库任务轮询刷新进度条。"""
    tasks, _ = list_tasks_by_user(None, page=1, page_size=50, db_path=settings.database_path)
    abandoned: set[int] = st.session_state.setdefault(STATE_ABANDONED, set())
    active = [
        task
        for task in tasks
        if task.status in ACTIVE_TASK_STATUSES and task.id not in abandoned
    ]
    if not active:
        return

    st.markdown('<div class="cqa-panel-head" style="border:0;padding:16px 0 6px"><h3>入库进度</h3></div>',
                unsafe_allow_html=True)
    bars = {task.id: st.progress(task.progress, text=_progress_text(task)) for task in active}

    # 轮询直到本批任务全部结束，再刷新页面让列表显示最终状态。
    # 必须有上限：若某条任务因异常停在 pending（例如进程在「建任务」与
    # 「启动线程」之间被杀），无上限的循环会让页面永久卡住。
    deadline = time.monotonic() + MAX_POLL_SECONDS
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_SECONDS)
        tasks, _ = list_tasks_by_user(
            None, page=1, page_size=50, db_path=settings.database_path
        )
        pending_left = False
        for task in tasks:
            if task.id not in bars:
                continue
            bars[task.id].progress(task.progress, text=_progress_text(task))
            if task.status in ACTIVE_TASK_STATUSES:
                pending_left = True
        if not pending_left:
            break
    else:
        # 放弃等待的任务记下来，避免下一次页面刷新又白白等一轮
        abandoned.update(bars)
        st.warning("部分任务长时间未推进，已停止等待。可在下方列表重试。")
        return

    st.rerun()


def _progress_text(task) -> str:
    return (
        f"#{task.doc_id}  {STATUS_LABELS.get(task.status, task.status)}"
        f"  {task.done_chunks}/{task.total_chunks} 个切片"
    )


def _status_chip(document, task) -> str:
    """状态胶囊。对应模板 .status-chip 的四种语义。"""
    if task is None:
        # 命令行入库（ingest_cli.py）不建 ingest_tasks 记录，
        # 这类文档只要已有切片就说明入库成功，不能标成「未入库」
        if document.chunk_count > 0:
            return '<span class="cqa-chip cqa-chip--ok">已入库</span>'
        return '<span class="cqa-chip cqa-chip--wait">未入库</span>'

    label = STATUS_LABELS.get(task.status, task.status)
    counts = f" {task.done_chunks}/{task.total_chunks}"
    if task.status == TASK_DONE:
        return f'<span class="cqa-chip cqa-chip--ok">已完成{counts}</span>'
    if task.status == TASK_RUNNING:
        return f'<span class="cqa-chip cqa-chip--run">入库中{counts}</span>'
    if task.status == TASK_FAILED:
        return '<span class="cqa-chip cqa-chip--fail">失败</span>'
    return f'<span class="cqa-chip cqa-chip--wait">{label}{counts}</span>'


def _render_list(*, settings: Settings, store: VectorStore) -> None:
    """文档列表：分页、状态、失败原因、删除、重试。"""
    page = int(st.session_state.get(STATE_PAGE, 1))
    page_size = settings.page_size
    rows, total = list_documents(
        include_public=True, page=page, page_size=page_size, db_path=settings.database_path
    )
    total_pages = max((total + page_size - 1) // page_size, 1)

    header, nav = st.columns([3, 2])
    with header:
        _panel_head("文档列表", f"共 {total} 条 · 每页 {page_size} 条")
    with nav:
        prev_col, page_col, next_col = st.columns([1, 2, 1])
        if prev_col.button("上一页", disabled=page <= 1 or not rows, key="page-prev"):
            st.session_state[STATE_PAGE] = max(page - 1, 1)
            st.rerun()
        page_col.markdown(
            f'<div class="cqa-pager">第 {page} / {total_pages} 页</div>',
            unsafe_allow_html=True,
        )
        if next_col.button("下一页", disabled=page >= total_pages or not rows, key="page-next"):
            st.session_state[STATE_PAGE] = page + 1
            st.rerun()

    if not rows:
        st.markdown(
            '<div class="cqa-panel"><div class="cqa-empty">'
            "还没有文档，上传学生手册或通知试试。</div></div>",
            unsafe_allow_html=True,
        )
        return

    for document in rows:
        _render_row(document, settings=settings, store=store)


def _row_html(document, task) -> str:
    """一行列表的 HTML：文件徽标 + 名称/元信息 + 状态胶囊。

    行边框由自有 .cqa-row 画，不用 st.container(border=True)——后者的边框写在
    随 Streamlit 版本变化的 emotion 类上，改版就会静默失效（见 theme.py 注释）。
    文件名来自用户上传，必须转义后再拼进 HTML。
    """
    name = html.escape(document.filename)
    badge = html.escape((document.filetype or "?").upper()[:4])
    meta = (
        f"{html.escape(document.category)} · {document.size_bytes / 1024:.0f} KB · "
        f"{document.chunk_count} 个切片"
    )
    return (
        f'<div class="cqa-row">'
        f'<div class="cqa-fileicon">{badge}</div>'
        f'<div class="cqa-rowmain"><div class="cqa-name">{name}</div>'
        f'<div class="cqa-meta">{meta}</div></div>'
        f'<div class="cqa-rowstatus">{_status_chip(document, task)}</div>'
        f"</div>"
    )


def _render_row(document, *, settings: Settings, store: VectorStore) -> None:
    """一行文档：信息 + 状态 + 失败原因 + 重试 / 删除。"""
    task = get_task_by_doc(document.id, db_path=settings.database_path)

    info_col, action_col = st.columns([7, 3])

    with info_col:
        st.markdown(_row_html(document, task), unsafe_allow_html=True)
        if task is not None and task.status == TASK_FAILED and task.error:
            # PG-03 要求悬停展示失败原因；Streamlit 列表无悬停提示，改为行内展示
            st.error(task.error)

    with action_col:
        retry_col, delete_col = st.columns(2)
        if task is not None and task.status == TASK_FAILED:
            if retry_col.button("重试", key=f"retry-{document.id}"):
                try:
                    retry_document(document.id, store=store, settings=settings)
                    st.rerun()
                except CampusQAError as exc:
                    st.error(str(exc))
        with delete_col.popover("删除"):
            st.markdown(
                f'<div class="cqa-page-desc">确认删除《{html.escape(document.filename)}》？'
                "原文与向量一并清除，不可恢复。</div>",
                unsafe_allow_html=True,
            )
            if st.button("确认删除", key=f"delete-{document.id}", type="primary"):
                try:
                    delete_document(document.id, store=store, settings=settings)
                    st.rerun()
                except CampusQAError as exc:
                    st.error(str(exc))
