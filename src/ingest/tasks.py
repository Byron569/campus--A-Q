"""异步入库任务：后台线程执行，进度与结果落 `ingest_tasks`。

设计依据：docs/02-架构设计.md §4.7 / §6.1、docs/06-接口文档.md §1.3

为什么不引入 Celery / Redis（docs/02 §2）：单机 + 单进程 Streamlit，
后台线程足够，避免重型依赖。代价是进程重启会丢失运行中的任务——
因此启动时必须调用 `recover_stale_tasks()` 把残留的 running 置为 failed 供重试
（docs/02 §12 技术债）。

异常约定（docs/02 §11）：失败**不抛到主线程**，一律写入 `ingest_tasks.error`
并把状态置为 failed，用户看到原因与「重试」按钮；完整堆栈只进日志。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from config.settings import EMBED_BATCH_SIZE, UNCATEGORIZED_KEY
from src.errors import CampusQAError
from src.ingest.pipeline import ingest_file
from src.repository import (
    TASK_DONE,
    TASK_FAILED,
    TASK_RUNNING,
    recover_stale_tasks as _recover_stale_tasks,
    update_task,
)
from src.store.chroma import VectorStore

logger = logging.getLogger(__name__)

# 未预期异常的对外文案：不暴露堆栈，但必须给出明确提示（docs/02 §11 统一原则 3/4）
UNEXPECTED_ERROR_TEXT = "入库失败（系统异常），请重试"


def recover_stale_tasks(db_path: Path | str | None = None) -> int:
    """服务启动时回收中断的入库任务。返回处理条数。

    SQL 落在 `repository`（唯一数据库访问入口），此处按 docs/06 §1.3
    暴露为入库模块的启动钩子。
    """
    return _recover_stale_tasks(db_path=db_path)


def submit_ingest(
    *,
    path: str | Path,
    doc_id: int,
    task_id: int,
    user_id: int | None,
    is_public: bool = False,
    category: str = UNCATEGORIZED_KEY,
    store: VectorStore | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
    db_path: Path | str | None = None,
) -> threading.Thread:
    """启动后台入库线程并立即返回。

    调用方需先落 `documents` 与 `ingest_tasks(pending)` 记录，并把两者的 id 传进来。
    幂等防重（DR-07）由调用方在创建任务前用 `repository.exists_active_task` 完成。
    """
    thread = threading.Thread(
        target=_run_ingest,
        kwargs={
            "path": path,
            "doc_id": doc_id,
            "task_id": task_id,
            "user_id": user_id,
            "is_public": is_public,
            "category": category,
            "store": store,
            "batch_size": batch_size,
            "db_path": db_path,
        },
        name=f"ingest-doc-{doc_id}",
        # daemon：Streamlit 进程退出时不因残留任务阻塞
        daemon=True,
    )
    thread.start()
    return thread


def _run_ingest(
    *,
    path: str | Path,
    doc_id: int,
    task_id: int,
    user_id: int | None,
    is_public: bool,
    category: str,
    store: VectorStore | None,
    batch_size: int,
    db_path: Path | str | None,
) -> None:
    """后台线程主体：把结果写回 ingest_tasks，绝不把异常抛到主线程。"""
    # warning 一并清空：重试时上一次的告警不能留到这一次
    update_task(task_id, db_path=db_path, status=TASK_RUNNING, error=None, warning=None)

    try:
        result = ingest_file(
            path,
            category=category,
            is_public=is_public,
            user_id=user_id,
            doc_id=doc_id,
            task_id=task_id,
            store=store,
            batch_size=batch_size,
            db_path=db_path,
        )
    except CampusQAError as exc:
        # 业务异常自带面向用户的中文文案（如「文件已加密，无法解析」）
        logger.warning("入库失败 doc_id=%s：%s", doc_id, exc)
        update_task(task_id, db_path=db_path, status=TASK_FAILED, error=str(exc))
        return
    except Exception:
        logger.exception("入库出现未预期异常 doc_id=%s", doc_id)
        update_task(task_id, db_path=db_path, status=TASK_FAILED, error=UNEXPECTED_ERROR_TEXT)
        return

    # 告警（如「疑似扫描件」）要落库，文档列表才能把它显示出来（CR-03）
    update_task(
        task_id, db_path=db_path, status=TASK_DONE, warning=result.warning or None
    )
    if result.warning:
        logger.warning("入库告警 doc_id=%s：%s", doc_id, result.warning)
