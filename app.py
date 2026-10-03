"""校答（campus-qa）应用入口与路由。

设计依据：docs/02-架构设计.md §3（`app.py`：入口 + 路由，登录态与角色守卫）

M1 阶段只挂一个页面：文档管理（docs/03 M1-18）。
登录态与角色守卫（docs/05 的 G-01 / G-02）依赖 M3 的认证模块，届时在此处加守卫，
现在没有登录，因此不做拦截。

启动动作放在 `@st.cache_resource` 里，保证整个进程只执行一次：
建目录、建表、回收上次进程中断的入库任务（docs/02 §12 技术债）。
"""

from __future__ import annotations

import logging

import streamlit as st

from config.settings import get_settings
from src.ingest.tasks import recover_stale_tasks
from src.store.chroma import VectorStore
from src.store.db import init_db
from src.ui import documents, theme

logger = logging.getLogger(__name__)


@st.cache_resource
def bootstrap() -> None:
    """进程内只跑一次的启动动作。"""
    settings = get_settings()
    settings.ensure_dirs()
    init_db()
    # 后台线程实现异步入库，进程重启会丢失运行中的任务，
    # 这里把残留的 running 统一置为 failed，列表中即可手动重试（docs/02 §11）
    recovered = recover_stale_tasks()
    logger.info("启动完成：数据目录 %s，回收中断任务 %d 条", settings.data_dir, recovered)


@st.cache_resource
def get_vector_store() -> VectorStore:
    """整个进程复用同一个向量库实例。

    本地 BGE 加载一次需要数秒，若每次交互或每个文件都新建，代价不可接受
    （FB-1.5 自审清单里登记的技术债，此处收口）。
    """
    return VectorStore(get_settings())


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # 只留业务日志，压掉两类与业务无关的噪音：
    # 1) Streamlit 的文件监视器会遍历所有已导入模块去定位源码，而 transformers
    #    的惰性导入会在遍历时逐个尝试导入全部子模块（图像处理模块依赖未安装的
    #    torchvision），于是刷出上千条 ModuleNotFoundError 堆栈。监视器本身已
    #    捕获该异常，属于纯日志噪音。
    # 2) httpx 会把每一次 HTTP 请求都按 INFO 打出来。
    logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    _configure_logging()
    st.set_page_config(page_title="校答 · 文档管理", layout="wide")
    # 注入设计令牌与控件覆盖（docs/07 §8），必须早于页面内容渲染
    theme.apply_theme()

    bootstrap()
    documents.render(settings=get_settings(), store=get_vector_store())


main()
