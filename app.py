"""校答（campus-qa）应用入口与路由。

设计依据：
- docs/02-架构设计.md §3（`app.py`：入口 + 路由，登录态与角色守卫）
- docs/05-产品原型与交互说明.md §G-03（侧边栏为 PG-02~PG-06 共用）、§G-04（刷新保持当前页）
- docs/07-设计令牌.md §7（导航图标映射）

当前可用：「问答」页（M2-05，默认落地）与「我的文档」页（M1-18）；
管理员 / 设置 / 关于分别在 M3-04 / M3-16 / M3-17 实现，这里先渲染占位页，
把导航框架立起来。
登录态守卫（G-01）与角色守卫（G-02，`user` 不显示管理员入口）依赖 M3 的认证模块，
届时在 `main()` 里加判断。

**为什么用查询参数记当前页**：G-04 要求「刷新页面保持当前页」，而
`st.session_state` 在浏览器刷新后会重置，只有 URL 里的查询参数能跨刷新存活。

启动动作放在 `@st.cache_resource` 里，保证整个进程只执行一次：
建目录、建表、回收上次进程中断的入库任务（docs/02 §12 技术债）。
"""

from __future__ import annotations

import html
import logging

import streamlit as st

from config.settings import get_settings, load_kb_config
from src.ingest.tasks import recover_stale_tasks
from src.store.chroma import VectorStore
from src.store.db import init_db
from src.ui import chat, documents, theme

logger = logging.getLogger(__name__)

QUERY_KEY = "page"

# lucide 图标路径（docs/07 §7 的映射表）。路径原样取自
# docs/design/doubao/assets/icons/，内联在代码里避免运行期依赖 docs/ 目录。
_SVG_ATTRS = (
    'class="cqa-nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"'
)
_ICONS = {
    "qa": '<path d="M2.992 16.342a2 2 0 0 1 .094 1.167l-1.065 3.29a1 1 0 0 0 1.236 1.168'
          'l3.413-.998a2 2 0 0 1 1.099.092 10 10 0 1 0-4.777-4.719"/>'
          '<path d="M8 12h.01"/><path d="M12 12h.01"/><path d="M16 12h.01"/>',
    "documents": '<path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706'
                 'l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z"/>'
                 '<path d="M14 2v5a1 1 0 0 0 1 1h5"/>',
    "admin": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/>'
             '<circle cx="12" cy="7" r="4"/>',
    "settings": '<path d="M13 21h8"/><path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174'
                'a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32'
                'a2 2 0 0 0 .83-.497z"/>',
    "about": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
             '<path d="M12 17h.01"/>',
}

# 导航项。plan 为空表示本期已实现；否则写明计划在哪个里程碑交付。
PAGES = (
    {"key": "qa", "label": "问答", "plan": ""},
    {"key": "documents", "label": "我的文档", "plan": ""},
    {"key": "admin", "label": "管理员", "plan": "公共文档与用户管理（docs/03 M3-04）"},
    {"key": "settings", "label": "设置", "plan": "修改显示名与密码、注销账号（docs/03 M3-16）"},
    {"key": "about", "label": "关于", "plan": "数据使用说明与学校联系方式（docs/03 M3-17）"},
)
PAGES_BY_KEY = {page["key"]: page for page in PAGES}
# G-04：登录成功默认落地问答页；无登录态时同样从问答页开始
DEFAULT_PAGE = "qa"


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


def resolve_page(raw: str | None) -> str:
    """把查询参数解析成合法的页面 key。缺失或非法时回落到默认页。"""
    return raw if raw in PAGES_BY_KEY else DEFAULT_PAGE


def render_sidebar(current: str) -> None:
    """渲染左侧导航。用 <a href="?page=xxx"> 而不是按钮：

    一是 G-04 要求刷新后保持当前页，锚点天然把状态写进 URL；
    二是只有自定义 HTML 才能用上设计指定的 lucide 图标（docs/07 §7）。
    """
    school = load_kb_config()["school"]["name"]
    # target="_self"：Streamlit 给 markdown 里的 <a> 自动加了 target="_blank"，
    # 不加这个属性每次点导航都会新开一个标签页
    items = "".join(
        f'<a class="cqa-nav-item{" is-active" if page["key"] == current else ""}" '
        f'target="_self" href="?{QUERY_KEY}={page["key"]}">'
        f'<svg {_SVG_ATTRS}>{_ICONS[page["key"]]}</svg>'
        f'<span class="cqa-navlabel">{html.escape(page["label"])}</span></a>'
        for page in PAGES
    )

    with st.sidebar:
        st.markdown(
            f'<div class="cqa-rail">'
            f'<div class="cqa-brand"><div class="cqa-brandmark">校</div>'
            f"<div><div class=\"cqa-brandname\">校答</div>"
            f'<div class="cqa-branddesc">{html.escape(str(school))}</div></div></div>'
            f'<nav class="cqa-nav">{items}</nav>'
            f'<div class="cqa-railfoot">未登录 · 账号体系在 M3 开放</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_placeholder(page: dict) -> None:
    """尚未实现的页面：明确写出计划交付的里程碑，不留空页。"""
    st.markdown(
        f'<div class="cqa-page-title">{html.escape(page["label"])}</div>'
        f'<div class="cqa-page-desc">该页面尚未实现，计划在 {html.escape(page["plan"])}。'
        "当前可用的是「问答」与「我的文档」。</div>",
        unsafe_allow_html=True,
    )


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
    st.set_page_config(
        page_title="校答 · 校园知识库问答",
        layout="wide",
        # "auto"：桌面端默认展开；窄屏（手机）由 Streamlit 自动折叠，
        # 否则 248px 的固定侧栏会挤掉大半内容区（M2 浏览器验收结论）
        initial_sidebar_state="auto",
    )
    # 注入设计令牌与控件覆盖（docs/07 §8），必须早于页面内容渲染
    theme.apply_theme()

    bootstrap()

    current = resolve_page(st.query_params.get(QUERY_KEY))
    render_sidebar(current)

    settings = get_settings()
    if current == "qa":
        chat.render(settings=settings, store=get_vector_store())
    elif current == "documents":
        documents.render(settings=settings, store=get_vector_store())
    else:
        render_placeholder(PAGES_BY_KEY[current])


# Streamlit 以 __main__ 执行本脚本；加这层判断是为了让测试可以安全 import
# （否则 import app 会直接把整个应用跑一遍）
if __name__ == "__main__":
    main()
