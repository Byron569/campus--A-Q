"""校答（campus-qa）应用入口与路由。

设计依据：
- docs/02-架构设计.md §3（`app.py`：入口 + 路由，登录态与角色守卫）
- docs/05-产品原型与交互说明.md §G-03（侧边栏为 PG-02~PG-06 共用）、§G-04（刷新保持当前页）
- docs/07-设计令牌.md §7（导航图标映射）

当前可用：「问答」（M2-05）、「我的文档」（M1-18）、「设置」（M3-16）、「关于」（M3-17）；
「管理员」页在 M3-04（FB-3.3）实现，先渲染占位页，把导航框架立起来。

登录态与守卫（FB-3.2）：
- `G-01 访客拦截`：未登录一律只渲染登录页，任何页面内容都不产出
- `G-02 角色守卫`：非 admin 访问管理员页 → 就地提示并回到问答页
- `G-03 侧边栏`：`user` 角色不显示「管理员」入口
账号被删除或被禁用时，已建立的登录态**立即失效**（下次交互即被踢回登录页），
否则禁用操作要等用户重新登录才生效。

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
from src.auth import service
from src.ingest.tasks import recover_stale_tasks
from src.repository import User, cleanup_metrics, get_user
from src.store.chroma import VectorStore
from src.store.db import init_db
from src.ui import about, admin, chat, documents, login, theme
from src.ui import settings as ui_settings

logger = logging.getLogger(__name__)

QUERY_KEY = "page"

# 导航项。icon 用 Streamlit 的 Material 图标语法。
# 设计原指定 lucide 图标（docs/07 §7），但导航已改为按钮实现（见下），
# Streamlit 按钮只接受 Material 图标，这处差异已登记在自审清单。
PAGES = (
    {"key": "qa", "label": "问答", "icon": ":material/forum:", "plan": ""},
    {"key": "documents", "label": "我的文档", "icon": ":material/description:", "plan": ""},
    {
        "key": "admin",
        "label": "管理员",
        "icon": ":material/manage_accounts:",
        "plan": "",
    },
    {"key": "settings", "label": "设置", "icon": ":material/settings:", "plan": ""},
    {"key": "about", "label": "关于", "icon": ":material/info:", "plan": ""},
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
    # M3-08：清理超过保留期的问答指标（docs/02 §6.3）。没有定时任务组件，
    # 就借启动钩子做，随每次部署自然执行一次。
    purged = cleanup_metrics()
    logger.info(
        "启动完成：数据目录 %s，回收中断任务 %d 条，清理过期指标 %d 条",
        settings.data_dir,
        recovered,
        purged,
    )


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


def current_user(*, settings) -> User | None:
    """取当前登录用户；无登录态或账号已失效时返回 None。

    账号被删除或被禁用时**立即失效**：否则管理员刚禁用一个人，
    那个人的浏览器还处在已登录状态，要等他重新登录才被拦住。
    """
    user_id = st.session_state.get(login.STATE_USER_ID)
    if user_id is None:
        return None

    user = get_user(user_id, db_path=settings.database_path)
    if user is None or not user.is_active:
        st.session_state.pop(login.STATE_USER_ID, None)
        return None
    return user


def goto_page(page_key: str) -> None:
    """切换页面：只改查询参数并重跑，**绝不整页跳转**。

    为什么不用 `<a href="?page=x">`：链接跳转会重新加载整个页面、重建 Streamlit
    会话，而登录态正存在会话里——点一下导航就被踢回登录页（FB-3.2 浏览器验收
    实测到的缺陷）。按钮走的是 WebSocket 事件，不重载页面，会话与登录态都不动。

    仍然写查询参数，是为了满足 G-04「刷新后保持当前页」：重新登录后会落回原页。
    """
    st.query_params[QUERY_KEY] = page_key
    st.rerun()


def render_sidebar(current: str, *, user: User) -> None:
    """渲染左侧导航与账号区。

    G-03：`user` 角色不显示「管理员」入口。侧边栏顺序为
    品牌 → 导航按钮 → 当前账号 → 退出登录 →（问答页的会话列表由 chat 追加）。
    """
    school = load_kb_config()["school"]["name"]
    pages = [page for page in PAGES if page["key"] != "admin" or user.is_admin]
    account = html.escape(user.display_name or user.username)
    role = "管理员" if user.is_admin else "学生"

    with st.sidebar:
        st.markdown(
            f'<div class="cqa-brand"><div class="cqa-brandmark">校</div>'
            f"<div><div class=\"cqa-brandname\">校答</div>"
            f'<div class="cqa-branddesc">{html.escape(str(school))}</div></div></div>',
            unsafe_allow_html=True,
        )

        for page in pages:
            if st.button(
                page["label"],
                key=f"nav-{page['key']}",
                icon=page["icon"],
                use_container_width=True,
                type="primary" if page["key"] == current else "secondary",
            ):
                goto_page(page["key"])

        st.markdown(
            f'<div class="cqa-railfoot">{account} · {role}</div>', unsafe_allow_html=True
        )
        if st.button(
            "退出登录", key="logout", icon=":material/logout:", use_container_width=True
        ):
            login.logout()


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

    settings = get_settings()
    user = current_user(settings=settings)
    if user is None:
        # G-01 访客拦截：未登录只渲染登录页，不产出任何页面内容
        login.render(settings=settings)
        return

    current = resolve_page(st.query_params.get(QUERY_KEY))
    if not service.can_access_page(user, current):
        # G-02 角色守卫：拦截后就地提示，并回到问答页
        st.warning("无权访问该页面")
        current = DEFAULT_PAGE

    render_sidebar(current, user=user)

    if current == "qa":
        chat.render(settings=settings, store=get_vector_store(), user=user)
    elif current == "documents":
        documents.render(settings=settings, store=get_vector_store(), user=user)
    elif current == "admin":
        admin.render(settings=settings, store=get_vector_store(), user=user)
    elif current == "settings":
        ui_settings.render(settings=settings, user=user, store=get_vector_store())
    elif current == "about":
        about.render(settings=settings)
    else:
        render_placeholder(PAGES_BY_KEY[current])


# Streamlit 以 __main__ 执行本脚本；加这层判断是为了让测试可以安全 import
# （否则 import app 会直接把整个应用跑一遍）
if __name__ == "__main__":
    main()
