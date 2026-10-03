"""全站主题：把设计令牌注入 Streamlit。

设计依据：docs/07-设计令牌.md（§3 令牌、§4 排版、§5 间距圆角阴影、§8 Streamlit 落地映射）
设计源：`docs/design/doubao/colors_and_type.css`（Doubao Design Library）

为什么必须注入 CSS：`config.toml` 只能配主色、背景、文字色、字体这五项，
圆角、描边、阴影、间距、组件形态都不支持，只能靠 CSS 覆盖（docs/07 §8.1 注）。

两条硬约束：
- **仅浅色**（CR-01）：不注入模板的 `.dark` 块。
- **键盘焦点必须可见**（DR-13）：所有可聚焦元素保留 `--ring` 焦点样式，
  禁止整体 `outline: none`。

本模块只负责「注入样式」，不产生任何业务行为；`apply_theme()` 可在每次脚本
运行时重复调用（Streamlit 每次交互都会重跑整个脚本）。
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# 一、设计令牌
# 逐条取自 docs/design/doubao/colors_and_type.css 的亮色块，仅两处按项目裁决调整：
#   1. --muted-foreground 由模板的 #7f8d9f 加深为 #5b6675（DR-04，对比度不足）
#   2. --radius 拆成三级（D-03）：控件 10px / 卡片 14px / 胶囊 999px
#   3. 新增 --success / --info 两个语义色（DR-02，模板只有 destructive）
# ---------------------------------------------------------------------------

TOKENS = """
:root {
  /* 语义色 */
  --background: #ffffff;
  --foreground: #0e1115;
  --card: #ffffff;
  --card-foreground: #0e1115;
  --popover: #f9f9fa;
  --popover-foreground: #0e1115;

  --primary: #0065fd;
  --primary-foreground: #ffffff;
  --secondary: #eff1f4;
  --secondary-foreground: #333942;
  --muted: #eff1f4;
  --muted-foreground: #5b6675;   /* DR-04：模板原值 #7f8d9f 对比度不足 */
  --accent: #e5e9ff;
  --accent-foreground: #00266b;
  --destructive: #ef4444;
  --destructive-foreground: #ffffff;
  --border: #e7eaef;
  --input: #e7eaef;
  --ring: #557fff;

  /* 语义色扩展（DR-02：模板未提供 success / info，入库状态无颜色可用） */
  --success: #16a34a;
  --info: #0ea5e9;

  /* 程度梯度：只表示「强度」，禁止用于区分类别（A-06 硬约束） */
  --chart-1: #557fff;
  --chart-2: #0065fd;
  --chart-3: #0057da;
  --chart-4: #0043ad;
  --chart-5: #002e7d;

  /* 侧边栏 */
  --sidebar: #eff1f4;
  --sidebar-foreground: #0e1115;
  --sidebar-primary: #0065fd;
  --sidebar-primary-foreground: #ffffff;
  --sidebar-accent: #d4daff;
  --sidebar-accent-foreground: #00266b;
  --sidebar-border: #e7eaef;

  /* 字体：补中文字体栈（A-01） */
  --font-sans: "Stack Sans Text", "PingFang SC", "Microsoft YaHei", "Noto Sans SC",
               ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-serif: "Source Serif 4", "Songti SC", serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;

  /* 圆角三级（D-03） */
  --radius-control: 10px;
  --radius-card: 14px;
  --radius-pill: 999px;

  /* 间距（A-03：模板的 0.24rem 只留给表格/列表行内） */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-6: 24px;
  --space-8: 32px;

  /* 弹窗轻阴影（D-02：仅弹窗允许，卡片一律不用阴影） */
  --shadow-dialog: 0 8px 24px rgba(14, 17, 21, 0.10);
}
"""

# ---------------------------------------------------------------------------
# 二、Streamlit 控件覆盖
# docs/07 §8.2 列出的每一项都在下面有对应规则；组件形态取自模板 preview：
# 按钮为胶囊（height 38 / sm 32 / lg 44），输入类控件 10px 圆角 + --muted 底，
# 卡片与面板 14px 圆角 + 1px 描边 + 无阴影。
# ---------------------------------------------------------------------------

OVERRIDES = """
/* ---------- 基础 ---------- */
.stApp, [data-testid="stAppViewContainer"] {
  background: var(--background);
  color: var(--foreground);
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.6;                 /* 中文行高不低于 1.5（docs/07 §4.2） */
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] {
  padding-top: var(--space-6);
  padding-bottom: var(--space-8);
  max-width: 1180px;
}
[data-testid="stDecoration"] { display: none; }   /* 去掉顶部彩色装饰条 */

/* 字号阶梯（docs/07 §4.2） */
h1, [data-testid="stHeading"] h1 { font-size: 24px !important; line-height: 34px; font-weight: 600; }
h2, [data-testid="stHeading"] h2 { font-size: 18px !important; line-height: 28px; font-weight: 500; }
h3, [data-testid="stHeading"] h3 { font-size: 15px !important; line-height: 24px; font-weight: 500; }
[data-testid="stMarkdownContainer"] p { font-size: 14px; line-height: 1.6; }
[data-testid="stCaptionContainer"], .stCaption, small {
  font-size: 13px !important; color: var(--muted-foreground) !important;
}
hr, [data-testid="stDivider"] { border-color: var(--border) !important; }

/* ---------- 按钮：胶囊，高度 38px ---------- */
/* Streamlit 基础规则用 min-height: 2.5rem 定高，只写 height 压不住，必须带 !important */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  height: 38px !important;
  min-height: 38px !important;
  padding: 0 var(--space-4) !important;
  border: 1px solid var(--border);
  border-radius: var(--radius-pill) !important;
  background: var(--background);
  color: var(--foreground);
  font: 500 13px/1 var(--font-sans) !important;
  box-shadow: none;
  transition: background-color .16s, border-color .16s, color .16s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--muted);
  border-color: var(--border);
  color: var(--foreground);
}
.stButton > button:active { transform: translateY(1px); }
.stButton > button:focus-visible {
  outline: 2px solid var(--ring) !important;
  outline-offset: 2px;
}
.stButton > button:disabled, .stButton > button[disabled] {
  opacity: .45; cursor: not-allowed;
}
/* 主按钮：品牌蓝 */
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: var(--primary);
  border-color: transparent;
  color: var(--primary-foreground);
}
.stButton > button[kind="primary"]:hover {
  background: color-mix(in srgb, var(--primary) 90%, var(--foreground));
  border-color: transparent;
  color: var(--primary-foreground);
}
/* 破坏性按钮：我们通过 class 标记传入 */
.stButton > button.cqa-destructive {
  background: var(--destructive);
  border-color: transparent;
  color: var(--destructive-foreground);
}

/* ---------- 文件上传区：做成卡片（模板无上传组件，按 app-card 语言派生） ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: var(--space-4);
  min-height: 96px;
  transition: border-color .16s, background .16s;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: color-mix(in srgb, var(--primary) 40%, var(--border));
  background: color-mix(in srgb, var(--accent) 22%, var(--card));
}
[data-testid="stFileUploaderDropzone"] button {
  border-radius: var(--radius-pill) !important;
  border: 1px solid var(--border) !important;
  background: var(--background) !important;
  color: var(--foreground) !important;
  font: 500 12px/1 var(--font-sans) !important;
  height: 32px !important;
  padding: 0 var(--space-3) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span {
  color: var(--foreground); font-size: 13px;
}
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--muted-foreground) !important; }
/* 已选文件列表 */
[data-testid="stFileUploaderFile"] {
  background: var(--muted);
  border-radius: var(--radius-control);
  padding: var(--space-1) var(--space-2);
  margin-top: var(--space-1);
}

/* ---------- 输入类控件：--muted 底 + 10px 圆角，聚焦换成卡片底 + 焦点环 ----------
   注意：当前 Streamlit 版本的下拉框已不是 data-baseweb，而是 react-aria ComboBox，
   真实结构为 [data-testid="stSelectbox"] > .react-aria-ComboBox > div[role="group"]。
   这里同时保留 data-baseweb 写法，避免升级/降级时静默失效。 */
[data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"],
[data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
  background: var(--muted) !important;
  border: 1px solid transparent !important;
  border-radius: var(--radius-control) !important;
  color: var(--foreground) !important;
  font-family: var(--font-sans) !important;
  min-height: 38px !important;
  box-shadow: none !important;
}
[data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"]:focus-within,
[data-baseweb="select"] > div:focus-within,
[data-testid="stTextInput"] input:focus {
  background: var(--card) !important;
  border-color: var(--ring) !important;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 18%, transparent) !important;
}
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] {
  font-size: 13px !important; font-weight: 500; color: var(--foreground) !important;
}

/* ---------- 进度条：细轨道 + info 色 ---------- */
[data-testid="stProgress"] > div > div {
  background: var(--muted);
  border-radius: var(--radius-pill);
  height: 6px;
}
[data-testid="stProgress"] > div > div > div {
  background: var(--info) !important;
  border-radius: var(--radius-pill);
}
[data-testid="stProgress"] p { font-size: 12px; color: var(--muted-foreground); }

/* ---------- 提示条：无阴影、圆角 10px ---------- */
[data-testid="stAlert"] {
  border-radius: var(--radius-control);
  border: 1px solid var(--border);
  box-shadow: none;
  font-size: 13px;
  padding: var(--space-2) var(--space-3);
}
[data-testid="stAlertContentSuccess"] { color: var(--success); }
[data-testid="stAlertContentInfo"] { color: var(--info); }
[data-testid="stAlertContentError"] { color: var(--destructive); }

/* ---------- 弹窗（二次确认）：允许一层轻阴影（D-02） ---------- */
[data-testid="stPopoverBody"] {
  background: var(--popover);
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  box-shadow: var(--shadow-dialog);
  padding: var(--space-3);
}
[data-testid="stPopover"] > div > button {
  border-radius: var(--radius-pill) !important;
  height: 32px !important;
  font-size: 12px !important;
}

/* ---------- 侧边栏 ---------- */
[data-testid="stSidebar"] {
  background: var(--sidebar);
  border-right: 1px solid var(--sidebar-border);
}
/* 宽度对齐设计模板的 rail（248px）。
   Streamlit 的宽度写在内联 style="width:300px" 上，min/max 写在随版本变化的
   emotion 类上，所以必须 !important 覆盖。
   只作用于展开态：折叠态靠 aria-expanded="false" 的 min/max 0 + translateX
   实现，这里不碰，避免把折叠动画弄坏。 */
[data-testid="stSidebar"][aria-expanded="true"] {
  width: 248px !important;
  min-width: 248px !important;
  max-width: 248px !important;
}
/* 宽度既然按设计钉死了，Streamlit 自带的拖拽调宽手柄就成了摆设，直接隐掉。
   注意：手柄外层还有一层无 testid / 无 class 的匿名包裹（8px 绝对定位），
   隐掉手柄后它仍会让侧栏的 scrollWidth 比 clientWidth 大 6px。该残留实测
   无可见影响（内部滚动容器的 scrollWidth == clientWidth，不出现滚动条，
   也不裁切任何内容），且没有稳定的选择器可选中它，故不加 hack 处理。 */
[data-testid="stSidebarResizeHandle"] { display: none !important; }
[data-testid="stSidebar"] * { color: var(--sidebar-foreground); }

/* ---------- 布局 ---------- */
[data-testid="stVerticalBlock"] { gap: var(--space-3); }
/* 注意：不要试图用 [data-testid="stVerticalBlockBorderWrapper"] 给容器加边框——
   当前 Streamlit 版本该 testid 已不存在，带边框的容器就是 stVerticalBlock 本身，
   边框写在随版本变化的 emotion 类上。页面因此不使用 st.container(border=True)，
   改用自有 .cqa-row 画边框，避免依赖内部类名。 */

/* ---------- 窄屏侧栏：不再遮挡主内容（浏览器实测的缺陷） ----------
   实测（604px 视口）：侧栏展开时 `stSidebar` 占左侧 248px、z-index 极高，
   而主内容 `stMain` 是 `position:absolute; left:0; right:0` 满宽、**不随侧栏右移**，
   于是左边 248px 的内容被压在侧栏下面且没有任何遮罩提示。分两档处理：

   - 小窗口 / 平板（481–900px）：把主内容整体右移一个侧栏宽度，二者并排，互不遮挡；
   - 手机（≤480px）：保持「抽屉浮层」这一手机通用形态（此时并排只剩一两百像素，
     反而不可用），只把抽屉宽度收到不超过 82vw，留出一条可点击的内容带。

   之所以用 `:has()`：Streamlit 未提供「侧栏是否展开」的稳定类名，只能按
   `aria-expanded` 反查父容器（Chrome 105+ / Safari 15.4+ / Firefox 121+ 支持）。 */
@media (min-width: 481px) and (max-width: 900px) {
  [data-testid="stSidebar"][aria-expanded="true"] {
    width: 248px !important;
    min-width: 248px !important;
    max-width: 248px !important;
  }
  /* 主内容既要右移、也要同比收窄：只加 margin-left 会把整块推到视口外
     （实测主内容 width 仍为视口宽度，右边界超出 248px，出现横向溢出）。 */
  [data-testid="stAppViewContainer"]:has([data-testid="stSidebar"][aria-expanded="true"])
    [data-testid="stMain"] {
    margin-left: 248px !important;
    width: calc(100% - 248px) !important;
    max-width: calc(100% - 248px) !important;
  }
}
@media (max-width: 480px) {
  [data-testid="stSidebar"][aria-expanded="true"] {
    width: min(248px, 82vw) !important;
    min-width: min(248px, 82vw) !important;
    max-width: min(248px, 82vw) !important;
  }
}
@media (max-width: 900px) {
  /* 收起按钮默认靠 :hover 显形，而触屏没有 hover——手机上侧栏一旦展开就关不掉。
     窄屏下让它常显（浏览器实测该 testid 为稳定公开选择器）。 */
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stSidebarCollapseButton"] button {
    visibility: visible !important;
    opacity: 1 !important;
  }
}
"""

# ---------------------------------------------------------------------------
# 三、本项目自有样式
# Streamlit 的列/文本写不出「卡片 + 边框行 + 状态胶囊」，这几类由页面拼 HTML 时引用。
# ---------------------------------------------------------------------------

COMPONENTS = """
.cqa-page-title { font-size: 24px; line-height: 34px; font-weight: 600; margin: 0 0 4px; }
.cqa-page-desc { font-size: 13px; line-height: 20px; color: var(--muted-foreground); margin: 0 0 4px; }

/* 面板（对应模板 data-table 的 .panel） */
.cqa-panel {
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  background: var(--card);
  overflow: hidden;
}
.cqa-panel-head {
  display: flex; align-items: center; gap: var(--space-2);
  padding: 14px 16px; border-bottom: 1px solid var(--border);
}
.cqa-panel-head h3 { margin: 0; font-size: 15px; font-weight: 600; }
.cqa-panel-head .cqa-sub { font-size: 12px; color: var(--muted-foreground); }

/* 列表行：1px 描边 + 14px 圆角 + 无阴影（对应模板 data-table 的行样式） */
.cqa-row {
  display: flex; align-items: center; gap: var(--space-3);
  padding: 12px var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  background: var(--card);
  transition: border-color .16s, background .16s;
}
.cqa-row:hover {
  border-color: color-mix(in srgb, var(--primary) 40%, var(--border));
  background: color-mix(in srgb, var(--accent) 16%, var(--card));
}
.cqa-rowmain { flex: 1; min-width: 0; }
.cqa-rowmain .cqa-name {
  font-size: 13px; font-weight: 500;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.cqa-rowmain .cqa-meta { font-size: 12px; color: var(--muted-foreground); }
.cqa-rowstatus { flex: none; }
/* 分页信息：与按钮垂直居中对齐 */
.cqa-pager {
  height: 38px; display: flex; align-items: center; justify-content: center;
  font-size: 13px; color: var(--muted-foreground);
}

/* 状态胶囊（对应模板 .status-chip）：成功 / 进行 / 等待 / 失败 */
.cqa-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: var(--radius-pill);
  font-size: 12px; font-weight: 500; white-space: nowrap;
}
.cqa-chip::before {
  content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor;
}
.cqa-chip--ok   { background: color-mix(in srgb, var(--success) 14%, transparent); color: var(--success); }
.cqa-chip--run  { background: color-mix(in srgb, var(--info) 14%, transparent);    color: var(--info); }
.cqa-chip--wait { background: color-mix(in srgb, var(--muted-foreground) 14%, transparent); color: var(--muted-foreground); }
.cqa-chip--fail { background: color-mix(in srgb, var(--destructive) 14%, transparent); color: var(--destructive); }

/* 文件类型徽标（对应模板 .ic，用程度梯度而非类别色，合规 A-06） */
.cqa-fileicon {
  width: 28px; height: 28px; flex: none; border-radius: 8px;
  display: grid; place-items: center;
  font: 600 10px/1 var(--font-mono); letter-spacing: .02em;
  background: color-mix(in srgb, var(--chart-1) 16%, transparent); color: var(--chart-3);
}
.cqa-empty {
  padding: var(--space-8) var(--space-4);
  text-align: center; color: var(--muted-foreground); font-size: 13px;
}

/* 侧边栏（对应模板 sidebar-nav）
   导航改用 Streamlit 按钮实现，**不再是 <a>**：链接跳转会重建 Streamlit 会话，
   把存在会话里的登录态一并丢掉（FB-3.2 浏览器实测的缺陷）。
   因此这里不再保留链接式导航的自有样式类，改为直接约束按钮外观。 */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stSidebar"] [data-testid="stElementContainer"] { padding: 0 !important; }

[data-testid="stSidebar"] .cqa-brand {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 11px 14px;
}
[data-testid="stSidebar"] .cqa-brandmark {
  width: 30px; height: 30px; flex: none; display: grid; place-items: center;
  border-radius: 9px;
  font: 600 14px/1 var(--font-sans); color: var(--primary-foreground);
  background: linear-gradient(135deg, var(--chart-1), var(--chart-4));
}
[data-testid="stSidebar"] .cqa-brandname { font-size: 14px; font-weight: 600; }
[data-testid="stSidebar"] .cqa-branddesc {
  font-size: 12px; color: var(--muted-foreground);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
[data-testid="stSidebar"] .cqa-railfoot {
  margin-top: var(--space-3); padding: 10px 11px;
  border-top: 1px solid var(--sidebar-border);
  font-size: 12px; color: var(--muted-foreground);
}

/* 导航 / 会话 / 退出登录 按钮统一成「左侧列表项」外观 */
[data-testid="stSidebar"] .stButton { margin: 0 0 2px; }
[data-testid="stSidebar"] .stButton > button {
  width: 100%; height: 38px !important; min-height: 38px !important;
  justify-content: flex-start;
  border-radius: var(--radius-control) !important;
  border-color: transparent;
  background: transparent;
  font-size: 14px !important; font-weight: 400;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: color-mix(in srgb, var(--background) 64%, var(--sidebar));
}
/* 当前页：模板的选中项是白底加粗，这里用主色按钮表达选中 */
[data-testid="stSidebar"] .stButton > button[kind="primary"] { font-weight: 600; }
[data-testid="stSidebar"] .stButton > button:focus-visible {
  outline: 2px solid var(--ring) !important; outline-offset: 2px;
}

/* ---------- 问答页（PG-02）---------- */

/* 检索 / 生成过程中的就地提示 */
.cqa-hint {
  display: inline-flex; align-items: center; gap: 8px;
  padding: var(--space-2) 0;
  font-size: 13px; color: var(--muted-foreground);
}
.cqa-hint::before {
  content: ""; width: 8px; height: 8px; border-radius: 50%;
  background: var(--info);
}

/* 聊天气泡：1px 描边 + 14px 圆角 + 无阴影（模板无阴影设计 A-04）。
   注意用 stChatMessage 的公开 testid，不用随版本变化的 emotion 类名。
   文档已知差异（docs/07 §8.4）：Streamlit 的 chat_message 一律靠左，不做左右分栏，
   这里沿用其默认布局，用头像区分角色。 */
[data-testid="stChatMessage"] {
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  background: var(--card);
  box-shadow: none;
  padding: var(--space-3) var(--space-4);
  margin-bottom: var(--space-3);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child { margin-bottom: 0; }

/* 引用来源卡片：--accent 高亮 + 虚线边框（docs/07 §8.2 app-card）。
   用 st.expander 承载「展开原文」，其 testid 是长期稳定的公开选择器；
   万一失效只会退回默认外观，不影响功能。 */
[data-testid="stExpander"] {
  border: 1px dashed color-mix(in srgb, var(--primary) 34%, var(--border)) !important;
  border-radius: var(--radius-control) !important;
  background: color-mix(in srgb, var(--accent) 42%, var(--card)) !important;
  box-shadow: none !important;
  overflow: hidden;
}
[data-testid="stExpander"] summary { font-size: 13px !important; font-weight: 500; padding: 8px 12px !important; }
[data-testid="stExpander"] summary:hover {
  background: color-mix(in srgb, var(--accent) 66%, var(--card)) !important;
}
[data-testid="stExpander"] summary:focus-visible { outline: 2px solid var(--ring) !important; outline-offset: -2px; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 0 12px 10px !important; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p { font-size: 13px; }

/* 答案操作行（反馈 / 复制）与下载按钮的紧凑间距 */
.cqa-answer-actions { margin-top: var(--space-1); }

/* 侧边栏会话列表（PG-02 左侧会话列表，与共用导航合并到同一侧栏） */
[data-testid="stSidebar"] .cqa-conv-head {
  margin-top: var(--space-2);
  padding: 10px 11px 4px;
  border-top: 1px solid var(--sidebar-border);
  font-size: 12px; font-weight: 600; color: var(--muted-foreground);
}
/* 会话行 = 会话名 + ⋯ 菜单，必须留在同一行。
   Streamlit 的列默认 flex-wrap: wrap 且列有 min-width，248px 侧栏里放不下就会
   把 ⋯ 挤到第二行；这里改成不换行并允许列收缩到 0。 */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
  flex-wrap: nowrap !important;
  gap: var(--space-1) !important;
}
[data-testid="stSidebar"] [data-testid="stColumn"] { min-width: 0 !important; }
[data-testid="stSidebar"] .stPopover { margin-bottom: 6px; }
[data-testid="stSidebar"] .stPopover > div > button {
  width: 100%; padding: 0 !important; justify-content: center;
}

/* 登录 / 注册页（PG-01）：未登录时的唯一入口，居中品牌区 + 表单卡片 */
.cqa-login-hero { text-align: center; padding: var(--space-8) 0 var(--space-6); }
.cqa-login-mark {
  width: 44px; height: 44px; margin: 0 auto var(--space-3);
  border-radius: 13px; display: grid; place-items: center;
  font: 600 20px/1 var(--font-sans); color: var(--primary-foreground);
  background: linear-gradient(135deg, var(--chart-1), var(--chart-4));
}
.cqa-login-title { font-size: 24px; line-height: 34px; font-weight: 600; }
.cqa-login-sub { font-size: 13px; color: var(--muted-foreground); margin-top: 2px; }
"""


def stylesheet() -> str:
    """返回完整样式表，便于测试断言与排查。"""
    return TOKENS + OVERRIDES + COMPONENTS


def apply_theme() -> None:
    """注入主题样式。每次脚本运行调用一次即可。"""
    st.markdown(f"<style>{stylesheet()}</style>", unsafe_allow_html=True)
