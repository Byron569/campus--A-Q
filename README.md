# 校答（campus-qa）

面向高校的校园知识库 RAG 问答系统。把散落在通知、学生手册、课件里的资料收进本地知识库，
用「检索 + 强制引用」的方式作答，答案可追溯回原文片段，避免大模型编造学校政策。

一期范围：校园知识库问答（OCR、日程管理、Agent 工具调度归二期）。

## 效果数据

30 题人工评测集（20 可回答 + 5 不可回答 + 5 多轮追问），绑定真实校园资料，实测四项指标：

| 指标 | 结果 | 说明 |
| --- | --- | --- |
| 准确率 | **100%**（30/30） | 可回答题答对、库外题正确拒答 |
| 引用命中率 | **100%**（20/20） | 回答引用的来源命中期望文档 |
| 拒答正确率 | **100%**（5/5） | 库外问题一律拒答，不编造 |
| 追问改写成功率 | **100%**（5/5） | 多轮追问经改写后仍召回正确资料 |

- 知识库：9 份文档、137 个切片，本地 BGE 向量化（文本不出本机）
- 单元测试：`pytest` **299 passed**
- 复现命令见下文「评测」

> 指标公式与评测集字段定义写在 `scripts/run_eval.py` 与 `eval/campus_qa_eval.jsonl` 的注释里。
> 调大模型有随机性，重复运行个别题目可能有措辞差异，指标可能小幅波动。

## 架构

```
┌──────────────────────────────────────────────────────────┐
│  界面层  Streamlit（app.py + src/ui/）                    │
│  问答 / 我的文档 / 管理员 / 设置 / 关于                    │
├──────────────────────────────────────────────────────────┤
│  RAG 层  src/rag/                                         │
│  chain.py    改写 → 检索 → 拒答判定 → 生成 → 引用解析 → 降级│
│  retriever.py 向量路 + BM25 路加权融合（阈值只管向量路）    │
│  rewrite.py  多轮追问改写（失败退化为原问题）              │
├──────────────────────────────────────────────────────────┤
│  接入与入库  src/ingest/                                  │
│  loader.py   解析 PDF / DOCX / TXT / MD                   │
│  splitter.py 中文两阶段切片（标题与其后正文同块）           │
│  pipeline.py 分批向量化 + 写库；tasks.py 后台异步入库       │
├──────────────────────────────────────────────────────────┤
│  存储层  src/store/                                       │
│  chroma.py 向量库唯一入口（强制 user_id 过滤，漏传即抛错）  │
│  db.py     SQLite（WAL + 写锁退避重试）                    │
├──────────────────────────────────────────────────────────┤
│  模型可插拔  src/providers/                               │
│  llm.py（deepseek / dashscope / zhipu / openai / custom）  │
│  embedding.py（本地 BGE / 云端）                           │
└──────────────────────────────────────────────────────────┘
```

分层与关键取舍见 `docs/02-架构设计.md`。

## 界面截图

> 以下为截图位，图片待补（放入 `docs/assets/screenshots/` 即可正常显示）。

| 文件 | 画面 |
| --- | --- |
| `docs/assets/screenshots/01-login.png` | 登录 / 注册页（协议勾选） |
| `docs/assets/screenshots/02-qa.png` | 问答页：回答带 `【来源N】` 与引用卡片 |
| `docs/assets/screenshots/03-documents.png` | 我的文档：上传、进度、列表 |
| `docs/assets/screenshots/04-admin.png` | 管理员页：公共文档、用户管理、五维统计 |
| `docs/assets/screenshots/05-settings.png` | 设置页：显示名 / 改密 / 注销 |

## 环境要求

- Python 3.11 及以上（开发环境为 3.12）
- macOS 或 Windows，64 位
- 不需要 Docker，不需要 Ollama
- 首次运行需联网下载本地 Embedding 模型（约 100 MB）
- OCR（图片 / 扫描件入库）使用 RapidOCR（ONNX），模型随包内置、无需额外下载；
  首次识别会加载约 10 秒，可用 `OCR_ENABLED=false` 关闭

## 安装

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # Windows: pip install -r requirements-windows.txt
cp .env.example .env                 # Windows: copy .env.example .env，然后填入 LLM_API_KEY
```

### 本地 Embedding 模型与下载镜像

默认使用本地模型 `BAAI/bge-small-zh-v1.5`（`EMBEDDING_PROVIDER=local`），文本向量不出本机。
**首次运行需要从 HuggingFace 下载模型权重。**

若 `huggingface.co` 无法直连（国内常见），先设置镜像站再启动：

```bash
export HF_ENDPOINT=https://hf-mirror.com       # Windows: set HF_ENDPOINT=https://hf-mirror.com
```

`run.sh` / `run.bat` 已默认代为设置，可用同名环境变量覆盖。
注意 `hf-mirror.com` 是社区镜像、非官方站点；若你的网络可直连官方，无需设置。

模型下载一次后缓存在 `~/.cache/huggingface`，之后可离线使用。

> 若不想下载模型：改用云端 Embedding，设 `EMBEDDING_PROVIDER=cloud` 并配置 `EMBEDDING_API_KEY`。

## 配置

| 文件 | 内容 | 是否提交 |
| --- | --- | --- |
| `.env` | 模型供应商、检索参数、端口与路径、管理员账号 | 否（含密钥） |
| `config/knowledge_base.yaml` | 学校信息与知识库分类 | 是 |

换一所学校只需改 `knowledge_base.yaml` 与资料，不用动代码。全部配置项见 `docs/02-架构设计.md` §10。

## 使用

```bash
# 1. 初始化管理员账号（读取 .env 的 ADMIN_USERNAME / ADMIN_PASSWORD）
python scripts/seed_users.py

# 2.（可选）生成 8 份模拟校园资料，用于开发与验证
python scripts/make_samples.py
python scripts/ingest_cli.py data/samples/freshman --public --category freshman
python scripts/ingest_cli.py data/samples/admin    --public --category admin
python scripts/ingest_cli.py data/samples/course   --public --category course

# 3.（可选）命令行检索验证
python scripts/ingest_cli.py --query "搬宿舍需要提前申请吗"

# 4. 启动网页
./run.sh                          # Windows: run.bat
```

浏览器打开 `http://127.0.0.1:8501`，用管理员账号登录后即可上传资料并提问。

### 局域网访问（手机 / 其他电脑）

`run.sh` / `run.bat` 默认绑定 `0.0.0.0`，同局域网的设备访问 `http://<本机内网IP>:8501` 即可。

需要放行防火墙：

- **Windows**：首次启动若弹出「Windows 安全中心」提示，勾选「专用网络」并允许；
  若未弹出，到「Windows Defender 防火墙 → 允许应用通过防火墙」放行 Python，
  或手动放行入站 TCP `8501`。
- **macOS**：控制面板「网络 → 防火墙」中允许 Python 接受传入连接（首次启动会弹出询问）。

内网 IP 查看：macOS `ipconfig getifaddr en0`；Windows `ipconfig`（看「IPv4 地址」）。

> 局域网内为明文 HTTP，仅建议在可信校园网络内使用。

## 备份与恢复

```bash
# 导出备份（默认写到 backups/campus-qa-<时间>.zip）
python scripts/backup.py

# 从备份恢复（整库替换，目标库非空时必须加 --force）
python scripts/restore.py backups/campus-qa-20261004-120000.zip --force
```

备份包含 `data/uploads/` 下的原始文件、六张业务表（用户 / 文档 / 会话 / 消息 / 引用 / 反馈）、
`knowledge_base.yaml` 快照；**不含 `.env`、不含问答日志、不含向量**。恢复时会复用入库流水线
**重新向量化**，因此换机器后首次恢复需要能加载 Embedding 模型。

> 恢复是整库替换，会清空现有业务数据；操作前请先导出当前备份。

## 评测

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/run_eval.py
```

输出准确率、引用命中率、拒答正确率、追问改写成功率四项指标；评测写入临时库、跑完即删，
不会影响真实 `data/app.db`。加 `--json` 输出机器可读结果，加 `--verbose` 查看逐题明细。

## 测试

```bash
pytest
```

## 项目结构

```
app.py                  入口与路由、登录态与角色守卫
config/                 配置中心与知识库业务配置
src/ui/                 各页面（问答 / 文档 / 管理员 / 设置 / 关于 / 登录）
src/rag/                改写、混合检索、问答主链路、提示词
src/ingest/             解析、切片、入库流水线、异步任务
src/store/              SQLite 与 Chroma 封装（向量访问唯一入口）
src/auth/               密码哈希与认证服务
src/providers/          LLM 与 Embedding 可插拔层
src/repository.py       全部数据访问
src/backup.py           备份导出与恢复
eval/                   30 题评测集
scripts/                模拟资料生成、批量入库、管理员初始化、备份恢复、评测
docs/                   需求、架构、任务、原型、接口、设计与开发计划
tests/                  单元测试
data/                   运行期数据（不入库）
```

## 安全与隐私

- Embedding 本地运行，文本向量不出本机；问答日志脱敏（不存问题原文与 user_id），保留 90 天后自动清理
- 向量检索强制按 `user_id` 过滤，个人资料仅本人可见；公共文档全员可见
- 密码以 pbkdf2 加盐哈希存储（20 万次迭代），不存明文
- 上传白名单（PDF / DOCX / TXT / MD）与 20 MB 上限

## 文档

| 文档 | 内容 |
| --- | --- |
| `docs/01-需求规格说明书.md` | 32 条功能需求与非功能需求（基线） |
| `docs/02-架构设计.md` | 分层架构、关键模块设计、数据与配置设计 |
| `docs/03-开发任务清单.md` | 三个里程碑的任务与验收标准 |
| `docs/04-需求评审纪要.md` | 需求评审会议记录 |
| `docs/05-产品原型与交互说明.md` | 页面布局与交互 |
| `docs/06-接口文档.md` | 模块接口与数据结构定义 |
| `docs/07-设计令牌.md` | 配色、字体、间距（仅浅色） |
| `docs/08-设计评审纪要.md` | 设计评审会议记录 |
| `docs/09-迭代开发计划.md` | Sprint 划分、DoD、联调矩阵与进度同步格式 |
