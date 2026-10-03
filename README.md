# 校答（campus-qa）

面向高校的校园知识库 RAG 问答系统。把散落在通知、学生手册、课件里的资料收进本地知识库，
用「检索 + 强制引用」的方式作答，答案可追溯回原文片段，避免大模型编造学校政策。

一期范围：校园知识库问答（OCR、日程管理、Agent 工具调度归二期）。

## 当前进度

| 里程碑 | 状态 |
| --- | --- |
| M1 数据底座（能入库、能检索） | 进行中：命令行入库与检索已可用；网页端（M1-18）待做 |
| M2 问答与界面 | 未开始 |
| M3 账号、隔离与跨设备 | 未开始 |

> 本文档为初版。架构图、评测效果数据与界面截图位将在 M3-15 补齐。

## 环境要求

- Python 3.11 及以上（开发环境为 3.12）
- macOS 或 Windows，64 位
- 不需要 Docker，不需要 Ollama
- 首次运行需联网下载本地 Embedding 模型（约 100 MB）

## 安装

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # 然后填入 LLM_API_KEY
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
| `.env` | 模型供应商、检索参数、端口与路径 | 否（含密钥） |
| `config/knowledge_base.yaml` | 学校信息与知识库分类 | 是 |

换一所学校只需改 `knowledge_base.yaml` 与资料，不用动代码。全部配置项见 `docs/02-架构设计.md` §10。

## 使用

```bash
# 1. 生成 8 份模拟校园资料（真实资料未到位时用于开发与验证）
python scripts/make_samples.py

# 2. 按分类批量入库（公共文档）
python scripts/ingest_cli.py data/samples/freshman --public --category freshman
python scripts/ingest_cli.py data/samples/admin    --public --category admin
python scripts/ingest_cli.py data/samples/course   --public --category course

# 3. 检索验证（M1 阶段只验证向量路，BM25 混合检索在 M2 实现）
python scripts/ingest_cli.py --query "搬宿舍需要提前申请吗"

# 4. 启动网页
./run.sh                          # Windows: run.bat
```

局域网访问：手机 / 其他电脑打开 `http://<本机内网IP>:8501`。

## 测试

```bash
pytest
```

## 项目结构

```
app.py                 入口与路由（M1-18 起）
config/                配置中心与知识库业务配置
src/ingest/            解析、切片、入库流水线、异步任务
src/store/             SQLite 与 Chroma 封装（向量访问唯一入口）
src/providers/         LLM 与 Embedding 可插拔层
src/repository.py      全部数据访问
scripts/               模拟资料生成、批量入库与检索 CLI
docs/                  需求、架构、任务、原型、接口、设计与开发计划
tests/                 单元测试
data/                  运行期数据（不入库）
```

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
