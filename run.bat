@echo off
REM 启动校答（campus-qa）网页服务。
REM
REM HF_ENDPOINT：本地 Embedding 首次运行需从 HuggingFace 下载模型权重（约 100 MB）。
REM 默认走社区镜像 hf-mirror.com；若你的网络能直连官方，可先
REM   set HF_ENDPOINT=https://huggingface.co
REM 覆盖本脚本的默认值。
REM
REM 依赖安装（CPU 版 torch，避免拉到体积数百 MB 的 GPU 版本）：
REM   .venv\Scripts\python.exe -m pip install -r requirements-windows.txt
REM
REM 提示：本脚本尚未在真实 Windows 机器上实测（M3-A5 跨设备验收需实机确认），
REM 若该机防火墙拦截，请放行入站 TCP 8501（见 README「局域网访问」）。
setlocal
cd /d "%~dp0"

if not defined HF_ENDPOINT set HF_ENDPOINT=https://hf-mirror.com
if not defined HOST set HOST=0.0.0.0
if not defined PORT set PORT=8501

if not exist app.py (
  echo 未找到 app.py，网页端尚未实现（见 docs/03 的 M1-18）。 1>&2
  exit /b 1
)

.venv\Scripts\python.exe -m streamlit run app.py --server.address %HOST% --server.port %PORT%
