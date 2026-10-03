#!/usr/bin/env bash
# 启动校答（campus-qa）网页服务。
#
# HF_ENDPOINT：本地 Embedding 首次运行需从 HuggingFace 下载模型权重（约 100 MB）。
# 默认走社区镜像 hf-mirror.com；若你的网络能直连官方，可先
#   export HF_ENDPOINT=https://huggingface.co
# 覆盖本脚本的默认值。
set -euo pipefail

cd "$(dirname "$0")"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [ ! -f app.py ]; then
  echo "未找到 app.py，网页端尚未实现（见 docs/03 的 M1-18）。" >&2
  exit 1
fi

exec .venv/bin/python -m streamlit run app.py \
  --server.address "${HOST:-0.0.0.0}" \
  --server.port "${PORT:-8501}"
