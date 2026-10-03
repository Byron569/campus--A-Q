"""测试共享夹具。

`FakeEmbeddings` 提供确定性字符袋向量，不下载模型，离线可快速跑完。
注意它只保证「字面重叠多 → 相似度高」，**不代表真实语义相似度**，
因此涉及语义效果的断言不能放在这里，只能靠真实模型手测。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from config.settings import Settings
from src.store.chroma import VectorStore
from src.store.db import get_conn, init_db


class FakeEmbeddings(Embeddings):
    """确定性字符袋向量：文本越相似，余弦相似度越高。无需网络。"""

    dim = 32

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.dim
        for char in text:
            values[ord(char) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """隔离的配置：数据目录指向 tmp_path，不碰真实 data/。"""
    return Settings(_env_file=None, data_dir=tmp_path, chroma_dir=tmp_path / "chroma")


@pytest.fixture()
def store(settings: Settings) -> VectorStore:
    return VectorStore(settings=settings, embeddings=FakeEmbeddings())


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    """隔离的 SQLite：建表并预置 users 1 / 2。

    `documents.user_id` 有外键约束（docs/02 §5.1），测试里用到用户身份时
    必须先把用户造出来，否则写入会因 FOREIGN KEY 失败。
    """
    path = tmp_path / "app.db"
    init_db(path)
    with get_conn(path) as conn:
        conn.executemany(
            "INSERT INTO users (id, username, password_hash) VALUES (?, ?, ?)",
            [(1, "user-1", "x"), (2, "user-2", "x")],
        )
    return path
