"""存储层：SQLite（关系数据）与 Chroma（向量）的封装。

设计依据：docs/02-架构设计.md §5、docs/06-接口文档.md §1.4
"""

from config.settings import COLLECTION_NAME, PUBLIC_USER_ID

__all__ = ["COLLECTION_NAME", "PUBLIC_USER_ID"]
