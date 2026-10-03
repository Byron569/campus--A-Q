"""项目自定义异常。

设计依据：docs/06-接口文档.md §4.1 自定义异常
说明：设计文档的目录结构未列出本文件，但异常需要被多个模块（config / providers /
ingest / store / auth / rag）共同引用，集中定义可避免循环依赖与重复定义。
此偏离已在自审清单中登记。
"""

from __future__ import annotations


class CampusQAError(Exception):
    """本项目所有自定义异常的基类，便于上层统一捕获。"""


class ConfigError(CampusQAError):
    """配置文件缺失或格式错误。对应 docs/02 §11：启动即失败，日志给出示例。"""


class ProviderError(CampusQAError):
    """模型供应商未配置、API Key 缺失或 provider 取值未知。"""


class UnsupportedFileType(CampusQAError):
    """上传文件后缀不在白名单内。对应 docs/02 §11：单文件报错，不影响同批。"""


class ParseError(CampusQAError):
    """文件损坏、已加密或解析器异常。对应 docs/02 §11：任务置 failed，可重试。"""


class IngestError(CampusQAError):
    """入库过程中的错误（向量化失败等）。对应 docs/02 §11：任务置 failed，可重试。"""


class PermissionFilterError(CampusQAError):
    """检索未携带用户身份。

    安全红线（FR-31 / docs/02 §4.8）：宁可抛错，也绝不降级为「无过滤」返回数据。
    """


class AccountDisabled(CampusQAError):
    """被禁用的账号尝试登录。对应 docs/02 §11：提示「账号已被禁用」。"""


class LLMUnavailable(CampusQAError):
    """大模型调用不可用（超时 / 401 / 欠费 / 网络中断）。

    对应 docs/02 §4.6：上层据此走降级，展示检索到的原文片段。
    """
