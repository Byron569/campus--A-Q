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


class AuthError(CampusQAError):
    """认证或注册校验失败（用户名密码错误、协议未勾选、两次密码不一致等）。

    文案直接面向用户，就地显示在表单字段下方（docs/05 §G-06）。
    设计文档的接口文档未单列本异常，与 `src/errors.py` 的创建理由相同：
    认证失败需要被 UI 层统一捕获，集中定义可避免各处重复判断。
    """


class AccountDisabled(AuthError):
    """被禁用的账号尝试登录。对应 docs/02 §11：提示「账号已被禁用」。"""


class UsernameTaken(AuthError):
    """注册时用户名已被占用（FR-01：用户名唯一）。"""


class PermissionDenied(CampusQAError):
    """角色不足（如 user 访问管理员页）。对应 docs/05 §G-02：提示「无权访问该页面」。"""


class LLMUnavailable(CampusQAError):
    """大模型调用不可用（超时 / 401 / 欠费 / 网络中断）。

    对应 docs/02 §4.6：上层据此走降级，展示检索到的原文片段。
    """


class BackupError(CampusQAError):
    """备份 / 恢复失败（FR-24，二期）。

    文案面向管理员：备份文件损坏、版本不支持、目标库非空却未授权覆盖等。
    """


class SummarizeError(CampusQAError):
    """课件总结 / 复习提纲生成失败（二期 2.4）。

    文案面向用户：内容为空、模型调用失败或返回空结果。
    """
