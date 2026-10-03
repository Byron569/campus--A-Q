"""密码哈希与校验（FR-01 / NFR-04）。

设计依据：docs/02-架构设计.md §4.8

- 算法：`hashlib.pbkdf2_hmac('sha256', 密码, 随机盐, 200_000)`，只用标准库，零额外依赖
- 存储格式：`算法$迭代次数$盐$哈希`——把迭代次数一起存下来，
  将来调高迭代次数不会导致老密码全部失效（旧哈希仍按自己的次数校验）
- **绝不存明文**；比较用 `hmac.compare_digest`，避免计时侧信道
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from config.settings import PBKDF2_ITERATIONS

ALGORITHM = "pbkdf2_sha256"
MIN_PASSWORD_LENGTH = 6
SALT_BYTES = 16
_DIGEST_BYTES = 32


def validate_password(password: str) -> str:
    """校验密码强度。返回空串表示通过，否则返回面向用户的提示（docs/05 §G-06 就地提示）。"""
    if not password:
        return "密码不能为空"
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"密码至少 {MIN_PASSWORD_LENGTH} 位"
    return ""


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    """把明文密码转成可入库的哈希串。"""
    salt = secrets.token_hex(SALT_BYTES)
    digest = _derive(password, salt, iterations)
    return f"{ALGORITHM}${iterations}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    """校验明文密码是否匹配已存的哈希串。

    任何格式异常（老数据、被截断、非本算法）都返回 False，**不抛异常**——
    调用方只需区分「对/不对」，不需要处理解析细节。
    """
    if not stored:
        return False

    try:
        algorithm, raw_iterations, salt, digest = stored.split("$")
        iterations = int(raw_iterations)
    except (ValueError, AttributeError):
        return False

    if algorithm != ALGORITHM or iterations <= 0:
        return False

    expected = _derive(password or "", salt, iterations)
    return hmac.compare_digest(expected, digest)


def _derive(password: str, salt: str, iterations: int) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", (password or "").encode("utf-8"), salt.encode("utf-8"), iterations,
        dklen=_DIGEST_BYTES,
    ).hex()
