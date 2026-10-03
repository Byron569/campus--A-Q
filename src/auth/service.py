"""认证服务：注册 / 登录 / 角色校验 / 改密 / 改显示名。

设计依据：
- docs/02-架构设计.md §4.8（密码、角色、登录态、权限过滤）
- docs/05-产品原型与交互说明.md §PG-01（登录注册）、§PG-05（设置）、§G-02（角色守卫）、§G-06（就地提示）
- docs/06-接口文档.md §1.6（`register` / `login` / `require_admin`，禁用账号抛 `AccountDisabled`）
- docs/03-开发任务清单.md §6.1 TC-U14 / TC-U15 / TC-U19

两条安全约定：
- **登录失败不区分「用户不存在」与「密码错误」**，统一提示「用户名或密码错误」，
  避免被用来枚举账号（PG-01「登录失败 ×3 次」条）。
- 先查账号状态再验密码：被禁用的账号即便密码正确也不放行（FR-02）。
"""

from __future__ import annotations

import logging
import sqlite3

from src.auth.security import hash_password, validate_password, verify_password
from src.errors import AccountDisabled, AuthError, PermissionDenied, UsernameTaken
from src.repository import (
    User,
    create_user,
    get_user,
    get_user_by_name,
    update_display_name,
    update_password,
)

logger = logging.getLogger(__name__)

MIN_USERNAME_LENGTH = 2
MAX_USERNAME_LENGTH = 32

INVALID_CREDENTIALS_TEXT = "用户名或密码错误"
DISABLED_TEXT = "账号已被禁用，请联系管理员"
AGREEMENT_REQUIRED_TEXT = "请先勾选《用户协议与隐私说明》"

# 仅 admin 可访问的页面 key（docs/05 §G-02）
ADMIN_PAGES = frozenset({"admin"})


def can_access_page(user: User | None, page_key: str) -> bool:
    """页面访问判定（纯函数，便于单测）。

    - 未登录：一律不可访问（docs/05 §G-01 访客拦截）
    - 管理员页：仅 admin（§G-02）
    - 其余页面：登录用户均可
    """
    if user is None:
        return False
    if page_key in ADMIN_PAGES:
        return user.is_admin
    return True


def validate_username(username: str) -> str:
    """校验用户名。返回空串表示通过。"""
    name = (username or "").strip()
    if not name:
        return "用户名不能为空"
    if len(name) < MIN_USERNAME_LENGTH:
        return f"用户名至少 {MIN_USERNAME_LENGTH} 个字符"
    if len(name) > MAX_USERNAME_LENGTH:
        return f"用户名不能超过 {MAX_USERNAME_LENGTH} 个字符"
    if any(char.isspace() for char in name):
        return "用户名不能包含空格"
    return ""


def register(
    username: str,
    password: str,
    *,
    confirm: str | None = None,
    agreed: bool = False,
    display_name: str | None = None,
    db_path=None,
) -> User:
    """注册并返回新账号（FR-01）。

    `agreed` 默认 False：**注册必须显式勾选**《用户协议与隐私说明》，
    默认放行等于让调用方忘了传就绕过合规校验。
    """
    name = (username or "").strip()

    reason = validate_username(name)
    if reason:
        raise AuthError(reason)

    reason = validate_password(password)
    if reason:
        raise AuthError(reason)

    if confirm is not None and password != confirm:
        raise AuthError("两次输入的密码不一致")

    if not agreed:
        raise AuthError(AGREEMENT_REQUIRED_TEXT)

    if get_user_by_name(name, db_path=db_path) is not None:
        raise UsernameTaken(f"用户名「{name}」已被占用")

    try:
        user_id = create_user(
            username=name,
            password_hash=hash_password(password),
            display_name=display_name,
            db_path=db_path,
        )
    except sqlite3.IntegrityError as exc:
        # 并发注册同名时的兜底：先查后写之间存在竞态窗口
        raise UsernameTaken(f"用户名「{name}」已被占用") from exc

    logger.info("新用户注册：%s（id=%s）", name, user_id)
    return get_user(user_id, db_path=db_path)


def login(username: str, password: str, *, db_path=None) -> User:
    """校验账号密码并返回用户（FR-02）。

    Raises:
        AuthError: 用户不存在或密码错误（文案统一，不透露账号是否存在）。
        AccountDisabled: 账号被禁用。
    """
    user = get_user_by_name((username or "").strip(), db_path=db_path)
    if user is None:
        raise AuthError(INVALID_CREDENTIALS_TEXT)

    # 状态先于密码：被禁用的账号即便密码正确也不放行
    if not user.is_active:
        raise AccountDisabled(DISABLED_TEXT)

    if not verify_password(password or "", user.password_hash):
        raise AuthError(INVALID_CREDENTIALS_TEXT)

    return user


def require_admin(user: User | None) -> None:
    """角色守卫（FR-03 / docs/05 §G-02）。"""
    if user is None or not user.is_admin:
        raise PermissionDenied("无权访问该页面")


def change_password(
    user_id: int,
    old_password: str,
    new_password: str,
    *,
    confirm: str | None = None,
    db_path=None,
) -> None:
    """修改密码（FR-26）：必须先验旧密码。"""
    user = get_user(user_id, db_path=db_path)
    if user is None:
        raise AuthError("账号不存在")

    if not verify_password(old_password or "", user.password_hash):
        raise AuthError("原密码不正确")

    reason = validate_password(new_password)
    if reason:
        raise AuthError(reason)

    if confirm is not None and new_password != confirm:
        raise AuthError("两次输入的密码不一致")

    if verify_password(new_password, user.password_hash):
        raise AuthError("新密码不能与原密码相同")

    update_password(user_id, hash_password(new_password), db_path=db_path)
    logger.info("用户 %s 已修改密码", user_id)


def change_display_name(user_id: int, display_name: str, *, db_path=None) -> User:
    """修改显示名（FR-26）。"""
    name = (display_name or "").strip()
    if not name:
        raise AuthError("显示名不能为空")

    if not update_display_name(user_id, name, db_path=db_path):
        raise AuthError("账号不存在")

    return get_user(user_id, db_path=db_path)
