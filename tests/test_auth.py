"""认证与账号单元测试。

覆盖 docs/03 §6.1 的 TC-U14（密码哈希与校验）、TC-U15（重复用户名注册）、
TC-U19（修改密码后旧密码失效），以及 docs/05 §PG-01 的登录分支。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.auth import service
from src.auth.security import hash_password, validate_password, verify_password
from src.auth.service import (
    INVALID_CREDENTIALS_TEXT,
    can_access_page,
    change_display_name,
    change_password,
    login,
    register,
    require_admin,
    validate_username,
)
from src.errors import AccountDisabled, AuthError, PermissionDenied, UsernameTaken
from src.repository import (
    USER_ROLE_ADMIN,
    User,
    add_message,
    create_conversation,
    create_document,
    create_user,
    get_user,
    get_user_by_name,
    list_conversations,
    list_documents,
)
from src.store.chroma import build_metadata
from src.store.db import get_conn

PASSWORD = "secret123"


def new_user(db: Path, username: str = "alice", password: str = PASSWORD) -> User:
    return register(username, password, agreed=True, db_path=db)


def new_admin(db: Path, username: str = "root", password: str = PASSWORD) -> User:
    user_id = create_user(
        username=username,
        password_hash=hash_password(password),
        role=USER_ROLE_ADMIN,
        db_path=db,
    )
    return get_user(user_id, db_path=db)


# ==================== TC-U14：密码哈希与校验 ====================


def test_hash_does_not_contain_plaintext() -> None:
    stored = hash_password(PASSWORD)

    assert PASSWORD not in stored
    assert stored.startswith("pbkdf2_sha256$")
    assert len(stored.split("$")) == 4


def test_same_password_hashes_differently(monkeypatch) -> None:
    """每次都要用新的随机盐，否则相同密码会产生相同哈希（可被彩虹表看出规律）。"""
    assert hash_password(PASSWORD) != hash_password(PASSWORD)


def test_verify_accepts_correct_and_rejects_wrong() -> None:
    stored = hash_password(PASSWORD)

    assert verify_password(PASSWORD, stored) is True
    assert verify_password("wrong-password", stored) is False


def test_verify_rejects_broken_or_empty_hashes() -> None:
    """老数据或被截断的哈希不能把异常抛给调用方，一律判为不通过。"""
    for broken in ("", "not-a-hash", "pbkdf2_sha256$abc$x$y", "md5$1$salt$hash"):
        assert verify_password(PASSWORD, broken) is False


def test_tc_u14_no_plaintext_in_database(db: Path) -> None:
    new_user(db)

    with get_conn(db) as conn:
        stored = conn.execute(
            "SELECT password_hash FROM users WHERE username = 'alice'"
        ).fetchone()["password_hash"]

    assert stored != PASSWORD
    assert PASSWORD not in stored


def test_validate_password_requires_six_characters() -> None:
    assert validate_password("") != ""
    assert validate_password("12345") != ""
    assert validate_password("123456") == ""


# ==================== TC-U15：注册校验 ====================


def test_register_creates_user_with_hashed_password(db: Path) -> None:
    user = new_user(db)

    assert user.username == "alice"
    assert user.display_name == "alice"
    assert user.is_active is True
    assert user.is_admin is False


def test_tc_u15_duplicate_username_is_rejected(db: Path) -> None:
    new_user(db)

    with pytest.raises(UsernameTaken):
        new_user(db)


def test_register_requires_agreement(db: Path) -> None:
    """FR-28：注册必须显式勾选协议，默认放行等于绕过合规校验。"""
    with pytest.raises(AuthError, match="用户协议"):
        register("bob", PASSWORD, agreed=False, db_path=db)


def test_register_rejects_mismatched_confirmation(db: Path) -> None:
    with pytest.raises(AuthError, match="不一致"):
        register("bob", PASSWORD, confirm="secret999", agreed=True, db_path=db)


def test_register_rejects_short_password(db: Path) -> None:
    with pytest.raises(AuthError, match="至少"):
        register("bob", "12345", agreed=True, db_path=db)


def test_validate_username_rules() -> None:
    assert validate_username("") != ""
    assert validate_username("a") != ""
    assert validate_username("a" * 33) != ""
    assert validate_username("with space") != ""
    assert validate_username("张三") == ""


# ==================== 登录分支（PG-01 / FR-02）====================


def test_login_success(db: Path) -> None:
    new_user(db)

    user = login("alice", PASSWORD, db_path=db)

    assert user.username == "alice"


def test_login_wrong_password_uses_generic_message(db: Path) -> None:
    new_user(db)

    with pytest.raises(AuthError) as excinfo:
        login("alice", "wrong-password", db_path=db)

    assert str(excinfo.value) == INVALID_CREDENTIALS_TEXT


def test_login_unknown_user_uses_same_message(db: Path) -> None:
    """不区分「不存在」与「密码错」，避免被用来枚举账号。"""
    with pytest.raises(AuthError) as excinfo:
        login("nobody", PASSWORD, db_path=db)

    assert str(excinfo.value) == INVALID_CREDENTIALS_TEXT


def test_disabled_account_cannot_login_even_with_correct_password(db: Path) -> None:
    user = new_user(db)
    with get_conn(db) as conn:
        conn.execute("UPDATE users SET status = 'disabled' WHERE id = ?", (user.id,))

    with pytest.raises(AccountDisabled, match="已被禁用"):
        login("alice", PASSWORD, db_path=db)


# ==================== 角色守卫（FR-03 / G-02）====================


def test_require_admin_accepts_admin(db: Path) -> None:
    require_admin(new_admin(db))


def test_require_admin_rejects_normal_user(db: Path) -> None:
    with pytest.raises(PermissionDenied, match="无权访问"):
        require_admin(new_user(db))


def test_require_admin_rejects_anonymous() -> None:
    with pytest.raises(PermissionDenied):
        require_admin(None)


# ==================== 页面访问判定（G-01 / G-02）====================


def test_can_access_page_denies_anonymous() -> None:
    """G-01 访客拦截：未登录任何页面都不可访问。"""
    assert can_access_page(None, "qa") is False
    assert can_access_page(None, "about") is False


def test_can_access_page_allows_normal_pages_for_user(db: Path) -> None:
    user = new_user(db)

    assert can_access_page(user, "qa") is True
    assert can_access_page(user, "documents") is True
    assert can_access_page(user, "settings") is True
    assert can_access_page(user, "about") is True


def test_can_access_page_blocks_admin_page_for_normal_user(db: Path) -> None:
    """G-02 角色守卫：user 访问管理员页被拦截。"""
    assert can_access_page(new_user(db), "admin") is False


def test_can_access_page_allows_admin_page_for_admin(db: Path) -> None:
    assert can_access_page(new_admin(db), "admin") is True


# ==================== TC-U19：修改密码 ====================


def test_tc_u19_old_password_invalid_after_change(db: Path) -> None:
    user = new_user(db)

    change_password(user.id, PASSWORD, "newsecret456", db_path=db)

    with pytest.raises(AuthError, match=INVALID_CREDENTIALS_TEXT):
        login("alice", PASSWORD, db_path=db)
    assert login("alice", "newsecret456", db_path=db).id == user.id


def test_change_password_requires_correct_old_password(db: Path) -> None:
    user = new_user(db)

    with pytest.raises(AuthError, match="原密码"):
        change_password(user.id, "wrong-password", "newsecret456", db_path=db)


def test_change_password_rejects_short_new_password(db: Path) -> None:
    user = new_user(db)

    with pytest.raises(AuthError, match="至少"):
        change_password(user.id, PASSWORD, "123", db_path=db)


def test_change_password_rejects_same_password(db: Path) -> None:
    user = new_user(db)

    with pytest.raises(AuthError, match="不能与原密码相同"):
        change_password(user.id, PASSWORD, PASSWORD, db_path=db)


# ==================== 显示名 ====================


def test_change_display_name(db: Path) -> None:
    user = new_user(db)

    updated = change_display_name(user.id, "  小张  ", db_path=db)

    assert updated.display_name == "小张"
    assert get_user_by_name("alice", db_path=db).display_name == "小张"


def test_change_display_name_rejects_blank(db: Path) -> None:
    user = new_user(db)

    with pytest.raises(AuthError, match="不能为空"):
        change_display_name(user.id, "   ", db_path=db)


# ==================== TC-U25：注销（跨库级联清理）====================


def test_delete_account_purges_vectors_files_and_records(
    db: Path, store, settings
) -> None:
    """FR-25：注销后不应有任何残留——文档、向量、文件、会话全清，公共数据不动。"""
    user = new_user(db)
    doc_id = create_document(
        filename="我的资料.txt", filetype="txt", category="uncategorized",
        user_id=user.id, db_path=db,
    )
    store.add_chunks([("私人内容", build_metadata(
        doc_id=doc_id, user_id=user.id, is_public=False, category="uncategorized",
        filename="我的资料.txt", chunk_index=0,
    ))])
    store.add_chunks([("公共内容", build_metadata(
        doc_id=999, user_id=None, is_public=True, category="admin",
        filename="公共资料.txt", chunk_index=0,
    ))])
    upload_dir = settings.uploads_path / str(user.id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / f"{doc_id}_我的资料.txt").write_text("内容", encoding="utf-8")
    conversation_id = create_conversation(user.id, db_path=db)
    add_message(conversation_id, "user", "问题", db_path=db)

    service.delete_account(user.id, store=store, settings=settings)

    assert get_user(user.id, db_path=db) is None
    assert list_documents(user_id=user.id, db_path=db) == ([], 0)
    assert list_conversations(user.id, db_path=db) == ([], 0)
    assert not upload_dir.exists()
    assert store.count() == 1  # 只剩公共向量


def test_delete_account_aborts_before_touching_database(
    db: Path, store, settings, monkeypatch
) -> None:
    """向量清理失败必须中止，且**不进入** SQLite 事务，避免半删状态。"""
    user = new_user(db)
    create_document(
        filename="我的资料.txt", filetype="txt", category="", user_id=user.id, db_path=db
    )

    def boom(*_args, **_kwargs):
        raise RuntimeError("chroma 挂了")

    monkeypatch.setattr(store, "delete_by_user_id", boom)

    with pytest.raises(RuntimeError):
        service.delete_account(user.id, store=store, settings=settings)

    # 账号与文档原封不动，可重试
    assert get_user(user.id, db_path=db) is not None
    assert list_documents(user_id=user.id, db_path=db)[1] == 1


def test_delete_account_rejects_missing_user(db: Path, store, settings) -> None:
    with pytest.raises(AuthError, match="账号不存在"):
        service.delete_account(999, store=store, settings=settings)
