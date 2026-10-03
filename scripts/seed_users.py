"""管理员初始化脚本（M3-09 / FR-01）。

设计依据：docs/03-开发任务清单.md M3-09「按 .env 创建管理员账号」。

为什么要单独一个脚本：注册页只创建普通学生账号（`role=user`），
而管理员页需要 `role=admin`。系统启动时不会自动造账号——
自动建号会在生产环境留下一个已知密码的后门。

账号密码取 `.env` 的 `ADMIN_USERNAME` / `ADMIN_PASSWORD`（默认 admin / change-me-please）。

用法：
    python scripts/seed_users.py                     # 按 .env 创建（已存在则跳过）
    python scripts/seed_users.py --username root --password 自定义密码
    python scripts/seed_users.py --reset-password    # 已存在时重置密码
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 直接以 `python scripts/xxx.py` 运行时，sys.path[0] 是 scripts/ 而非项目根，
# 因此需要手动把项目根加进来才能 import config / src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings  # noqa: E402
from src.auth.security import hash_password  # noqa: E402
from src.errors import CampusQAError  # noqa: E402
from src.repository import (  # noqa: E402
    USER_ROLE_ADMIN,
    create_user,
    get_user_by_name,
    update_password,
)
from src.store.db import init_db  # noqa: E402


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="创建管理员账号")
    parser.add_argument("--username", default=settings.admin_username, help="管理员用户名")
    parser.add_argument("--password", default=settings.admin_password, help="管理员密码（≥6 位）")
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="账号已存在时重置其密码（默认跳过，不覆盖现有账号）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    if len(args.password) < 6:
        print("密码至少 6 位，未做任何修改。")
        return 1

    settings.ensure_dirs()
    init_db()
    db_path = settings.database_path

    existing = get_user_by_name(args.username, db_path=db_path)
    if existing is not None:
        if not args.reset_password:
            print(f"管理员「{args.username}」已存在（id={existing.id}，角色={existing.role}），未做修改。")
            print("如需重置密码，请加 --reset-password。")
            return 0
        try:
            update_password(existing.id, hash_password(args.password), db_path=db_path)
        except CampusQAError as exc:  # pragma: no cover - 实际不会走到
            print(f"重置密码失败：{exc}")
            return 1
        print(f"已重置管理员「{args.username}」的密码（id={existing.id}）。")
        return 0

    user_id = create_user(
        username=args.username,
        password_hash=hash_password(args.password),
        display_name="管理员",
        role=USER_ROLE_ADMIN,
        db_path=db_path,
    )
    print(f"已创建管理员「{args.username}」（id={user_id}）。")
    if args.password == "change-me-please":
        print("提醒：当前仍是 .env.example 的默认密码，上线前请务必修改。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
