"""工作区路径的唯一解析入口。"""

from __future__ import annotations

from pathlib import Path


# 迁移后所有模块均通过此常量定位论文数据，避免依赖脚本所在子目录。
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def workspace_path(*parts: str) -> Path:
    """返回位于论文工作区中的绝对路径。"""

    return WORKSPACE_ROOT.joinpath(*parts)
