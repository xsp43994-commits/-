"""v3.2正式实验的稳定命令行入口。"""

from __future__ import annotations

from .paper_v3_2_experiments import *  # noqa: F401,F403
from .paper_v3_2_experiments import main


if __name__ == "__main__":
    raise SystemExit(main())
