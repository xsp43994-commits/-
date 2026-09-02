#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把试训前v3.1.13身份迁移到标准入带路线快速路径后的v3.1.14。"""

from __future__ import annotations

import migrate_v3_1_13 as migration


migration.OLD_PROTOCOL_HASH = (
    "bf164f92348cf73f17a5fbef12391f64f3b183a66addcd03c271235b54598c81"
)
migration.NEW_PROTOCOL_HASH = (
    "e1bc7637a212639f7792594ce7cb664563e3334a8f8fb53cb216c159f89b1f7a"
)
migration.EXPECTED_UNIQUE_TRAINING_RECORDS = 228
migration.MIGRATION_LABEL = "v3.1.14"
migration.MIGRATION_REPORT_NAME = "migration_v3_1_14.json"
migration.MIGRATION_DESCRIPTION = (
    "v3.1.14 checks an already in-band standard MILP incumbent before "
    "running lower-threshold solvers; retained records are byte-identical"
)


if __name__ == "__main__":
    migration.main()
