#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把试训前v3.1.14身份迁移到阈值短缺门禁后的v3.1.15。"""

from __future__ import annotations

import migrate_v3_1_13 as migration


migration.OLD_PROTOCOL_HASH = (
    "e1bc7637a212639f7792594ce7cb664563e3334a8f8fb53cb216c159f89b1f7a"
)
migration.NEW_PROTOCOL_HASH = (
    "92b25776749a9430e71e47e0882970c5ea149ed778906104ad38604b870860ba"
)
migration.EXPECTED_UNIQUE_TRAINING_RECORDS = 229
migration.MIGRATION_LABEL = "v3.1.15"
migration.MIGRATION_REPORT_NAME = "migration_v3_1_15.json"
migration.MIGRATION_DESCRIPTION = (
    "v3.1.15 limits mixed threshold rescue to incumbents no more than "
    "one discrete priority unit below the band; retained records are "
    "byte-identical"
)


if __name__ == "__main__":
    migration.main()
