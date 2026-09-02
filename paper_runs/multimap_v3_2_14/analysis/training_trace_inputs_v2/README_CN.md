# 冻结训练曲线输入

本目录仅保存 `manuscript_training_aware_v2.py` 计算 D6（训练稳定性）和
D7（样本效率）时实际使用的10条历史训练曲线：`full` 与
`a2c_pointer` 各5个种子。

这些文件从清理前的 `paper_runs/training` 原样迁移，文件字节及SHA-256
均未改变。保留它们是为了让既有v2–v5分析、综合得分和敏感性结果能够
按冻结协议复算；它们不属于35个正式多地图模型checkpoint，也不得替代
`paper_runs/multimap_v3_1/formal_training` 中的正式训练曲线用于其他分析。

完整逻辑原路径、当前保存路径和哈希见 `source_manifest.json`。
