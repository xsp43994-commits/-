# Python源码迁移说明

## 活动模块分层

| 旧根目录模块 | 新模块前缀 |
|---|---|
| `final_python_ppo_pointer`、`ppo_training_scenario`、`paper_protocol`、`paper_evaluation` | `uav_inspection.core` |
| `paper_experiments`、`paper_multimap_experiments`、`paper_difficulty_experiments`、`paper_v3_2_experiments` | `uav_inspection.experiments` |
| 真实/合成任务、证书搜索、证书组合和任务迁移脚本 | `uav_inspection.generation` |
| `v3_2_14_*worker`与正式评价共享实现 | `uav_inspection.evaluation` |
| `v3_2_14_statistics`、`manuscript_*`和相应审计器 | `uav_inspection.analysis` |
| 两个v3.2.14正式制图模块 | `uav_inspection.figures` |
| `make_v3_2*`、`make_manuscript*` | `scripts.protocol_builders` |
| 工作区清理和审计 | `tools.maintenance` |

逐文件映射以`paper_runs/code_snapshots/pre_python_reorganization/source_relocation_map.csv`为准。

## 兼容边界

- 不在根目录创建旧模块名包装器，避免再次堆积脚本。
- 活动代码统一从`uav_inspection.paths.WORKSPACE_ROOT`解析工作区路径。
- 旧冻结协议、manifest和figure manifest保持字节不变。
- 需要重放历史脚本时，使用源码快照中的原始文件和对应原始环境；当前活动入口使用新的包路径。
- checkpoint保存的是tensor/state-dict，不以旧源码模块名作为加载前提；审计仍需逐个验证35个checkpoint。

## 历史源码版本说明

本次迁移前的67个脚本均已原样快照并通过哈希复核。冻结manifest更早记录的9项实现哈希中，6项能由该快照直接复核；另3项的旧版本内容在本次迁移前已经不在工作区，只能保留其冻结哈希和当前版本差异记录。此项属于既有版本留存缺口，不影响正式矩阵、结果、路线、checkpoint或图片文件哈希。
