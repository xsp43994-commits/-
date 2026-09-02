# 论文专用工作区索引

更新时间：2026-08-02

## 正式实验身份

| 项目 | 冻结值 |
|---|---:|
| 正式学习模型 | 7种 × 5种子 = 35 |
| 论文有效训练 | 105,000回合 |
| 未见合成任务 | 216 |
| 真实DSM任务 | 144 |
| 正式评价结果 | 21,648 |
| 正式路线 | 21,648 |
| 正式独立图面板 | 72 |

- 评价矩阵SHA-256：`48a31ee9b58d41a617fff61acb6eba6a2d9a930767d7af15856f70a964686224`
- 正式结果SHA-256：`4b620c21566c2e33c875f6bea2017b741b02a7d30d70aa50add60a6d06214a2c`
- 活跃checkpoint为35个，旧`ppo_mlp`不属于论文模型集合。

## 论文资产位置

- 模型：`paper_runs/multimap_v3_1/formal_training`和`paper_runs/multimap_v3_2/formal_training`。
- 地图与DSM：`map_data/multimap_v3_1`。
- 协议链：`paper_runs/protocols`。
- 合成任务：`paper_runs/multimap_v3_2_14/manifests/synthetic_test/records.jsonl`。
- 真实任务：`paper_runs/multimap_v3_2_14/formal_evaluation/real_tasks_parallel/records.jsonl`。
- 评价矩阵：`paper_runs/multimap_v3_2_14/formal_evaluation/evaluation_matrix.jsonl`。
- 正式结果：`paper_runs/multimap_v3_2_14/formal_evaluation/results/final_results.jsonl`。
- 统计与敏感性分析：`paper_runs/multimap_v3_2_14/analysis`。
- 唯一正式图片包：`paper_runs/multimap_v3_2_14/figures/paper_final`。

## Python代码结构

- `uav_inspection/core`：模型、训练场景、协议和基础评价。
- `uav_inspection/experiments`：正式实验与checkpoint目录接口。
- `uav_inspection/generation`：地图、任务和证书构建。
- `uav_inspection/evaluation`：v3.2.14正式评价工作器。
- `uav_inspection/analysis`：统计、综合评价和敏感性分析。
- `uav_inspection/figures`：正式制图。
- `scripts/protocol_builders`：协议生成器。
- `tools/maintenance`：清理、源码迁移和完整性审计。
- `python_classical_algs`：正式传统规划基线。

整理前64个Python文件和3个PowerShell脚本原样保存在：

`paper_runs/code_snapshots/pre_python_reorganization`

其中`source_snapshot_manifest.json`保存原始哈希，`source_relocation_map.csv`记录旧路径、新路径和状态。冻结协议中的历史源码路径不改写，由该快照提供审计证据。

需要特别说明：现存冻结manifest中的9项历史实现引用，有6项能与整理前源码逐字节匹配；`final_python_ppo_pointer.py`、`v3_2_14_statistics.py`和`v3_2_14_split_publication_figures.py`在本次整理开始前就已更新，较早的冻结版本源码字节未留存在工作区。本次整理没有造成或掩盖该差异，详情记录在`post_cleanup_audit.json`。

完整迁移表见`docs/PYTHON_SOURCE_LAYOUT.md`。

## 统一命令

```powershell
python -X utf8 -B paper_cli.py show-paths
python -X utf8 -B paper_cli.py audit-workspace
python -X utf8 -B paper_cli.py audit-checkpoints
python -X utf8 -B paper_cli.py statistics
python -X utf8 -B paper_cli.py figures
```

低层正式入口示例：

```powershell
python -X utf8 -B -m uav_inspection.experiments.v3_2 audit-checkpoints
python -X utf8 -B -m uav_inspection.analysis.v3_2_14_statistics
python -X utf8 -B -m uav_inspection.figures.v3_2_14_split_publication_figures
```

测试：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES=''
& 'D:\anaconda3\envs\Deeplearning-gpu\python.exe' -X utf8 -m pytest -q tests
```

## 禁止混入论文的内容

- 旧`ppo_mlp`模型及结果；
- 第一轮实验训练、评价和旧publication目录；
- v3.2.1—v3.2.13失败输出；
- pilot、debug、monitoring和失败fallback结果；
- `latest.pt`、`best_candidate.pt`；
- 旧19,600条结果口径、旧测试域和旧综合评分规则。

不得根据正式结果重新挑选任务、地图、扰动、模型或代表路线。源码分层整理只改变活动代码路径，不改变上述冻结实验身份。
