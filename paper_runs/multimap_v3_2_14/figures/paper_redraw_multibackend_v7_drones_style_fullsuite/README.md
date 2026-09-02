# v7 Drones-style full scientific figure suite

本目录是 v3.2.14 冻结结果的独立制图交付，共 21 图、中文和英文两套。英文为 EAAI 投稿主版本，中文用于确认；二者共享 Source Data、坐标范围和视觉编码。

## 快速入口

- `exports/en/`：英文投稿图件。
- `exports/zh/`：中文确认图件。
- `source_data/`：逐图 Source Data；S09 为新增训练回报证据。
- `editable/origin/`：10 个 OPJU。
- `editable/matlab/`：V02 MATLAB 源文件与 FIG。
- `editable/python/`：统一双语绘图源文件。
- `captions/`：21 份双语图注。
- `qa/automatic_qa_summary.json`：42 项双语自动 QA 总结。
- `qa/style_alignment_review.md`：逐图权威文献风格对齐复审。
- `qa/style_alignment_side_by_side.png`：权威文献与最终全套图并排证据。
- `qa/eaai_single_column_preview_contact_sheet.png`：EAAI 单栏放置预览总览。
- `manifests/`：文件哈希与交付清单。

## 证据边界

- M06 是固定 108 任务外部验证曲线；S06 和 S09 是训练批次证据，不能混称为测试表现。
- S09 每个点是 16 回合训练批次的平均 episode return；无时间平滑、无插值。
- V01/V02 是冻结代表任务的描述性证据，不用于推断总体差异。
- S07 是事后、权重依赖摘要，须与 S08 敏感性图共同解释。
- 没有新增测试集、重跑评价、恢复 `ppo_mlp`、引入 SPL 或人为成功率阈值。
