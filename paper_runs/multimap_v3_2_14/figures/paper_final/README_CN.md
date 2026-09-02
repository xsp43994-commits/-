# v3.2.14 独立小图正式输出

6张正文组合图和8张补充组合图已拆为独立小图；组合画布不再作为论文插图交付。

- 冻结矩阵 SHA-256：`48a31ee9b58d41a617fff61acb6eba6a2d9a930767d7af15856f70a964686224`
- 正式结果 SHA-256：`4b620c21566c2e33c875f6bea2017b741b02a7d30d70aa50add60a6d06214a2c`
- 独立小图：72张，每张含SVG/PDF/PNG/TIFF、图注和source data。
- 路线图均按冻结任务几何裁剪到局部走廊；同一任务的算法共享同一视窗。

## 正文独立小图

- [fig01_study_design_a.png](figures/fig01_study_design_a.png) — 任务场景与安全返航约束 ｜ [图注](captions/fig01_study_design_a.md)
- [fig01_study_design_b.png](figures/fig01_study_design_b.png) — 冻结地图资产概览 ｜ [图注](captions/fig01_study_design_b.md)
- [fig01_study_design_c.png](figures/fig01_study_design_c.png) — 正式评价矩阵 ｜ [图注](captions/fig01_study_design_c.md)
- [fig01_study_design_d.png](figures/fig01_study_design_d.png) — 五条互补证据链 ｜ [图注](captions/fig01_study_design_d.md)
- [fig02_integrated_score_a.png](figures/fig02_integrated_score_a.png) — 七个效应维度 ｜ [图注](captions/fig02_integrated_score_a.md)
- [fig02_integrated_score_b.png](figures/fig02_integrated_score_b.png) — 100分制算术综合得分 ｜ [图注](captions/fig02_integrated_score_b.md)
- [fig02_integrated_score_c.png](figures/fig02_integrated_score_c.png) — PPO+Pointer相对A2C+Pointer的层级bootstrap ｜ [图注](captions/fig02_integrated_score_c.md)
- [fig02_integrated_score_d.png](figures/fig02_integrated_score_d.png) — 归一化下限×权重联合敏感性 ｜ [图注](captions/fig02_integrated_score_d.md)
- [fig02_integrated_score_e.png](figures/fig02_integrated_score_e.png) — PPO相对A2C的维度贡献 ｜ [图注](captions/fig02_integrated_score_e.md)
- [fig03_operational_tradeoffs_a.png](figures/fig03_operational_tradeoffs_a.png) — 合成与真实DSM的安全加权覆盖率 ｜ [图注](captions/fig03_operational_tradeoffs_a.md)
- [fig03_operational_tradeoffs_b.png](figures/fig03_operational_tradeoffs_b.png) — 覆盖、安全和返航的地图级效应 ｜ [图注](captions/fig03_operational_tradeoffs_b.md)
- [fig03_operational_tradeoffs_c.png](figures/fig03_operational_tradeoffs_c.png) — 安全路线的资源与总任务时间 ｜ [图注](captions/fig03_operational_tradeoffs_c.md)
- [fig03_operational_tradeoffs_d.png](figures/fig03_operational_tradeoffs_d.png) — 在线规划时间ECDF ｜ [图注](captions/fig03_operational_tradeoffs_d.md)
- [fig03_operational_tradeoffs_e.png](figures/fig03_operational_tradeoffs_e.png) — 安全加权覆盖率—在线规划时间Pareto视图 ｜ [图注](captions/fig03_operational_tradeoffs_e.md)
- [fig04_training_a.png](figures/fig04_training_a.png) — 五种子收敛过程 ｜ [图注](captions/fig04_training_a.md)
- [fig04_training_b.png](figures/fig04_training_b.png) — 学习曲线AUC ｜ [图注](captions/fig04_training_b.md)
- [fig04_training_c.png](figures/fig04_training_c.png) — 阈值样本效率 ｜ [图注](captions/fig04_training_c.md)
- [fig04_training_d.png](figures/fig04_training_d.png) — 尾段与跨种子稳定性 ｜ [图注](captions/fig04_training_d.md)
- [fig04_training_e.png](figures/fig04_training_e.png) — PPO更新诊断 ｜ [图注](captions/fig04_training_e.md)
- [fig05_ablation_a.png](figures/fig05_ablation_a.png) — 四个消融的地图级总体效应 ｜ [图注](captions/fig05_ablation_a.md)
- [fig05_ablation_b.png](figures/fig05_ablation_b.png) — 显式优先级偏置 ｜ [图注](captions/fig05_ablation_b.md)
- [fig05_ablation_c.png](figures/fig05_ablation_c.png) — 资源塑形与瓶颈类型 ｜ [图注](captions/fig05_ablation_c.md)
- [fig05_ablation_d.png](figures/fig05_ablation_d.png) — 域随机化与扰动退化 ｜ [图注](captions/fig05_ablation_d.md)
- [fig05_ablation_e.png](figures/fig05_ablation_e.png) — 返航储备的仿真安全效应 ｜ [图注](captions/fig05_ablation_e.md)
- [fig06_generalization_robustness_routes_a.png](figures/fig06_generalization_robustness_routes_a.png) — 24张未见合成地图的程序化泛化 ｜ [图注](captions/fig06_generalization_robustness_routes_a.md)
- [fig06_generalization_robustness_routes_b.png](figures/fig06_generalization_robustness_routes_b.png) — 8张真实DSM的零样本仿真迁移 ｜ [图注](captions/fig06_generalization_robustness_routes_b.md)
- [fig06_generalization_robustness_routes_c.png](figures/fig06_generalization_robustness_routes_c.png) — 风、功率、DEM与定位误差的退化热力图 ｜ [图注](captions/fig06_generalization_robustness_routes_c.md)
- [fig06_generalization_robustness_routes_d.png](figures/fig06_generalization_robustness_routes_d.png) — 跨扰动保持率与地图一致性 ｜ [图注](captions/fig06_generalization_robustness_routes_d.md)
- [fig06_generalization_robustness_routes_e1.png](figures/fig06_generalization_robustness_routes_e1.png) — 未见合成任务：PPO+Pointer ｜ [图注](captions/fig06_generalization_robustness_routes_e1.md)
- [fig06_generalization_robustness_routes_e2.png](figures/fig06_generalization_robustness_routes_e2.png) — 未见合成任务：A2C+Pointer ｜ [图注](captions/fig06_generalization_robustness_routes_e2.md)
- [fig06_generalization_robustness_routes_e3.png](figures/fig06_generalization_robustness_routes_e3.png) — 未见合成任务：传统PPO ｜ [图注](captions/fig06_generalization_robustness_routes_e3.md)
- [fig06_generalization_robustness_routes_e4.png](figures/fig06_generalization_robustness_routes_e4.png) — 未见合成任务：MILP ｜ [图注](captions/fig06_generalization_robustness_routes_e4.md)
- [fig06_generalization_robustness_routes_f1.png](figures/fig06_generalization_robustness_routes_f1.png) — 真实DSM任务：PPO+Pointer ｜ [图注](captions/fig06_generalization_robustness_routes_f1.md)
- [fig06_generalization_robustness_routes_f2.png](figures/fig06_generalization_robustness_routes_f2.png) — 真实DSM任务：A2C+Pointer ｜ [图注](captions/fig06_generalization_robustness_routes_f2.md)
- [fig06_generalization_robustness_routes_f3.png](figures/fig06_generalization_robustness_routes_f3.png) — 真实DSM任务：传统PPO ｜ [图注](captions/fig06_generalization_robustness_routes_f3.md)
- [fig06_generalization_robustness_routes_f4.png](figures/fig06_generalization_robustness_routes_f4.png) — 真实DSM任务：MILP ｜ [图注](captions/fig06_generalization_robustness_routes_f4.md)

## 补充独立小图

- [figS01_audit_a.png](figures/figS01_audit_a.png) — 评价家族完整性 ｜ [图注](captions/figS01_audit_a.md)
- [figS01_audit_b.png](figures/figS01_audit_b.png) — 算法×评价家族行数 ｜ [图注](captions/figS01_audit_b.md)
- [figS01_audit_c.png](figures/figS01_audit_c.png) — 嵌套结构与独立单位 ｜ [图注](captions/figS01_audit_c.md)
- [figS01_audit_d.png](figures/figS01_audit_d.png) — 冻结哈希与审计状态 ｜ [图注](captions/figS01_audit_d.md)
- [figS02_scenarios_a.png](figures/figS02_scenarios_a.png) — 节点规模（训练范围内） ｜ [图注](captions/figS02_scenarios_a.md)
- [figS02_scenarios_b.png](figures/figS02_scenarios_b.png) — 认证难度 ｜ [图注](captions/figS02_scenarios_b.md)
- [figS02_scenarios_c.png](figures/figS02_scenarios_c.png) — 约束类型 ｜ [图注](captions/figS02_scenarios_c.md)
- [figS02_scenarios_d.png](figures/figS02_scenarios_d.png) — 优先级布局 ｜ [图注](captions/figS02_scenarios_d.md)
- [figS03_baselines_a.png](figures/figS03_baselines_a.png) — 传统基线任务效果 ｜ [图注](captions/figS03_baselines_a.md)
- [figS03_baselines_b.png](figures/figS03_baselines_b.png) — 参考解差距与区间 ｜ [图注](captions/figS03_baselines_b.md)
- [figS03_baselines_c.png](figures/figS03_baselines_c.png) — 传统规划器计算代价 ｜ [图注](captions/figS03_baselines_c.md)
- [figS03_baselines_d.png](figures/figS03_baselines_d.png) — MILP求解状态与gap ｜ [图注](captions/figS03_baselines_d.md)
- [figS04_training_all_a.png](figures/figS04_training_all_a.png) — 七个学习模型的共同定义训练指标 ｜ [图注](captions/figS04_training_all_a.md)
- [figS04_training_all_b.png](figures/figS04_training_all_b.png) — 七个学习模型的返航与安全过程 ｜ [图注](captions/figS04_training_all_b.md)
- [figS05_score_sensitivity_a.png](figures/figS05_score_sensitivity_a.png) — 算术聚合敏感性 ｜ [图注](captions/figS05_score_sensitivity_a.md)
- [figS05_score_sensitivity_b.png](figures/figS05_score_sensitivity_b.png) — 几何聚合诊断 ｜ [图注](captions/figS05_score_sensitivity_b.md)
- [figS05_score_sensitivity_c.png](figures/figS05_score_sensitivity_c.png) — 全权重网格分差分布 ｜ [图注](captions/figS05_score_sensitivity_c.md)
- [figS05_score_sensitivity_d.png](figures/figS05_score_sensitivity_d.png) — D4/D6/D7配对证据 ｜ [图注](captions/figS05_score_sensitivity_d.md)
- [figS06_ablation_maps_a.png](figures/figS06_ablation_maps_a.png) — 合成地图的消融方向一致性 ｜ [图注](captions/figS06_ablation_maps_a.md)
- [figS06_ablation_maps_b.png](figures/figS06_ablation_maps_b.png) — 真实DSM的消融方向一致性 ｜ [图注](captions/figS06_ablation_maps_b.md)
- [figS06_ablation_maps_c.png](figures/figS06_ablation_maps_c.png) — 终止原因全集 ｜ [图注](captions/figS06_ablation_maps_c.md)
- [figS06_ablation_maps_d.png](figures/figS06_ablation_maps_d.png) — 首次失败约束 ｜ [图注](captions/figS06_ablation_maps_d.md)
- [figS07_robustness_failures_a.png](figures/figS07_robustness_failures_a.png) — 扰动下的安全率 ｜ [图注](captions/figS07_robustness_failures_a.md)
- [figS07_robustness_failures_b.png](figures/figS07_robustness_failures_b.png) — 扰动下的返航率 ｜ [图注](captions/figS07_robustness_failures_b.md)
- [figS07_robustness_failures_c.png](figures/figS07_robustness_failures_c.png) — 扰动下的违规率 ｜ [图注](captions/figS07_robustness_failures_c.md)
- [figS07_robustness_failures_d.png](figures/figS07_robustness_failures_d.png) — 扰动下的Stranded率 ｜ [图注](captions/figS07_robustness_failures_d.md)
- [figS08_route_atlas_a.png](figures/figS08_route_atlas_a.png) — 真实DSM路线图集 1 ｜ [图注](captions/figS08_route_atlas_a.md)
- [figS08_route_atlas_b.png](figures/figS08_route_atlas_b.png) — 真实DSM路线图集 2 ｜ [图注](captions/figS08_route_atlas_b.md)
- [figS08_route_atlas_c.png](figures/figS08_route_atlas_c.png) — 真实DSM路线图集 3 ｜ [图注](captions/figS08_route_atlas_c.md)
- [figS08_route_atlas_d.png](figures/figS08_route_atlas_d.png) — 真实DSM路线图集 4 ｜ [图注](captions/figS08_route_atlas_d.md)
- [figS08_route_atlas_e.png](figures/figS08_route_atlas_e.png) — 真实DSM路线图集 5 ｜ [图注](captions/figS08_route_atlas_e.md)
- [figS08_route_atlas_f.png](figures/figS08_route_atlas_f.png) — 真实DSM路线图集 6 ｜ [图注](captions/figS08_route_atlas_f.md)
- [figS08_route_atlas_g.png](figures/figS08_route_atlas_g.png) — 真实DSM路线图集 7 ｜ [图注](captions/figS08_route_atlas_g.md)
- [figS08_route_atlas_h.png](figures/figS08_route_atlas_h.png) — 真实DSM路线图集 8 ｜ [图注](captions/figS08_route_atlas_h.md)

## 展示图

- [figV01_3d_taihang_route.png](figures/figV01_3d_taihang_route.png) — 太行山DSM、公路巡检点与三维安全返航航迹 ｜ [图注](captions/figV01_3d_taihang_route.md)
- [figV02_outcome_flow_a.png](figures/figV02_outcome_flow_a.png) — 算法→覆盖→终止结果流 ｜ [图注](captions/figV02_outcome_flow_a.md)

## 复核入口

- [自动QA](qa_report_CN.md)
- [图件manifest](figure_manifest.json)
- [分页缩略图](review_contact_sheets/)
