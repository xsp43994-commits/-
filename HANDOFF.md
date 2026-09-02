# v3.2.14 科研制图与实验结果交接文档

> 更新时间：2026-08-31（Asia/Shanghai）  
> 面向对象：一个完全没有此前对话上下文的新 Codex 会话  
> 当前工作重点：围绕第三轮完整科研图 `paper_redraw_multibackend_v3` 做逐图、针对性优化，并回答用户对图形数据、统计含义、绘图方式和论文合规性的具体问题。  
> 第一原则：实验结果已经冻结。后续是制图和解释工作，不是重新训练、重新测试或通过改权重改变结论。

---

## 1. 项目在研究什么

本项目研究山区公路无人机固定巡检点的路径规划：

- 场景中有两条相交山区公路，机场位于道路交界附近；
- 巡检对象是道路上的固定巡检点，无人机只需覆盖巡检点，不要求沿道路连续巡检；
- 巡检点具有高、中、低优先级；
- 路线受到电量、距离、时间、风场、地形、动力学和强制安全返航等约束；
- 目标是在满足安全约束的前提下提高优先级加权覆盖率，同时兼顾安全率、返航率、鲁棒性、能耗、航程、在线规划时间、总任务时间、训练稳定性和样本效率；
- 真实 DSM 实验是 Copernicus GLO-30 地形上的零样本仿真迁移，不是实飞验证或安全认证。

目标期刊背景为 *Engineering Applications of Artificial Intelligence*（EAAI），但下一阶段首先是图件优化，不要自行改投、改文章结构或启动投稿工作。

---

## 2. 已冻结的实验事实

### 2.1 正式学习模型

论文有效模型共 7 种，每种 5 个训练种子（42–46），每种子 3,000 回合：

1. `full`：PPO＋Pointer 完整模型；
2. `traditional_ppo`：传统扁平 MLP PPO 学习基线；
3. `a2c_pointer`：A2C＋Pointer 学习基线；
4. `no_priority_bias`；
5. `no_domain_randomization`；
6. `no_resource_shaping`；
7. `no_return_reserve`。

论文有效训练账目：35 个模型、105,000 回合。

历史 `ppo_mlp` 已被 `traditional_ppo` 替代，只作归档，禁止出现在论文图、统计或模型列表中。项目累计执行过 120,000 个正式训练回合，是因为旧 `ppo_mlp` 的 15,000 回合仍保留历史证据。

### 2.2 正式评价

- 正式结果：21,648 条；
- 正式路线：21,648 条；
- 冻结矩阵 SHA-256：`48a31ee9b58d41a617fff61acb6eba6a2d9a930767d7af15856f70a964686224`；
- 正式结果 SHA-256：`4b620c21566c2e33c875f6bea2017b741b02a7d30d70aa50add60a6d06214a2c`；
- 最终审计：`passed=true`、`ppo_mlp_absent=true`。

权威入口：

- `paper_runs/multimap_v3_2_14/formal_evaluation/results/final_results.jsonl`
- `paper_runs/multimap_v3_2_14/formal_evaluation/results/final_audit_status.json`

未经用户明确授权，绝对不要：

- 重训模型；
- 重建正式测试集；
- 重跑 21,648 条正式评价；
- 修改任务、扰动、checkpoint、算法预算或统计协议；
- 根据已有结果筛选地图、任务、种子或代表路线。

### 2.3 统计边界

- 地图是主要独立统计单位；任务、训练种子、规划种子均嵌套在地图内；
- 不能把 144 个真实任务当作 144 个独立地理样本，真实地形独立单位只有 8 张 DSM；
- 16/20/24 三种规模都参加训练，只能称为“训练范围内多规模表现”，不能称为未训练规模泛化；
- 24 张未见合成地图用于程序化跨地图泛化；
- 8 张真实 DSM 用于跨地区、跨地形分布的零样本仿真迁移；
- `safe_weighted_coverage` 是安全门控覆盖指标：危险返航或硬约束失败时记 0；
- 综合得分是事后、权重依赖的辅助摘要，不能替代原始指标、区间和敏感性结果；
- PPO＋Pointer 并不是所有原始覆盖率场景中的绝对冠军，不能预设或强行制造其领先结论。

---

## 3. 当前完整制图基线：第三轮 v3

完整 20 张第三轮科研图位于：

`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v3/`

组成：

- 正文图 M01–M10：10 张；
- 补充图 S01–S08：8 张；
- 展示图 V01–V02：2 张；
- Origin 10 张、Python 9 张、MATLAB 1 张；
- 20 篇论文、60 幅原始论文图的图型审计；
- 10 个可编辑 Origin OPJU；
- V02 有 MATLAB `.m` 和 `.fig`；
- 每图有 Source Data、中文图注和 manifest；
- 总 QA 通过。

最适合新会话快速总览的文件：

- 全部缩略图：`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v3/thumbnail_index/all_20_figures.png`
- 图形注册表：`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v3/manifests/figure_registry_manual_v3.json`
- 完整状态：`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v3/manifests/full_render_status.json`
- 最终 QA：`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v3/qa/final_qa_report.json`
- Origin UI 审计：`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v3/qa/origin_opju_ui_audit.json`
- 文献图型审计：`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v3/literature_audit/literature_style_audit.csv`

### 3.1 v3 图表清单

| ID | 名称 | v3 后端 | 当前用途 |
|---|---|---|---|
| M01 | 优先级加权覆盖率分布 | Python | 正文完整基线 |
| M02 | 安全率与返航率效应 | Origin | 正文完整基线 |
| M03 | 高中低优先级巡检效果 | Origin | 正文完整基线 |
| M04 | 能耗、航程与总任务时间 | Origin | 正文完整基线 |
| M05 | 在线规划时间 ECDF | Origin | 正文完整基线 |
| M06 | 五种子收敛曲线 | Python | **v3 数据语义已过时，禁止继续使用；改用 v6** |
| M07 | 训练稳定性与样本效率 | Origin | **v3 数值已过时；改用 v4 纠正版** |
| M08 | 未见地图与真实 DSM 迁移 | Origin | 正文完整基线 |
| M09 | 已知偏移与隐藏误差鲁棒性 | Python | 正文完整基线 |
| M10 | 四项消融总体效应 | Origin | 正文完整基线 |
| S01 | 全算法 Performance Profile | Python | 补充完整基线 |
| S02 | 覆盖效果—在线时间 Pareto | Origin | 补充完整基线 |
| S03 | Oracle regret—计算代价 | Origin | 补充完整基线 |
| S04 | 场景分层结果热力图 | Python | 补充完整基线 |
| S05 | 鲁棒性与失败模式 | Python | 补充完整基线 |
| S06 | 七个学习模型训练过程 | Python | **v3 数据语义已过时；七模型图用 v4，核心三模型动态图用 v6** |
| S07 | 七维指标与 100 分综合摘要 | Origin | **v3 D6/D7 已过时；改用 v4 纠正版** |
| S08 | 权重与归一化联合敏感性 | Python | **v3 D6/D7 已过时；改用 v4 纠正版** |
| V01 | 固定合成任务路线 | Python | 展示基线 |
| V02 | 固定真实 DSM 地形路线 | MATLAB | 展示基线；已修复顶部坐标裁切和标注重叠 |

### 3.2 v3 Origin 图的真实状态

Origin 图：M02、M03、M04、M05、M07、M08、M10、S02、S03、S07。

- Origin 版本：OriginPro 2021；
- 10 个 OPJU 都已在 Origin 界面中逐个打开核验；
- 工作簿、图页、图层和绘图对象均可编辑；
- UI 审计时未保存或改动项目；
- Origin 2021 原生 SVG 对这 10 张图导出不可靠，因此 v3 没有用损坏 SVG 凑数；
- 对 Origin 图，可编辑主版本是 OPJU，稳定矢量版本是原生 PDF；PNG/TIFF 用于预览和投稿位图。

不要把 Origin 原生 SVG 缺失误判为图件没完成，也不要通过伪造或转换坏 SVG 来满足格式数量。

### 3.3 v3 Python 图的可复现性限制

v3 的 Python 图保留了 Source Data、manifest、PDF/SVG/PNG/TIFF，但原始 Python 渲染脚本没有全部保存在 v3 目录中。后续优化这些图时：

- 必须以 v3 Source Data 和 manifest 为输入重建新脚本；
- 不得从 PNG 反推、描点或手工改数据；
- 新脚本和配置必须保存在新的目标优化目录；
- M06 不得从 v3 Source Data 重建，必须使用 v6 正式训练来源。

---

## 4. 训练曲线纠错的版本关系（极其重要）

### 4.1 为什么 v3 的训练相关图不能直接用

早期训练曲线曾误用了历史训练轨迹/错误验证语义，导致：

- 纵轴覆盖率被误画到接近 1；
- 曲线上方出现不自然的平台；
- 训练批次覆盖率、外部验证安全加权覆盖率被混淆；
- 旧 D6/D7 和部分综合得分基于错误轨迹；
- 曾出现 0.90 阈值效率等不适用于正式验证数据的定义。

这不是简单的视觉风格问题，而是数据身份问题。

### 4.2 v4：全链路训练数据纠正

位置：

`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v4_training_corrected/`

v4 重绘并纠正：

- M06；
- M07；
- S06；
- S07；
- S08。

对应分析真值：

`paper_runs/multimap_v3_2_14/analysis/training_curve_correction_v6/`

正确训练来源：

- PPO＋Pointer、A2C＋Pointer及四个消融：`paper_runs/multimap_v3_1/formal_training/`
- 传统 PPO：`paper_runs/multimap_v3_2/formal_training/`
- 共 35 个正式模型；
- 每模型 192 条训练记录；
- 每模型 26 个外部验证检查点；
- 每模型 3,000 回合。

正确验证身份：

- `validation_mode=external_multimap_v3_1`
- `validation_instance_count=108`
- 验证集哈希：`64b3e7eb929c5ddc5f8cd2efc3a4c199933c03d038bdbe8cd2ab5acb207388a5`

明确拒绝：

- `external_fixed_v1`
- 旧 64 任务验证集
- 哈希前缀 `bd605…`
- `training_trace_inputs_v2`

纠正后的 D6/D7：

- D6：尾段 20%（回合 ≥ 2400）的跨种子一致性和时间一致性，保持 60/40 定义；
- D7：固定外部验证安全加权覆盖率相对环境交互数的归一化 AUC；
- 公共交互窗口：80–17,702；
- D7 不再使用 0.90 阈值；
- D6/D7 已完成尾段窗口和预算敏感性分析。

### 4.3 v5：M06 文献风格中间版

位置：

`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v5_m06_reference_style/`

v5 使用了正确的固定 108 任务验证数据，但只画中位数＋IQR，没有淡色种子线。用户认为它缺乏明显起伏、视觉上不如第一版。v5 只作中间归档，已被 v6 取代。

### 4.4 v6：当前 M06 和训练动态图权威版本

位置：

`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v6_learning_curves_dual_evidence/`

本轮最终冻结为两张独立图：

1. 正文 M06：固定 108 任务外部验证学习曲线；
2. 补充 S06-Core：三个核心模型训练批次学习动态。

M06：

- 3 模型 × 5 种子 × 26 检查点＝390 条种子记录；
- 78 条中位数/IQR汇总；
- 纵轴：`validation.safe_weighted_coverage`；
- 横轴：0–3000 训练回合；
- 15 条淡色种子线＋3 条中位线＋3 个 IQR 带；
- 无平滑、无补点、无重采样、无样条插值。

S06-Core：

- 3 模型 × 5 种子 × 192 训练记录＝2,880 条种子记录；
- 576 条在 192 个真实记录回合上的中位数/IQR汇总；
- 纵轴：训练批次 `mean_weighted_coverage`；
- 保留真实高频训练波动；
- 不使用旧的 151 点规则网格。

关键文件：

- M06：`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v6_learning_curves_dual_evidence/main/M06_validation_learning_curve_v6.png`
- S06-Core：`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v6_learning_curves_dual_evidence/supplementary/S06_core_training_dynamics_v6.png`
- 脚本：`scripts/v3_2_14_learning_curves_v6.py`
- Source Data：`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v6_learning_curves_dual_evidence/source_data/`
- QA：`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v6_learning_curves_dual_evidence/qa/QA_REPORT.md`
- 文献曲线审计：`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v6_learning_curves_dual_evidence/literature_audit/learning_curve_literature_audit.csv`

v6 已独立验证：

- 15/15 个正式训练 JSONL 逐点回读一致；
- M06 Source Data 为 468 行；
- S06-Core Source Data 为 3,456 行；
- 中英文 PDF、SVG、600 dpi PNG/TIFF 均通过；
- v4、v5、21,648 条结果和21,648条路线未改变；
- D6、D7及下游综合分析未改变。

---

## 5. 下一阶段到底要做什么

用户已经结束旧会话。下一会话的任务不是“一次性重画所有图”，而是：

1. 围绕 v3 的具体图逐张回答问题；
2. 判断问题属于数据、统计含义、图型选择、软件实现还是排版；
3. 根据用户指定的图进行针对性优化；
4. 每次先给出诊断证据，再实施修改；
5. 优化结果写入新目录，不覆盖 v3/v4/v5/v6；
6. 样图经用户确认后再做完整格式导出和 QA。

建议的新输出根目录（只有用户要求实际修改时才创建）：

`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v7_targeted_optimization/`

建议按图建立子目录，例如：

`paper_redraw_multibackend_v7_targeted_optimization/M03/`

每个图子目录至少包含：

- `source_data/`
- `python_sources/`、`origin_projects/`或`matlab_sources/`
- `preview/`
- `exports/`
- `captions_CN/`和必要的英文图注
- `manifest/`
- `qa/`

不要为了优化一张图复制整个 368 MB 的 v3 目录。等逐图均通过后，再由用户决定是否组装统一终版。

### 5.1 每张图的固定处理流程

1. 读取本 HANDOFF；
2. 用户指出图 ID 或具体问题；
3. 打开该图的 v3 PNG/PDF、Source Data、caption、manifest；
4. 判断是否属于训练相关图；若是，切换到 v4/v6 权威来源；
5. 对照相关领域原始权威论文，确认图型语法和统计表达；
6. 先只读诊断，不要未经解释直接改图；
7. 选择最合适后端；
8. 生成一个独立样图，检查实际论文尺寸；
9. 回读图中数值，检查单位、方向、样本量和区间；
10. 用户认可后导出 PDF、SVG（后端可靠时）、600 dpi PNG/TIFF；
11. 保存 Source Data、脚本/OPJU/FIG、图注、manifest和QA；
12. 重新核验旧目录哈希未改变。

### 5.2 后端选择原则

- Origin：常规点—区间、森林图、Cleveland、ECDF、Pareto等，且用户需要 OPJU 后续人工微调时使用；
- Python：雨云/半眼、复杂热力图、performance profile、训练曲线、密集注释和需要严格数据变换的图；
- MATLAB：DSM、三维/2.5D地形、阴影、等高线和路线叠加；
- 不要规定所有图必须用 Origin；
- 也不要为了省事把所有图都改成 Python。

若使用 Origin：

- 本机没有可调用的 Origin MCP；
- 可使用 Origin 2021 COM/ApplicationSI 或真实 UI；
- 应像人工操作一样导入数据、指定列角色、选择合适内置模板、再逐项调整 Plot Details、Axis、图例、标签和页面；
- 不能只执行一次 `plotxy`、自动图例和 Fit Page 就宣布完成；
- 用户此前明确质疑过“是否只是套模板”，因此必须保留人工式精修证据；
- Origin 安装：`D:\Program Files\OriginLab\Origin2021\`。

Python 环境：

`D:\anaconda3\envs\Deeplearning-gpu\python.exe`

已核验版本：Python 3.9.25、Matplotlib 3.7.2、NumPy 1.26.4、Pandas 1.2.4。

MATLAB：v3 V02 使用 MATLAB R2020a；源脚本在：

`paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v3/matlab_sources/render_V02.m`

---

## 6. 用户的制图偏好和硬要求

- 中文为主，算法名、缩写和必要术语保留英文；
- 一张图就是一张独立图片，禁止把多张子图拼成一张；
- 先查原始权威论文，再决定图型、配色、线宽、图例和证据结构；
- 可学习图形语法，但不得复制论文数据、布局、任务实例或受版权保护的视觉资产；
- 图要有真实学术风格，不能有明显软件默认模板感；
- Origin 图应保留 OPJU；Python 图应保留脚本；MATLAB 图应保留 `.m` 和 `.fig`；
- 导出时去掉无效大白边，保留约 1.5–3 mm 安全边距；
- PDF/SVG保留可编辑文字；TIFF/PNG 为 RGB、600 dpi；
- 颜色之外还要有线型、点形或空心/实心编码，保证灰度和色觉缺陷可读；
- 禁止彩虹色图、3D柱状图、雷达总分、装饰性阴影和均值冠军柱状图；
- 不按成绩挑代表任务、路线或种子；
- 用户明确要求：整个制图过程不要调用任何绘图 skill，尤其不要调用 Nature/Nature Figure skill。新会话必须继续遵守。

---

## 7. 已踩过且绝对不能再踩的坑

### 7.1 把训练批次曲线当验证曲线

训练批次 `mean_weighted_coverage` 起伏密集，反映探索和训练任务采样；固定验证 `validation.safe_weighted_coverage` 只有26个检查点，更平滑。两者可以各画一张，但不能混称、拼接或互相替代。

### 7.2 为了“像收敛曲线”制造高频起伏

固定外部验证只有26个真实检查点。禁止插值出更多点、加随机噪声、平滑后再反向制造波动。想展示真实高频变化，应使用独立的训练批次动态图，并明确其训练语义。

### 7.3 把纵轴强制拉到1或裁轴夸大差异

正确 M06 数据实际约在 0.04–0.51，当前纵轴 0–0.55。不要再次画到1造成大量空白和错误观感，也不要把纵轴截在0.45附近夸大 PPO 与 A2C 的细小差异。

### 7.4 只画一条光滑均值线

项目只有5个训练种子。应保留种子轨迹或适当区间。均值/中位数可能很平滑，单独显示会掩盖不稳定性。M06 v6 使用5条淡色种子线＋中位数＋IQR。

### 7.5 用回合横轴直接宣称样本效率

三个模型均训练3,000回合，但实际环境交互数不同。回合横轴适合比较共同训练进程；样本效率必须依赖环境交互数，当前由 D7 公共交互窗口 AUC 处理。

### 7.6 用正式测试集画训练过程

21,648条正式评价只能用于训练结束后的冻结结果。禁止在训练检查点上反复使用正式测试集，禁止把正式测试结果画成收敛曲线。

### 7.7 继续使用 v3 的 M06/M07/S06/S07/S08 数值

这些图的训练数据或D6/D7在v3中已经过时。必须使用第4节规定的v4/v6来源。

### 7.8 无脑套 Origin 默认模板

用户对第二轮 Origin 图质量不满意，核心原因是默认模板感强、自动排版、标签和留白不够精修。使用 Origin 时必须逐项人工式调整，并实际打开 OPJU 检查。

### 7.9 Origin SVG损坏仍强行交付

Origin 2021的部分原生 SVG 会空白、乱码或文字异常。OPJU＋原生PDF才是这些图的可编辑/矢量主版本。不要用损坏 SVG 凑齐格式数量。

### 7.10 V02再次出现裁切和重叠

V02初稿顶部坐标轴显示不全，图例/标注与地形内容重叠。v3现版本已修复。以后改V02必须检查：顶部轴线、三维框、色条、图例、机场标记、优先级点和所有路线的遮挡。

### 7.11 图形自动QA通过就等于视觉合格

自动QA只能检查文件、DPI、哈希、空白、行数和基本边界。每张图仍必须在论文实际尺寸下人工查看字体、线宽、重叠、图例、灰度和信息密度。

### 7.12 结果导向地改权重或归一化

此前用户曾担心 PPO＋Pointer 对 A2C 的优势太小，但最后已经接受不能规定某模型必须领先或规定分差。不要通过权重、归一化区间或图轴操纵排名。任何综合分都要注明事后和权重依赖，并展示敏感性。

### 7.13 把机制结论说得超过消融证据

`no_return_reserve`只能支持返航感知多资源可行性掩码这一复合机制，不能声称能量、距离、时间和动力学子掩码都经过独立消融。

### 7.14 覆盖率、能耗和时间的幸存者偏差

能耗、航程、任务时间只对安全路线统计时，必须同步展示安全样本比例，不能让失败算法因为只剩少量安全路线而看起来更省资源。

---

## 8. 新会话启动顺序

新会话收到用户关于某张图的问题后，按以下顺序工作：

1. 完整读取本 `HANDOFF.md`；
2. 读取 `final_audit_status.json`，确认正式结果仍为21,648；
3. 打开 v3 全图缩略索引；
4. 读取用户指定图的 manifest、caption和Source Data；
5. 若涉及 M06/M07/S06/S07/S08，先切换到v4/v6真值；
6. 明确告诉用户问题属于“数据错误、统计语义、后端实现还是纯视觉”；
7. 先诊断，再根据用户授权修改；
8. 任何修改写到 v7 定向目录；
9. 修改后提供新旧对比、Source Data回读和QA；
10. 不要把逐图优化自动扩大为全套重绘。

推荐的只读启动命令：

```powershell
Get-Content -LiteralPath 'HANDOFF.md' -Raw
Get-Content -LiteralPath 'paper_runs\multimap_v3_2_14\formal_evaluation\results\final_audit_status.json' -Raw
Get-Content -LiteralPath 'paper_runs\multimap_v3_2_14\figures\paper_redraw_multibackend_v3\manifests\figure_registry_manual_v3.json' -Raw
Get-Content -LiteralPath 'paper_runs\multimap_v3_2_14\figures\paper_redraw_multibackend_v3\qa\final_qa_report.json' -Raw
Get-Content -LiteralPath 'paper_runs\multimap_v3_2_14\figures\paper_redraw_multibackend_v6_learning_curves_dual_evidence\qa\QA_REPORT.md' -Raw
```

---

## 9. 当前版本哈希快照

以下组合哈希按“相对路径＋文件SHA-256”排序后再次SHA-256生成，用于后续检查目录是否被意外修改：

| 目录 | 文件数 | 当前组合哈希 |
|---|---:|---|
| `paper_redraw_multibackend_v3` | 240 | `18ba59d33727121104dcb439d510c532b3ddbe699b8301509fe8ab296b4a5357` |
| `paper_redraw_multibackend_v4_training_corrected` | 66 | `43d81aa3a83809e763cbc2a0011783dd5379676293cd2e44350958a645355ddc` |
| `paper_redraw_multibackend_v5_m06_reference_style` | 18 | `534f9a2315af56f1bb852a7c94096f77d36ae3c93a73a0b51bdb83053426d745` |
| `paper_redraw_multibackend_v6_learning_curves_dual_evidence` | 37 | `d11ed6ffb2238af58bded9f14d24c431db5ebfdaa29734ce309b022d04e21118` |

注意：组合哈希算法必须与上述定义一致；不要用“把整个目录压缩后再哈希”等不同算法比较。

---

## 10. 当前停止点

截至本交接：

- 训练、正式评价、统计和20张v3完整图均已完成；
- 训练曲线错误已通过v4–v6完成纠正；
- 最新M06和S06-Core双证据曲线已经通过数值、视觉、格式和哈希审计；
- 尚未创建v7定向优化目录；
- 尚未开始下一张图的具体优化；
- 下一会话应等待用户指出想优化的图或提出具体问题，然后按第8节逐图推进。

不要自行假定用户想先改哪张图，也不要在没有新要求时批量重绘。
