# 保研面试科研成果图素材包

本目录只整理 `paper_redraw_multibackend_v7_drones_style_fullsuite` 中已经冻结并通过交付审计的图件，不修改训练、正式评价、Source Data 或 v7 原图。PPT 建议按文件名前缀 `01`–`04` 的顺序使用。

## 推荐展示顺序

| 顺序 | 图件 | PPT中的作用 | 建议结论 |
|---|---|---|---|
| 1 | `exports/01_V02_real_DSM_route_en.png` | 建立山区真实DSM应用场景 | 模型能在真实地形数据构建的仿真环境中输出满足任务约束的路线 |
| 2 | `exports/02_M01_map_level_distribution_zh.png` | 展示跨地图总体定量结果 | PPO+Pointer 与 A2C+Pointer 在未见合成地图和真实DSM地图上明显优于传统PPO |
| 3 | `exports/03_S09_training_return_dynamics_zh.png` | 展示强化学习训练回报动态 | PPO+Pointer 与 A2C+Pointer 学到稳定策略，传统PPO的回报长期处于较低水平 |
| 4 | `exports/04_M10_paired_ablation_zh.png` | 说明关键机制 | 返航感知多资源可行性掩码是安全加权覆盖率的关键机制 |

如果汇报时间不足，只展示前三张。V02 使用英文版，因为当前 v7 中文版色条标题存在方框乱码；这不是数据问题，英文版内容与数据均正常。S06和M06保存在 `alternatives/`，只在老师追问七模型训练差异或固定外部验证时调用。

## 绘图软件与工具一览

以下说明的是本素材包所选最终图片的实际生成工具，而不只是历史版本的计划后端。Python 是编程语言，Matplotlib 是实际绘图库；pandas、NumPy 和 SciPy 分别承担数据读取、数值处理和相关统计计算，不应混称为绘图软件。

| 图件 | 最终图片使用的软件／库 | 具体绘制工具 | 可核验源文件（本目录内） |
|---|---|---|---|
| V02 | MATLAB R2020a | `surf` 地形表面、`contour3` 等高线、`scatter3` 巡检点、`plot3` 路线 | `editable/render_V02_v7.m` |
| M01 | Python + Matplotlib；SciPy 辅助计算密度 | `scipy.stats.gaussian_kde` 计算核密度；Matplotlib 的 `fill_between`、`scatter`、`plot` 绘制分布、散点和中位数 | `editable/v3_2_14_drones_style_v7.py` → `plot_m01` |
| S09 | Python + Matplotlib | `plot` 绘制种子轨迹与中位数，`fill_between` 绘制四分位距阴影 | 同上 → `plot_s09` / `plot_learning` |
| M10 | 最终 v7 图片：Python + Matplotlib；另保留 OriginPro 2021 工程 | `scatter` 绘制效应点，`hlines` 绘制区间；Origin OPJU 用于保留可编辑工程，不是所选最终 PNG 的直接导出来源 | 同上 → `plot_m10`；另有 `editable/M10.opju` |
| S06（备选） | Python + Matplotlib | `plot` 绘制七模型中位数曲线，`fill_between` 绘制四分位距阴影 | 同上 → `plot_s06` / `plot_learning` |
| M06（备选） | Python + Matplotlib | `plot` 绘制独立种子与中位数曲线，`fill_between` 绘制四分位距阴影 | 同上 → `plot_learning` |

包内主图清单记录的 Python 绘图环境为 Python 3.9.25 / Matplotlib 3.7.2。答辩时可直接回答“使用 Python 的 Matplotlib 库绘制”或“使用 MATLAB 绘制”，随后再解释数据来源；软件名称本身不是真实性证明，真实性仍需通过 Source Data、绘图源文件和冻结结果对应关系核验。

## 01 V02：真实DSM三维路线

**绘图软件／工具：MATLAB R2020a，使用内置三维绘图函数 `surf`、`contour3`、`scatter3` 和 `plot3`，由 `.m` 脚本生成。**

**如何绘制**

- 冻结任务：`real_test__cn_taihang__road_00__task_08`；路线种子：42。
- `terrain.csv` 包含 71,289 个高程点，`roads.csv` 包含 99 个道路点，`points.csv` 包含 25 个任务点，`routes.csv` 保存四种方法的路线坐标。
- MATLAB 使用 `surf` 绘制DSM表面、`contour3` 叠加等高线、`scatter3` 显示优先级巡检点、`plot3` 显示各算法路线。
- 任务、地图和原始路线文件哈希记录在 `source_data/V02/metadata.csv`；绘图源文件为 `editable/render_V02_v7.m`。

**20秒真实性回答**

> 这不是人工描绘的示意路线。图中地形、巡检点和路线都从冻结正式评价文件读取，再由MATLAB叠加到DSM高程表面。该任务事先固定，没有根据结果挑选；它代表真实地形数据上的仿真迁移，不是实飞验证。

**禁止过度表述**

- 不得称为真实飞行、外场试验或安全认证。
- 该图是描述性代表任务，不能单独证明总体统计优势。

## 02 M01：地图级覆盖率分布

**绘图软件／工具：Python 的 Matplotlib 库；SciPy 的 `gaussian_kde` 计算核密度，pandas 和 NumPy 辅助读取、整理与计算数据。不是用 Origin 绘制。**

**如何绘制**

- `source_data/M01_source_data.csv` 共 192 行：24 张未见合成地图和 8 张真实DSM地图，分别统计 6 种方法。
- 每个空心点对应一张地图；横坐标是该地图的优先级加权覆盖率。
- 半眼曲线使用 Scott 带宽的高斯核密度估计；短竖线为地图级中位数。
- 纵向抖动仅避免散点重叠，不改变任何横坐标数值；随机种子固定为 `20260805`。
- 最终图由 `editable/v3_2_14_drones_style_v7.py` 中的 `plot_m01` 生成。

**20秒真实性回答**

> 图中每个点是一张独立地图上的聚合结果，曲线只用于表示这些地图级数值的分布，竖线是中位数。统计单位是地图，没有把重复任务或规划种子伪装成更多独立地图，也没有删除失败地图或按成绩挑选任务。

**禁止过度表述**

- 真实DSM只有 8 张独立地图，不能把地图内的任务数量当成独立地理样本数。
- PPO+Pointer并非所有原始覆盖场景中的绝对冠军，不得表述为“在所有条件下最优”。

## 03 S09：训练回报动态

**绘图软件／工具：Python 的 Matplotlib 库，使用 `plot` 绘制曲线、`fill_between` 绘制阴影；pandas 和 NumPy 辅助处理 Source Data。不是用 Origin 绘制。**

**为什么调整为主图**

- 上传的 Drones 权威论文 Figure 5 和 Figure 12 均使用训练回报曲线；同领域的PPO、DQN和多无人机路径规划原始研究也普遍用平均回报、回报波动或成功率曲线展示学习过程。
- S09比单条平滑均值线更严格：它保留15条独立种子轨迹、中位数和四分位距，而且没有时间平滑。
- M01已经承担最终跨地图性能证据，因此S09在PPT中只负责回答“策略是否通过强化学习逐步学到”，不承担最终测试结论。

**如何绘制**

- `source_data/S09_source_data.csv` 共 3,456 行：2,880 行种子记录和 576 行逐检查点汇总。
- 数据结构为 3 种模型 × 5 个独立训练种子 × 192 个训练记录；每个观测是16回合训练批次的平均episode return。
- 15条淡线是独立种子原始轨迹，粗线是逐记录回合中位数，阴影是四分位距。
- 没有时间平滑、插值、样条、补点或人为噪声。
- 最终图由 `editable/v3_2_14_drones_style_v7.py` 中的 `plot_learning` / `plot_s09` 生成。

**20秒真实性回答**

> 每个观测都是训练日志中真实保存的16回合批次平均回报，淡线对应五个独立训练种子，粗线和阴影是同一训练回合位置上的种子中位数和四分位距。图没有做时间平滑或插值，因此保留了真实训练波动。

**禁止过度表述**

- 回报由项目奖励函数定义，不能直接等同于安全加权覆盖率或最终泛化性能。
- 这只是训练证据；最终效果必须由M01等冻结正式评价图支撑。

## 04 M10：配对消融效应

**绘图软件／工具：所选最终 v7 图片使用 Python 的 Matplotlib 库绘制（`scatter` 画点、`hlines` 画区间）；另保留 OriginPro 2021 的 `.opju` 可编辑工程。不能将这张最终 PNG 说成由 Origin 直接导出。**

**如何绘制**

- `source_data/M10_source_data.csv` 共 8 行：2 个域乘以 4 个消融模型。
- 未见合成域以 24 张地图配对，真实DSM域以 8 张地图配对。
- 圆点是 Hodges–Lehmann 配对效应估计，横线是 bootstrap 区间，竖虚线是零效应基准。
- 显著性来自配对 Wilcoxon 检验，并经过 Holm 多重比较校正。
- 最终统一双语导出由 `editable/v3_2_14_drones_style_v7.py` 的 `plot_m10` 完成；`editable/M10.opju` 保留Origin可编辑工程。

**20秒真实性回答**

> 这里不是比较两个独立样本的均值，而是在同一地图上计算完整模型和消融模型的配对差值。点是稳健效应估计，区间通过bootstrap得到，星号对应经过Holm校正后的配对检验结果。

**禁止过度表述**

- 只能说“返航感知多资源可行性掩码这一复合机制有效”。
- 能量、距离、时间和动力学子掩码没有分别独立消融，不能声称每个子机制均得到独立证明。
- 其余三个消融在该指标上未达到校正后显著，不得包装成全部显著。

## 备选 S06：七模型训练批次覆盖率

**绘图软件／工具：Python 的 Matplotlib 库，使用 `plot` 和 `fill_between` 绘制曲线及四分位距阴影；与 S09 共用 `plot_learning` 绘图函数。不是用 Origin 绘制。**

**如何绘制**

- `source_data/S06_source_data.csv` 保留七个学习模型、五个种子的训练批次覆盖率记录和汇总。
- 图中线和阴影为跨种子中位数与四分位距，用于展示完整模型、学习基线和四个消融的训练动态。
- 它是训练批次证据，不是固定外部验证或最终测试。

**20秒真实性回答**

> 这张图比较七个学习模型在训练批次上的优先级加权覆盖率。曲线按五个独立训练种子汇总，反映的是训练任务采样和探索过程，因此不能用某条训练曲线较高来替代最终泛化比较。

**禁止过度表述**

- “无域随机化”在训练批次中较高，不代表其在未见地图或真实DSM上的泛化更好。
- 七条曲线信息密度很高，因此仅作为老师追问消融训练动态时的备选图。

## 备选 M06：固定外部验证学习曲线

**绘图软件／工具：Python 的 Matplotlib 库，由 `plot_learning` 中的 `plot` 和 `fill_between` 绘制种子线、中位数线与四分位距阴影。不是用 Origin 绘制。**

- `alternatives/M06_fixed_validation_learning_curve_zh.png` 是3个核心模型在固定108任务外部验证上的曲线。
- 淡线为5个独立训练种子，粗线与阴影为中位数和四分位距；没有平滑、插值或补点。
- 当老师质疑“训练回报是否真正对应外部任务表现”时，用M06补充回答；它比S09更接近验证性能，但不如S09符合路径规划文献常见的训练回报展示习惯。

## 文件结构

- `exports/`：按PPT展示顺序编号的4张600 dpi PNG。
- `alternatives/`：S06和M06两张按需调用的备选图。
- `source_data/`：主图与备选训练图对应的Source Data或V02数据表。
- `editable/`：Python、MATLAB和Origin可编辑源文件。
- `captions/`：原始双语图注。
- `qa/source_figure_reports/`：从v7复制的逐图QA报告。
- `qa/thumbnail_index.png`：4张入选图的缩略索引，仅用于浏览，不替代原图。
- `literature_figure_selection_audit.md`：上传论文和同领域原始研究的图目惯例审计。
- `manifests/`：素材包文件哈希与来源对应关系。

PPT插图时直接使用 `exports/` 中的PNG，不要从聊天窗口、缩略索引或截图中复制。
