# 代码变更报告

## 变更范围

- 新增 `uav_inspection/figures/v3_2_14_drones_style_v7.py`：复制冻结 Source Data、构建 S09、统一双语风格、导出 PDF/SVG/PNG/TIFF、生成逐图规范与 QA 元数据。
- 新增 `editable/matlab/render_V02_v7.m`：原生 MATLAB 双语三维 DSM 路线图与 FIG 导出。
- 新增 `scripts/export_origin_native_v7.py`：只读打开 10 个 OPJU 并导出原生 PDF 以验证 Origin 后端和工程可编辑性。
- 未修改训练、模型、正式评价、checkpoint、冻结结果、冻结路线或 v3–v6 图件。

## 关键可调参数

主要参数集中在 `uav_inspection/figures/v3_2_14_drones_style_v7.py` 顶部：

- `CORE_COLORS`：三个核心模型的固定颜色。调整会同步影响中英文全部图。
- `WIDTH_MM` / `HEIGHT_MM`：Elsevier 140/190 mm 画布与各图内容高度。
- `TICK_PT` / `AXIS_PT` / `LEGEND_PT`：8/9/8 pt 字号系统。
- `AXIS_LW` / `GRID_LW` / `MAIN_LW` / `SEED_LW`：0.8/0.5/1.45/0.6 pt 线宽层级。
- `PAD_MM`：1.5 mm 外安全边距目标。
- `PNG_DPI` / `TIFF_1000_IDS`：600 dpi PNG 与纯线图 1000 dpi TIFF 路由。

V02 的视角、画布、坐标轴、色标和图例区域集中在 `editable/matlab/render_V02_v7.m`。调整视角会改变遮挡关系；调整轴区或图例区会影响裁切与最终可读性，因此应重新运行 42 项 QA。

## 验证结果

- 冻结正式审计：`passed=true`，结果和路线均 21,648 条，`ppo_mlp_absent=true`。
- M06：390 条种子记录 + 78 条汇总记录。
- S09：2,880 条种子记录 + 576 条汇总记录，15 份日志均为 192 个训练批次。
- 双语自动 QA：42/42 PASS，0 warning。
- PDF：42/42 单页且字体为嵌入字体或 PDF Base-14 标准字体。
- Origin：10/10 OPJU 只读加载并成功导出原生 PDF。
- v3、v4、v5、v6 旧目录哈希和文件数均未漂移。
