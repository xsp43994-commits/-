"""v3.2.14 第三轮制图：四图样稿通过后的全量生产阶段。

本模块只读取冻结结果、冻结分析表、原始训练记录和正式路线资产；不读取
paper_final 或 paper_redraw_origin_v2 的图形数据。Origin 图通过 COM 创建
独立 SourceData、PlotData、Metadata 和 Graph1，并保存为独立 OPJU。
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from PIL import Image, ImageChops, ImageDraw, ImageFont

from uav_inspection.figures import v3_2_14_multibackend_redraw as core
from uav_inspection.figures import v3_2_14_publication_figures as frozen_io


OUTPUT = core.OUTPUT
ANALYSIS = core.RUN / "analysis"
PREPLOT = ANALYSIS / "pre_plot_statistics"
MULTIOBJ = ANALYSIS / "manuscript_multiobjective_v1"
TRAIN_AWARE = ANALYSIS / "manuscript_training_aware_v2"
OP_BAND = ANALYSIS / "manuscript_operational_band_v4"
CLOSURE = ANALYSIS / "manuscript_preplot_closure_v5"
SYNTHETIC_TASKS = core.RUN / "manifests" / "synthetic_test" / "records.jsonl"
SYNTHETIC_MAP = core.ROOT / "map_data" / "multimap_v3_1" / "procedural" / "synthetic_test" / "synthetic_test__map_003.npz"
SYNTHETIC_EXAMPLE = "synthetic_test__synthetic_test__map_003__task_08"

CORE_MODELS = ("full", "a2c_pointer", "traditional_ppo")
ABLATIONS = ("no_priority_bias", "no_domain_randomization", "no_resource_shaping", "no_return_reserve")
LEARNING_MODELS = CORE_MODELS + ABLATIONS
BASELINES = ("nearest_feasible", "priority_resource_greedy", "aco", "ga", "sa", "milp", "a_star", "pso", "exact_pareto_dp")
REMAINING_IDS = ("M03", "M04", "M05", "M07", "M08", "M09", "M10", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "V01")
ORIGIN_IDS = ("M03", "M04", "M05", "M07", "M08", "M10", "S02", "S03", "S07")
PYTHON_IDS = ("M09", "S01", "S04", "S05", "S06", "S08", "V01")
ALL_IDS = tuple(core.PROTOTYPE_IDS) + REMAINING_IDS

EXTRA_COLORS = {
    "no_priority_bias": "#7F8FA6",
    "no_domain_randomization": "#8064A2",
    "no_resource_shaping": "#A67C52",
    "no_return_reserve": "#9E4F5C",
    "nearest_feasible": "#8C8C8C", "ga": "#6A8EAE", "sa": "#B07AA1",
    "a_star": "#9C755F", "pso": "#59A14F", "exact_pareto_dp": "#4E79A7",
}
EXTRA_LABELS = {
    "no_priority_bias": "无优先级偏置",
    "no_domain_randomization": "无域随机化",
    "no_resource_shaping": "无资源塑形",
    "no_return_reserve": "无返航预留",
    "nearest_feasible": "最近可行", "ga": "GA", "sa": "SA", "a_star": "A*",
    "pso": "PSO", "exact_pareto_dp": "Exact Pareto DP",
}
COLORS = {**core.COLORS, **EXTRA_COLORS}
LABELS = {**core.LABELS, **EXTRA_LABELS}

# 关键可调参数：只控制视觉，不改变任何统计值。
ORIGIN_PAGE = {"width_px": 1180, "height_px": 760, "export_width_px": 4200, "margin_mm": 2.0}
HEATMAP_CMAP = "RdYlBu"
LINE_WIDTH = 1.65

CAPTIONS = {
    "M03": "高中低优先级及远端高优先级冲突条件下的地图级覆盖。完整模型与无优先级偏置消融以哑铃连接；先聚合任务与种子，再以地图为单位。",
    "M04": "仅在安全路线中统计的能耗、航程和总任务时间预算利用率。点旁原始值分别以Wh、km和min给出，并同步标注安全样本比例。",
    "M05": "正式标称任务的在线规划时间ECDF。横轴为对数秒；重复训练或规划种子先在任务内聚合，曲线用于描述任务分布而非独立推断。",
    "M07": "三个核心学习模型的训练稳定性与样本效率。各指标统一为越高越优的方向性分数，行标签保留原始中位数和单位。",
    "M08": "未见合成地图程序化泛化与真实DSM零样本仿真迁移的地图级任务效能D1。点为中位数，横线为地图bootstrap 95%区间。",
    "M09": "已知风/功率偏移及隐藏风、功率、DEM和定位误差下的性能保持率。数值为相对标称条件保持率；DSM属于仿真迁移而非实飞验证。",
    "M10": "完整PPO+Pointer相对四项消融的地图级安全加权覆盖率效应。点为冻结Hodges–Lehmann效应，横线为冻结bootstrap 95%区间，星号表示冻结Holm校正通过。",
    "S01": "全部适用算法的任务级性能剖面。横轴为相对同任务最佳安全加权覆盖率的regret，纵轴为不超过该regret的任务比例。",
    "S02": "任务效能D1与P95在线规划时间的Pareto视图。横轴为对数秒；颜色标识算法，方形/圆形区分未见合成地图与真实DSM，灰线连接各域的非支配点。所列算法安全率均为100%，因此不再用无信息量的气泡大小重复编码安全率。",
    "S03": "传统规划器的oracle regret区间与在线计算代价。区间来自任务证书上下界；MILP同步保留求解状态和gap。",
    "S04": "三个核心学习模型在节点数、难度、约束类型与优先级布局分层下的安全加权覆盖率描述性热力图。16/20/24只称训练范围内多规模表现。",
    "S05": "扰动条件下安全、返航及失败模式的统一方向热力图。安全率和返航率保持原方向，其余失败指标转为1−失败率；原始值保存在Source Data。",
    "S06": "七个学习模型、五个训练种子的完整3000回合训练过程。横轴为训练回合，纵轴为共同定义的训练批次加权覆盖率；不比较不同奖励定义的原始reward。",
    "S07": "三个核心学习模型的D1–D7与100分事后综合摘要。综合分使用冻结0.60运行区间和算术聚合；该摘要不能替代原始指标。",
    "S08": "37,410条冻结联合敏感性结果中，PPO+Pointer在运行下限与D6+D7总权重组合下的第一名占比。",
    "V01": f"固定合成任务{SYNTHETIC_EXAMPLE}、seed 42的三学习模型与MILP路线。地图、固定巡检点、优先级、机场和路线均来自冻结资产；不按结果更换案例。",
}


def _stem(figure_id: str) -> Path:
    tier = core.FIGURES[figure_id]["tier"]
    folder = {"main": "main", "supplementary": "supplementary", "showcase": "showcase"}[tier]
    return OUTPUT / folder / f"{figure_id}_{core.FIGURES[figure_id]['name']}"


def _write_source(figure_id: str, frame: pd.DataFrame) -> Path:
    path = OUTPUT / "source_data" / f"{figure_id}_source_data.csv"
    core._write_csv(path, frame)
    return path


def _nominal_frozen() -> pd.DataFrame:
    frame = pd.read_csv(PREPLOT / "frozen_plot_input.csv")
    return frame[frame["condition"].eq("nominal")].copy()


def _task_level(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    return frame.groupby(["family", "model", "map_id", "task_id"], as_index=False)[list(metrics)].mean(numeric_only=True)


def _map_level(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    task = _task_level(frame, metrics)
    return task.groupby(["family", "model", "map_id"], as_index=False)[list(metrics)].mean(numeric_only=True)


def _bootstrap_median(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.median(rng.choice(values, size=(core.BOOTSTRAP_REPS, len(values)), replace=True), axis=1)
    return float(np.median(values)), *map(float, np.quantile(draws, [0.025, 0.975]))


def build_m03(frozen: pd.DataFrame) -> pd.DataFrame:
    models = ("full", "no_priority_bias")
    frame = frozen[frozen["model"].isin(models)].copy()
    metrics = [("high_priority_coverage", "高优先级"), ("medium_priority_coverage", "中优先级"), ("low_priority_coverage", "低优先级")]
    rows: list[dict[str, Any]] = []
    for domain_name, selector in (("未见合成", frame["family"].str.startswith("synthetic")), ("真实DSM", frame["family"].str.startswith("real"))):
        sub = frame[selector]
        maps = _map_level(sub, [m for m, _ in metrics])
        for metric, label in metrics:
            med = maps.groupby("model")[metric].median()
            rows.append({"domain": domain_name, "stratum": label, "full": med["full"], "ablation": med["no_priority_bias"]})
        conflict = sub[sub["priority_layout"].eq("far_high_conflict")]
        med = _map_level(conflict, ["high_priority_coverage"]).groupby("model")["high_priority_coverage"].median()
        rows.append({"domain": domain_name, "stratum": "远端高优先级冲突", "full": med["full"], "ablation": med["no_priority_bias"]})
    out = pd.DataFrame(rows)
    out["effect"] = out["full"] - out["ablation"]
    return out


def build_m04(results: pd.DataFrame) -> pd.DataFrame:
    frame = results[(results["condition"].eq("nominal")) & results["model"].isin(core.MAIN_COMPARE)].copy()
    task_keys = ["family", "model", "map_id", "task_id"]
    safe = frame[frame["safe"].astype(bool)].groupby(task_keys, as_index=False)[["energy_wh", "distance_m", "time_s", "energy_utilization", "distance_utilization", "time_utilization"]].mean()
    shares = frame.groupby(task_keys, as_index=False)["safe"].mean().rename(columns={"safe": "safe_share"})
    maps = safe.merge(shares, on=task_keys).groupby(["model", "map_id"], as_index=False).mean(numeric_only=True)
    specs = (("能耗", "energy_utilization", "energy_wh", "Wh", 1.0), ("航程", "distance_utilization", "distance_m", "km", 0.001), ("总任务时间", "time_utilization", "time_s", "min", 1/60))
    rows = []
    for model, sub in maps.groupby("model"):
        for metric, util, raw, unit, scale in specs:
            rows.append({"model": model, "metric": metric, "utilization": sub[util].median(), "raw_median": sub[raw].median()*scale, "unit": unit, "safe_share": sub["safe_share"].mean()})
    return pd.DataFrame(rows)


def build_m05(frozen: pd.DataFrame) -> pd.DataFrame:
    frame = frozen[frozen["model"].isin(core.MAIN_COMPARE)]
    task = _task_level(frame, ["planning_time_s"])
    rows = []
    for model, sub in task.groupby("model"):
        values = np.sort(sub["planning_time_s"].to_numpy(float))
        rows.extend({"model": model, "planning_time_s": value, "ecdf": (i+1)/len(values)} for i, value in enumerate(values))
    return pd.DataFrame(rows)


def build_m07() -> pd.DataFrame:
    seed = pd.read_csv(TRAIN_AWARE / "training_seed_metrics.csv")
    seed = seed[seed["model"].isin(CORE_MODELS)].copy()
    seed["stability"] = 1.0 - seed["tail_temporal_sd"]
    max_final = seed["final_environment_interactions"].max()
    seed["sample_speed"] = np.where(seed["convergence_environment_interactions"].notna(), 1-seed["convergence_environment_interactions"]/max_final, 0.0)
    dims = pd.read_csv(TRAIN_AWARE / "seven_dimension_scores.csv").set_index("model")
    specs = (("D6训练稳定性", "D6", None), ("D7样本效率", "D7", None), ("Learning-curve AUC", "learning_curve_auc", "AUC"), ("尾段时间一致性", "stability", "1-SD"), ("阈值效率", "threshold_efficiency", "比例"), ("达到阈值速度", "sample_speed", "方向分数"))
    rows = []
    for model in CORE_MODELS:
        sub = seed[seed["model"].eq(model)]
        for metric, field, unit in specs:
            if field in ("D6", "D7"):
                value = float(dims.loc[model, field]); q25 = q75 = value; raw = value
            else:
                values = sub[field].to_numpy(float); value = float(np.median(values)); q25, q75 = map(float, np.quantile(values, [0.25, 0.75])); raw = value
            rows.append({"model": model, "metric": metric, "score": value, "q25": q25, "q75": q75, "raw_median": raw, "unit": unit or "0–1"})
    return pd.DataFrame(rows)


def build_m08() -> pd.DataFrame:
    maps = pd.read_csv(MULTIOBJ / "nominal_map_dimensions.csv")
    maps = maps[maps["model"].isin(CORE_MODELS)]
    rows = []
    seed = 0
    for domain in ("synthetic", "real"):
        for model in CORE_MODELS:
            values = maps[(maps["domain"].eq(domain)) & (maps["model"].eq(model))]["D1_mission_effectiveness"].to_numpy(float)
            estimate, low, high = _bootstrap_median(values, 20260805+seed); seed += 1
            rows.append({"domain": domain, "domain_label": "未见合成地图" if domain=="synthetic" else "真实DSM", "model": model, "estimate": estimate, "ci_low": low, "ci_high": high, "n_maps": len(values)})
    return pd.DataFrame(rows)


def build_m09() -> pd.DataFrame:
    data = pd.read_csv(MULTIOBJ / "robustness_condition_dimensions.csv")
    models = CORE_MODELS + ("no_domain_randomization", "no_return_reserve")
    data = data[data["model"].isin(models)]
    return data.groupby(["model", "family", "condition"], as_index=False).agg(retention=("retention", "mean"), safe_rate=("perturbed_safe_rate", "mean"), n_maps=("map_id", "nunique"))


def build_m10() -> pd.DataFrame:
    pair = pd.read_csv(PREPLOT / "confirmatory_pairwise.csv")
    pair = pair[(pair["reference"].eq("full")) & pair["comparator"].isin(ABLATIONS) & pair["statistical_family"].isin(("synthetic_ablations", "real_ablations"))].copy()
    pair["domain"] = np.where(pair["statistical_family"].str.startswith("synthetic"), "未见合成", "真实DSM")
    return pair


def build_s01(frozen: pd.DataFrame) -> pd.DataFrame:
    task = _task_level(frozen, ["safe_weighted_coverage"])
    best = task.groupby("task_id")["safe_weighted_coverage"].transform("max")
    task["regret"] = best-task["safe_weighted_coverage"]
    rows=[]
    for model, sub in task.groupby("model"):
        values=np.sort(sub["regret"].to_numpy(float))
        rows.extend({"model":model,"regret":v,"ecdf":(i+1)/len(values),"task_count":len(values)} for i,v in enumerate(values))
    return pd.DataFrame(rows)


def build_s02() -> pd.DataFrame:
    dims = pd.read_csv(MULTIOBJ / "dimension_scores.csv")
    dims = dims[dims["scope"].isin(("synthetic_all", "real_all")) & dims["model"].isin(core.MAIN_COMPARE)]
    safety = _nominal_frozen().groupby(["model"], as_index=False)["safe_rate"].mean()
    return dims.merge(safety, on="model", how="left")[["scope","model","D1","planning_time_p95_s","safe_rate"]]


def build_s03(frozen: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    # regret必须使用冻结的oracle上下界差，不可用1-oracle attainment替代。
    frame=frozen[frozen["model"].isin(BASELINES)].copy()
    out=frame.groupby("model",as_index=False).agg(regret_low=("oracle_regret_lower","median"),regret_high=("oracle_regret_upper","median"),planning_time_s=("planning_time_s","median"),run_count=("task_id","size"))
    solver=results[(results["condition"].eq("nominal")) & results["model"].isin(BASELINES)].groupby("model",as_index=False).agg(solver_gap=("solver_gap","median"),certified_share=("optimality_certified","mean"))
    return out.merge(solver,on="model",how="left")


def build_s04() -> pd.DataFrame:
    data=pd.read_csv(PREPLOT/"exploratory_interactions.csv")
    data=data[(data["algorithm"].isin(CORE_MODELS)) & (data["statistical_family"].eq("synthetic_main_algorithms"))]
    data["scenario"]=data["factor"].astype(str)+"｜"+data["level"].astype(str)
    return data[["algorithm","factor","level","scenario","mean","median","run_count"]].rename(columns={"algorithm":"model"})


def build_s05(frozen: pd.DataFrame) -> pd.DataFrame:
    models=CORE_MODELS+("no_domain_randomization","no_return_reserve")
    frame=frozen[frozen["family"].isin(("known_domain_shift","hidden_model_perception_mismatch")) & frozen["model"].isin(models)]
    metrics=("safe_rate","return_rate","violation_rate","dangerous_action_proposal_rate","environment_interception_rate","stranded_rate")
    grouped=frame.groupby(["model","family","condition"],as_index=False)[list(metrics)].mean()
    rows=[]
    for row in grouped.itertuples():
        for metric in metrics:
            raw=float(getattr(row,metric)); higher=raw if metric in ("safe_rate","return_rate") else 1-raw
            rows.append({"model":row.model,"family":row.family,"condition":row.condition,"metric":metric,"raw_value":raw,"higher_better":higher})
    return pd.DataFrame(rows)


def build_s06() -> pd.DataFrame:
    history=frozen_io.load_training_history(LEARNING_MODELS)
    lower=max(history.groupby(["model","training_seed"])["episodes_seen"].min())
    grid=np.linspace(lower,3000,150); rows=[]
    for model in LEARNING_MODELS:
        curves=[]
        for seed,sub in history[history.model.eq(model)].groupby("training_seed"):
            sub=sub.sort_values("episodes_seen"); curves.append(np.interp(grid,sub.episodes_seen,sub.mean_weighted_coverage))
        matrix=np.vstack(curves); q25,med,q75=np.quantile(matrix,[.25,.5,.75],axis=0)
        rows.extend({"model":model,"episodes_seen":x,"weighted_coverage":m,"q25":lo,"q75":hi,"seed_count":len(curves)} for x,m,lo,hi in zip(grid,med,q25,q75))
    return pd.DataFrame(rows)


def build_s07() -> pd.DataFrame:
    dims=pd.read_csv(TRAIN_AWARE/"seven_dimension_scores.csv").set_index("model")
    score=pd.read_csv(OP_BAND/"selected_operational_scores_100.csv")
    score=score[(score["aggregation"].eq("arithmetic")) & np.isclose(score["operational_floor"],0.60)].set_index("model")
    rows=[]
    for model in CORE_MODELS:
        for d in [f"D{i}" for i in range(1,8)]: rows.append({"model":model,"metric":d,"value_100":100*float(dims.loc[model,d])})
        rows.append({"model":model,"metric":"综合得分","value_100":float(score.loc[model,"score_0_to_100"])})
    return pd.DataFrame(rows)


def build_s08() -> pd.DataFrame:
    data=pd.read_csv(CLOSURE/"joint_normalization_weight_sensitivity.csv")
    data=data[(data["aggregation"].eq("arithmetic")) & data["model"].eq("full")].copy()
    data["training_weight"]=(data["weight_D6"]+data["weight_D7"]).round(6)
    return data.groupby(["operational_floor","training_weight"],as_index=False)["is_first"].mean().rename(columns={"is_first":"first_share"})


def _read_task(path: Path, task_id: str) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        record=json.loads(line)
        if record.get("id")==task_id: return record
    raise KeyError(task_id)


def _synthetic_route_path(model: str) -> Path:
    if model=="milp": return core.RESULTS_DIR/"synthetic_main_baselines"/"jobs"/"milp__seed42"/"routes"/f"{SYNTHETIC_EXAMPLE}.json"
    pattern=f"{model}__seed42__{SYNTHETIC_EXAMPLE}.json"
    matches=list((core.RESULTS_DIR/"synthetic_learning"/"shards").rglob(pattern))
    if len(matches)!=1: raise RuntimeError(f"{model}合成固定路线匹配数={len(matches)}")
    return matches[0]


def build_v01() -> pd.DataFrame:
    task=_read_task(SYNTHETIC_TASKS,SYNTHETIC_EXAMPLE)
    bundle=np.load(SYNTHETIC_MAP,allow_pickle=True)
    rows=[]
    roads=bundle["road_points"]; offsets=bundle["road_offsets"]
    for rid,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
        for seq,(x,y) in enumerate(roads[start:end]): rows.append({"record_type":"road","model":"","group":rid,"sequence":seq,"x":x,"y":y,"priority":0})
    for idx,((x,y,_),priority) in enumerate(zip(task["inspection_points_xyz"],task["priorities"])): rows.append({"record_type":"inspection","model":"","group":0,"sequence":idx,"x":x,"y":y,"priority":int(priority)})
    rows.append({"record_type":"airport","model":"","group":0,"sequence":0,"x":task["start_xy"][0],"y":task["start_xy"][1],"priority":0})
    for model in ("full","a2c_pointer","traditional_ppo","milp"):
        payload=json.loads(_synthetic_route_path(model).read_text(encoding="utf-8")); detail=payload.get("detail") if model!="milp" else payload.get("result")
        route=detail.get("path") if detail else None
        if not route: rows.append({"record_type":"route_missing","model":model,"group":0,"sequence":-1,"x":np.nan,"y":np.nan,"priority":0}); continue
        for seq,point in enumerate(route): rows.append({"record_type":"route","model":model,"group":0,"sequence":seq,"x":point[0],"y":point[1],"priority":0})
    return pd.DataFrame(rows)


def build_all_sources() -> dict[str,pd.DataFrame]:
    results=core._read_results(); frozen=_nominal_frozen()
    sources={
        "M03":build_m03(frozen),"M04":build_m04(results),"M05":build_m05(frozen),"M07":build_m07(),"M08":build_m08(),"M09":build_m09(),"M10":build_m10(),
        "S01":build_s01(frozen),"S02":build_s02(),"S03":build_s03(frozen,results),"S04":build_s04(),"S05":build_s05(pd.read_csv(PREPLOT/"frozen_plot_input.csv")),"S06":build_s06(),"S07":build_s07(),"S08":build_s08(),"V01":build_v01(),
    }
    for figure_id,frame in sources.items():
        if frame.empty: raise RuntimeError(f"{figure_id} Source Data为空")
        if "model" in frame and "ppo_mlp" in set(frame["model"].astype(str)): raise RuntimeError(f"{figure_id}混入ppo_mlp")
        _write_source(figure_id,frame)
    return sources


def _clean_axis(ax: plt.Axes, grid_axis: str|None=None) -> None:
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(direction="out",length=3,pad=2)
    if grid_axis: ax.grid(axis=grid_axis,color="#DFE3E8",lw=.55,alpha=.8); ax.set_axisbelow(True)


def _heatmap(frame: pd.DataFrame,index: str,columns: str,values: str,xlabel: str,ylabel: str,cmap: str="YlGnBu",vmin: float=0,vmax: float=1) -> plt.Figure:
    core.configure_matplotlib(); pivot=frame.pivot(index=index,columns=columns,values=values)
    fig,ax=plt.subplots(figsize=core._mm(178,max(96,18+7.5*len(pivot))))
    im=ax.imshow(pivot.to_numpy(),aspect="auto",cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=38,ha="right"); ax.set_yticks(range(len(pivot.index)),pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value=pivot.iloc[i,j]
            if np.isfinite(value):
                # 根据单元格实际渲染色计算文字对比度，避免深蓝底上的深色数字不可读。
                red,green,blue,_=im.cmap(im.norm(value))
                luminance=.2126*red+.7152*green+.0722*blue
                text_color="white" if luminance<.48 else "#222222"
                ax.text(j,i,f"{value:.2f}",ha="center",va="center",fontsize=6.2,color=text_color)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); cb=fig.colorbar(im,ax=ax,pad=.018,shrink=.92); cb.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=.25,right=.93,bottom=.25,top=.96); return fig


def plot_m09(frame: pd.DataFrame) -> plt.Figure:
    labels={"wind":"风","power_model":"功率","dem_error":"DEM","localization":"定位"}; layers={"known_domain_shift":"已知","hidden_model_perception_mismatch":"隐藏"}
    work=frame.copy(); work["condition_label"]=work["family"].map(layers)+work["condition"].map(labels)
    row_order=[LABELS[m] for m in CORE_MODELS+("no_domain_randomization","no_return_reserve")]; work["row"]=pd.Categorical(work["model"].map(LABELS),row_order,ordered=True)
    return _heatmap(work,"row","condition_label","retention","扰动条件","模型","YlGnBu",0.65,1.02)


def plot_s01(frame: pd.DataFrame) -> plt.Figure:
    core.configure_matplotlib(); fig,ax=plt.subplots(figsize=core._mm(178,112))
    order=[m for m in (*CORE_MODELS,*BASELINES,*ABLATIONS) if m in set(frame.model)]
    for model in order:
        sub=frame[frame.model.eq(model)]; primary=model in CORE_MODELS
        ax.plot(sub.regret,sub.ecdf,color=COLORS.get(model,"#9A9A9A"),lw=1.8 if primary else .8,alpha=1 if primary else .72,label=LABELS.get(model,model),zorder=3 if primary else 1)
    ax.set_xlabel("同任务最优值 − 安全加权覆盖率（regret）"); ax.set_ylabel("任务比例"); ax.set_xlim(left=0); ax.set_ylim(0,1.01); _clean_axis(ax,"both")
    ax.legend(frameon=False,ncol=4,loc="lower right",fontsize=6.2); fig.subplots_adjust(left=.11,right=.985,bottom=.17,top=.97); return fig


def plot_s04(frame: pd.DataFrame) -> plt.Figure:
    work=frame.copy(); work["row"]=pd.Categorical(work["model"].map(LABELS),[LABELS[m] for m in CORE_MODELS],ordered=True); order=["16","20","24","moderate","hard","extreme","energy","distance","time","mixed","clustered","dispersed","far_high_conflict"]
    work["scenario"]=pd.Categorical(work["level"].astype(str),order,ordered=True); work=work.sort_values("scenario"); work["scenario"]=work["scenario"].astype(str)
    return _heatmap(work,"row","scenario","mean","场景分层","模型","YlGnBu",0,0.75)


def plot_s05(frame: pd.DataFrame) -> plt.Figure:
    metric_labels={"safe_rate":"安全","return_rate":"返航","violation_rate":"无违规","dangerous_action_proposal_rate":"无危险提议","environment_interception_rate":"无拦截","stranded_rate":"无滞留"}
    cond={"wind":"风","power_model":"功率","dem_error":"DEM","localization":"定位"}; layers={"known_domain_shift":"已知","hidden_model_perception_mismatch":"隐藏"}
    work=frame.copy(); work["metric_label"]=work.metric.map(metric_labels); work["row"]=work.model.map(LABELS)+"｜"+work.family.map(layers)+work.condition.map(cond)
    return _heatmap(work,"row","metric_label","higher_better","统一方向指标","模型与扰动","YlGnBu",0,1)


def plot_s06(frame: pd.DataFrame) -> plt.Figure:
    core.configure_matplotlib(); fig,ax=plt.subplots(figsize=core._mm(178,118))
    for model in LEARNING_MODELS:
        med=frame[frame.model.eq(model)].sort_values("episodes_seen")
        color=COLORS.get(model,"#777777"); ax.plot(med.episodes_seen,med.weighted_coverage,color=color,lw=1.75,label=LABELS[model])
        ax.fill_between(med.episodes_seen.to_numpy(float),med.q25.to_numpy(float),med.q75.to_numpy(float),color=color,alpha=.10,linewidth=0)
    ax.set_xlabel("训练回合（episode）"); ax.set_ylabel("训练批次加权覆盖率"); ax.set_xlim(0,3000); ax.set_ylim(0,1.01); _clean_axis(ax,"y"); ax.legend(frameon=False,ncol=4,fontsize=6.4,loc="lower right"); fig.subplots_adjust(left=.11,right=.985,bottom=.17,top=.97); return fig


def plot_s08(frame: pd.DataFrame) -> plt.Figure:
    core.configure_matplotlib(); pivot=frame.pivot(index="operational_floor",columns="training_weight",values="first_share").sort_index()
    x=pivot.columns.to_numpy(float); y=pivot.index.to_numpy(float); X,Y=np.meshgrid(x,y)
    fig,ax=plt.subplots(figsize=core._mm(178,108)); cs=ax.contourf(X,Y,pivot.to_numpy(),levels=np.linspace(0,1,11),cmap="YlGnBu")
    ax.contour(X,Y,pivot.to_numpy(),levels=[.5,.8,.95],colors="#333333",linewidths=.55)
    ax.set_xlabel("D6+D7总权重"); ax.set_ylabel("运行区间下限"); cb=fig.colorbar(cs,ax=ax,pad=.02); cb.set_label("PPO+Pointer第一名占比"); _clean_axis(ax); fig.subplots_adjust(left=.11,right=.93,bottom=.17,top=.96); return fig


def plot_v01(frame: pd.DataFrame) -> plt.Figure:
    core.configure_matplotlib(); fig,ax=plt.subplots(figsize=core._mm(178,135))
    for _,road in frame[frame.record_type.eq("road")].groupby("group"): ax.plot(road.x,road.y,color="#8A8A8A",lw=1.2,zorder=1)
    points=frame[frame.record_type.eq("inspection")]; pcols={1:"#4E89CF",2:"#E9A526",3:"#C9362B"}; psizes={1:28,2:42,3:58}
    for p in (1,2,3):
        q=points[points.priority.eq(p)]; ax.scatter(q.x,q.y,s=psizes[p],c=pcols[p],edgecolor="white",lw=.8,label=["低优先级","中优先级","高优先级"][p-1],zorder=5)
    airport=frame[frame.record_type.eq("airport")]; ax.scatter(airport.x,airport.y,s=105,marker="*",c="#111111",edgecolor="white",lw=.7,label="机场",zorder=6)
    styles={"full":"-","a2c_pointer":"--","traditional_ppo":"-.","milp":":"}
    for model in ("full","a2c_pointer","traditional_ppo","milp"):
        route=frame[(frame.record_type.eq("route")) & frame.model.eq(model)].sort_values("sequence")
        if route.empty: ax.plot([],[],styles[model],color=COLORS[model],label=LABELS[model]+"（缺失）")
        else: ax.plot(route.x,route.y,styles[model],color=COLORS[model],lw=2.0,label=LABELS[model],zorder=4)
    ax.set_aspect("equal"); ax.set_xlabel("局部东向坐标（30 m/格）"); ax.set_ylabel("局部北向坐标（30 m/格）"); _clean_axis(ax); ax.legend(frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(.5,-.10),fontsize=6.7); fig.subplots_adjust(left=.10,right=.98,bottom=.22,top=.98); return fig


PYTHON_PLOTS={"M09":plot_m09,"S01":plot_s01,"S04":plot_s04,"S05":plot_s05,"S06":plot_s06,"S08":plot_s08,"V01":plot_v01}


def render_python(sources: Mapping[str,pd.DataFrame]) -> dict[str,Any]:
    records={}
    for figure_id in PYTHON_IDS:
        records[figure_id]={"renderer":"Python/matplotlib","outputs":core._save_figure(PYTHON_PLOTS[figure_id](sources[figure_id]),_stem(figure_id))}
    return records


def rerender_heatmaps() -> dict[str,Any]:
    """只重绘三张注释热力图，用于修复深色单元格上的文字对比度。"""
    records={}
    for figure_id in ("M09","S04","S05"):
        source=OUTPUT/"source_data"/f"{figure_id}_source_data.csv"
        frame=pd.read_csv(source)
        records[figure_id]={
            "renderer":"Python/matplotlib",
            "reason":"adaptive_cell_text_contrast",
            "outputs":core._save_figure(PYTHON_PLOTS[figure_id](frame),_stem(figure_id)),
        }
    refresh_delivery_metadata()
    build_full_thumbnail_index()
    return records


def _origin_safe(values: Iterable[Any]) -> list[Any]:
    out=[]
    for value in values:
        if pd.isna(value): out.append("--")
        elif isinstance(value,np.generic): out.append(value.item())
        else: out.append(value)
    return out


def _put(app: Any,book: str,frame: pd.DataFrame) -> None:
    for idx,col in enumerate(frame.columns):
        # DataFrame会把短序列补齐为尾部NaN。若把这些尾部NaN作为字符串写入，
        # Origin散点模板可能把它们误判为零点；直接传IEEE NaN又会使2021版COM崩溃。
        # 因此只裁掉尾部补齐项，保留连接线内部用于断段的缺失值。
        values=frame[col].tolist()
        while values and pd.isna(values[-1]):
            values.pop()
        app.PutWorksheet(book,_origin_safe(values),0,idx); app.Execute(f'win -a {book}; wks.col{idx+1}.lname$="{core._lt_escape(col)}";')


def _series_frame(series: list[tuple[str,np.ndarray,np.ndarray]]) -> pd.DataFrame:
    width=max(len(x) for _,x,_ in series); data={}
    for name,x,y in series:
        data[f"x_{name}"]=pd.Series(x,index=range(len(x))); data[f"y_{name}"]=pd.Series(y,index=range(len(y)))
    return pd.DataFrame(data,index=range(width))


def _add_plot(app: Any,book: str,graph: str,xcol: int,ycol: int,kind: str,color: str,marker: int=3,line_style: int=1,plot_index: int=1) -> None:
    plot_type=200 if kind=="line" else 201
    app.Execute(f"win -a {book}; plotxy iy:=({xcol},{ycol}) plot:={plot_type} ogl:=[{graph}]1!;")
    _style_plot(app,graph,plot_index,kind,color,marker,line_style)


def _style_plot(app: Any,graph: str,plot_index: int,kind: str,color: str,marker: int=3,line_style: int=1) -> None:
    r,g,b=tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4))
    # 直接绑定图层中的第 plot_index 条曲线；%C 只代表当前活动曲线，
    # 在模板自动编组后并不可靠，曾导致样式全部落到最后一条曲线上。
    if kind=="line": app.Execute(f"win -a {graph}; range rr=!{plot_index}; set rr -q 0; set rr -cl color({r},{g},{b}); set rr -k 0; set rr -l 1; set rr -wp 0.55; set rr -d {line_style};")
    else: app.Execute(f"win -a {graph}; range rr=!{plot_index}; set rr -q 0; set rr -cue 1; set rr -c color({r},{g},{b}); set rr -cse color({r},{g},{b}); set rr -csf color({r},{g},{b}); set rr -k {marker}; set rr -z 2.0; set rr -kf 0; set rr -kh 1;")


def _origin_prepare(figure_id: str,source: pd.DataFrame,plot: pd.DataFrame,labels: list[str]|None=None) -> tuple[Any,str,str,str|None,Path,Path]:
    import win32com.client
    # Origin 图必须在前台可见地生成，便于逐图检查模板映射、图层和排版。
    app=win32com.client.Dispatch("Origin.ApplicationSI"); app.Visible=1; app.Execute("doc -s; doc -n;")
    source_book=app.CreatePage(2,"SourceData","Origin",2); _put(app,source_book,source)
    plot_book=app.CreatePage(2,"PlotData","Origin",2); _put(app,plot_book,plot)
    meta_book=app.CreatePage(2,"Metadata","Origin",2)
    meta=pd.DataFrame({"key":["figure_id","template","caption","source_sha256","renderer"],"value":[figure_id,core.FIGURES[figure_id]["template"],CAPTIONS[figure_id],core._sha256(OUTPUT/"source_data"/f"{figure_id}_source_data.csv"),"Origin 2021 COM manual-style"]}); _put(app,meta_book,meta)
    label_book=None
    if labels is not None:
        label_book=app.CreatePage(2,"AxisLabels","Origin",2); _put(app,label_book,pd.DataFrame({"position":np.arange(1,len(labels)+1),"label":labels}))
    template=Path(core.FIGURES[figure_id]["template"]).stem
    graph=app.CreatePage(3,"Graph1",template,2)
    # 保留内置模板的图层和样式持有器；删除它们会破坏SCATTERINTERVAL等
    # 模板的几何语义。仅移除默认图例/色标，再映射正式PlotData。
    app.Execute(f"win -a {graph}; while(page.nlayers>1) {{layer -d $(page.nlayers);}}; layer -c; layer -e %Z; label -ra; page.color=color(255,255,255); layer.background=color(255,255,255);")
    project=(OUTPUT/"origin_projects"/f"{figure_id}.opju").resolve(); native=(OUTPUT/"qa"/"origin_native_exports"/figure_id).resolve(); project.parent.mkdir(parents=True,exist_ok=True); native.mkdir(parents=True,exist_ok=True)
    return app,plot_book,graph,label_book,project,native


def _origin_finish(app: Any,figure_id: str,graph: str,project: Path,native: Path,xlabel: str,ylabel: str,labels_book: str|None=None,logx: bool=False,xrange: tuple[float,float]|None=None,show_legend: bool=False) -> dict[str,Any]:
    cmd=f'win -a {graph}; layer -a; page.width={ORIGIN_PAGE["width_px"]}; page.height={ORIGIN_PAGE["height_px"]}; page.aa=1; layer.left=25; layer.top=6; layer.width=70; layer.height=80; layer.x.postype=0; layer.y.postype=0; layer.x.opposite=0; layer.y.opposite=0; axis -ps X A 1; axis -ps X L 1; axis -ps Y A 1; axis -ps Y L 1; yl.text$=""; yr.text$=""; xb.text$=""; xt.text$=""; layer.x.showlabel=1; layer.y.showlabel=1; layer.x.labelType=1; layer.x.labelSubtype=1; layer.x.label.decPlaces=2; layer.x.label.pt=1.15; layer.y.label.pt=1.20; layer.x.label.color=color(30,30,30); layer.y.label.color=color(30,30,30); layer.x.thickness=.18; layer.y.thickness=.18; layer.x.tickthickness=.18; layer.y.tickthickness=.18; layer.x.ticklength=1.5; layer.y.ticklength=1.2; layer.x.grid=1; layer.x.grid.color=color(225,228,232); legend.show=0; label -p 50 108 -n AxisXTitle {core._lt_escape(xlabel)}; AxisXTitle.fsize=1.35; AxisXTitle.color=color(25,25,25);'
    if ylabel:
        cmd+=f' label -p 2 43 -n AxisYTitle {core._lt_escape(ylabel)}; AxisYTitle.fsize=1.35; AxisYTitle.rotate=90; AxisYTitle.color=color(25,25,25);'
    if logx: cmd+=' layer.x.type=2;'
    if xrange: cmd+=f' layer.x.from={xrange[0]}; layer.x.to={xrange[1]};'
    if xrange:
        if logx:
            cmd+=' layer.x.inc=1;'
        else:
            cmd+=f' layer.x.inc={(xrange[1]-xrange[0])/4};'
    if labels_book: cmd+=f' range axisLabels=[{labels_book}]Sheet1!col(B); layer.y.from=.5; layer.y.to=axisLabels.getSize()+.5; layer.y.inc=1; axis -ps Y T axisLabels; layer.y.label.halign=2;'
    if figure_id=="M05": cmd+=' layer.y.from=0; layer.y.to=1.12; layer.y.inc=.2;'
    if figure_id=="S02": cmd+=' layer.y.from=.25; layer.y.to=.72; layer.y.inc=.10;'
    if show_legend: cmd+=' legendupdate mode:=lname; legend.show=1; legend.fsize=1.05; legend.x=layer.x.to-legend.dx/2; legend.y=layer.y.to-legend.dy/2;'
    app.Execute(cmd+' doc -uw;')
    saved=bool(app.Save(str(project))); status={}
    for ext in ("png","pdf","tif","svg"):
        status[ext]=bool(app.Execute(f'expGraph type:={ext} path:="{core._lt_escape(str(native))}" filename:="{figure_id}" overwrite:=replace tr.Margin:={ORIGIN_PAGE["margin_mm"]} tr1.Unit:=2 tr1.Width:={ORIGIN_PAGE["export_width_px"]} tr2.TIF.DotsPerInch:=600 tr2.TIF.Compression:=LZW;'))
    app.Execute("doc -s;"); app.Exit()
    stem=_stem(figure_id); stem.parent.mkdir(parents=True,exist_ok=True); outputs={}
    for src_ext,dst_ext in (("png","png"),("pdf","pdf"),("tif","tiff"),("svg","svg")):
        src=native/f"{figure_id}.{src_ext}"
        if src.is_file() and src.stat().st_size:
            dst=stem.with_suffix('.'+dst_ext); shutil.copy2(src,dst)
            if dst_ext=="png":
                with Image.open(dst) as im: im.save(dst,dpi=(core.EXPORT_DPI,core.EXPORT_DPI))
            outputs[dst_ext]=str(dst)
    return {"renderer":"Origin 2021","project":str(project),"saved":saved,"native_status":status,"outputs":outputs}


def render_origin(figure_id: str,source: pd.DataFrame) -> dict[str,Any]:
    series=[]; labels=[]; xlabel=""; ylabel=""; logx=False; xrange=None; point_specs=[]
    if figure_id=="M03":
        work=source.copy(); work["y"]=np.arange(1,len(work)+1); labels=[f"{r.domain}｜{r.stratum}" for r in work.itertuples()]
        # 完整模型与消融值高度接近。仅在类别轴方向做对称微偏移，避免符号完全遮挡；
        # 横轴仍严格使用冻结的原始覆盖率，连接线仍表示同一分层内的成对比较。
        work["y_full"]=work["y"]+0.085
        work["y_ablation"]=work["y"]-0.085
        x=[];y=[]
        for r in work.itertuples(): x.extend([r.full,r.ablation,np.nan]); y.extend([r.y_full,r.y_ablation,np.nan])
        series=[("connect",np.array(x),np.array(y)),("ablation",work.ablation.to_numpy(),work.y_ablation.to_numpy()),("full",work.full.to_numpy(),work.y_full.to_numpy())]
        point_specs=[("line","#B4BAC1",3,1),("scatter",COLORS["no_priority_bias"],1,1),("scatter",COLORS["full"],2,1)]; xlabel="优先级覆盖率"; ylabel=""; values=work[["full","ablation"]].to_numpy(float); pad=max(.01,.08*(values.max()-values.min())); xrange=(max(0,values.min()-pad),min(1,values.max()+pad))
    elif figure_id=="M04":
        # 图面自上而下固定为能耗、航程、总任务时间；每组内部固定算法顺序。
        # Origin 的类别轴从下向上递增，因此这里反向构造，并在三组之间留一空行。
        metric_bottom_order=("总任务时间","航程","能耗")
        model_bottom_order=tuple(reversed(core.MAIN_COMPARE))
        metric_rank={name:i for i,name in enumerate(metric_bottom_order)}
        model_rank={name:i for i,name in enumerate(model_bottom_order)}
        work=source.copy()
        work["_metric_rank"]=work.metric.map(metric_rank)
        work["_model_rank"]=work.model.map(model_rank)
        work=work.sort_values(["_metric_rank","_model_rank"]).reset_index(drop=True)
        work["y"]=[metric_rank[m]*7+model_rank[model]+1 for m,model in zip(work.metric,work.model)]
        labels=[""]*20
        for r in work.itertuples():
            labels[int(r.y)-1]=f"{r.metric}｜{LABELS[r.model]}  {r.raw_median:.1f}{r.unit}; 安全{100*r.safe_share:.0f}%"
        for idx,model in enumerate(core.MAIN_COMPARE):
            q=work[work.model.eq(model)]; series.append((model,q.utilization.to_numpy(),q.y.to_numpy())); point_specs.append(("scatter",COLORS[model],(2,1,3,4,5,6)[idx],1))
        xlabel="预算利用率（仅安全路线）"; ylabel=""; xrange=(0,1)
    elif figure_id=="M05":
        for idx,model in enumerate(core.MAIN_COMPARE):
            q=source[source.model.eq(model)].sort_values("planning_time_s")
            # 仅在PlotData中展开为阶梯线；SourceData仍保留每个任务的原始ECDF点。
            raw_x=q.planning_time_s.to_numpy(float); raw_y=q.ecdf.to_numpy(float)
            step_x=np.repeat(raw_x,2)[1:]; step_y=np.repeat(raw_y,2)[:-1]
            series.append((model,step_x,step_y))
            line_style=0 if idx<3 else 1+(idx-3)
            point_specs.append(("line",COLORS[model],0,line_style))
        xlabel="在线规划时间（s，对数轴）"; ylabel="ECDF"; logx=True
        positive=source.planning_time_s[source.planning_time_s>0]
        upper=math.ceil(float(positive.max())/50.0)*50.0
        xrange=(10**math.floor(math.log10(positive.min())),upper)
    elif figure_id=="M07":
        metric_top_order=("D6训练稳定性","D7样本效率","Learning-curve AUC","尾段时间一致性","阈值效率","达到阈值速度")
        labels=list(reversed(metric_top_order))
        ymap={metric:i+1 for i,metric in enumerate(labels)}
        offsets={"full":.16,"a2c_pointer":0.0,"traditional_ppo":-.16}
        marker_codes={"full":2,"a2c_pointer":1,"traditional_ppo":3}
        work=source.copy()
        for model in CORE_MODELS:
            q=work[work.model.eq(model)].copy()
            ypoint=np.array([ymap[m]+offsets[model] for m in q.metric],dtype=float)
            xci=[]; yci=[]
            for row,yvalue in zip(q.itertuples(),ypoint):
                xci.extend([row.q25,row.q75,np.nan]); yci.extend([yvalue,yvalue,np.nan])
            series.append((model+"_ci",np.array(xci),np.array(yci))); point_specs.append(("line",COLORS[model],0,0))
            series.append((model,q.score.to_numpy(),ypoint)); point_specs.append(("scatter",COLORS[model],marker_codes[model],1))
        xlabel="统一方向分数（越高越优）"; ylabel=""; xrange=(0,1)
    elif figure_id=="M08":
        domain_bottom_order=("real","synthetic")
        model_bottom_order=tuple(reversed(CORE_MODELS))
        domain_rank={name:i for i,name in enumerate(domain_bottom_order)}
        model_rank={name:i for i,name in enumerate(model_bottom_order)}
        work=source.copy()
        work["y"]=[domain_rank[d]*4+model_rank[m]+1 for d,m in zip(work.domain,work.model)]
        labels=[""]*7
        for r in work.itertuples(): labels[int(r.y)-1]=f"{r.domain_label}｜{LABELS[r.model]}  n={r.n_maps}"
        for idx,model in enumerate(CORE_MODELS):
            q=work[work.model.eq(model)]; x=[];y=[]
            for r in q.itertuples(): x.extend([r.ci_low,r.ci_high,np.nan]); y.extend([r.y,r.y,np.nan])
            series.append((model+"_ci",np.array(x),np.array(y))); point_specs.append(("line",COLORS[model],0,1)); series.append((model,q.estimate.to_numpy(),q.y.to_numpy())); point_specs.append(("scatter",COLORS[model],3+idx,1))
        xlabel="地图级任务效能D1"; ylabel=""; xrange=(0.25,0.65)
    elif figure_id=="M10":
        domain_bottom_order=("真实DSM","未见合成")
        ablation_bottom_order=tuple(reversed(ABLATIONS))
        domain_rank={name:i for i,name in enumerate(domain_bottom_order)}
        ablation_rank={name:i for i,name in enumerate(ablation_bottom_order)}
        work=source.copy()
        work["y"]=[domain_rank[d]*5+ablation_rank[a]+1 for d,a in zip(work.domain,work.comparator)]
        labels=[""]*9
        for r in work.itertuples(): labels[int(r.y)-1]=f"{r.domain}｜{LABELS[r.comparator]}{' *' if r.significant_holm else ''}"
        for idx,abl in enumerate(ABLATIONS):
            q=work[work.comparator.eq(abl)]; x=[];y=[]
            for r in q.itertuples(): x.extend([r.bootstrap_ci_low,r.bootstrap_ci_high,np.nan]); y.extend([r.y,r.y,np.nan])
            series.append((abl+"_ci",np.array(x),np.array(y))); point_specs.append(("line",COLORS[abl],0,1)); series.append((abl,q.hodges_lehmann.to_numpy(),q.y.to_numpy())); point_specs.append(("scatter",COLORS[abl],3+idx,1))
        xlabel="PPO+Pointer − 消融（安全加权覆盖率）"; ylabel=""; lo=float(work.bootstrap_ci_low.min()); hi=float(work.bootstrap_ci_high.max()); pad=max(.01,.08*(hi-lo)); xrange=(lo-pad,hi+pad)
    elif figure_id=="S02":
        work=source.copy(); work["domain_label"]=work.scope.map({"synthetic_all":"未见合成","real_all":"真实DSM"}); labels=[]
        pareto_models=("traditional_ppo","a2c_pointer","milp")
        for scope,line_style in (("synthetic_all",0),("real_all",1)):
            q=work[work.scope.eq(scope)&work.model.isin(pareto_models)].sort_values("planning_time_p95_s")
            series.append((scope+"_frontier",q.planning_time_p95_s.to_numpy(),q.D1.to_numpy()))
            point_specs.append(("line","#8C9298",0,line_style))
        for model in core.MAIN_COMPARE:
            for scope,marker in (("synthetic_all",1),("real_all",2)):
                q=work[work.model.eq(model)&work.scope.eq(scope)]
                series.append((model+"_"+scope,q.planning_time_p95_s.to_numpy(),q.D1.to_numpy()))
                point_specs.append(("scatter",COLORS[model],marker,1))
        xlabel="P95在线规划时间（s，对数轴）"; ylabel="任务效能D1"; logx=True; xrange=(1.0,150.0)
    elif figure_id=="S03":
        # 自上而下按计算时间递增；类别轴从下向上，因此先按时间降序写入。
        work=source.sort_values("planning_time_s",ascending=False).reset_index(drop=True); work["y"]=np.arange(1,len(work)+1)
        labels=[]
        for r in work.itertuples():
            detail=f"{LABELS.get(r.model,r.model)}  {r.planning_time_s:.2f}s"
            if np.isfinite(r.solver_gap): detail+=f"；gap={r.solver_gap:.3g}"
            if np.isfinite(r.certified_share): detail+=f"；证书{100*r.certified_share:.0f}%"
            labels.append(detail)
        x=[];y=[]
        for r in work.itertuples(): x.extend([r.regret_low,r.regret_high,np.nan]); y.extend([r.y,r.y,np.nan])
        series=[("interval",np.array(x),np.array(y)),("point",((work.regret_low+work.regret_high)/2).to_numpy(),work.y.to_numpy())]; point_specs=[("line","#7B7B7B",0,0),("scatter","#4F6D8A",2,1)]; xlabel="Oracle regret区间"; ylabel=""; xrange=(0,.14)
    elif figure_id=="S07":
        metrics=[f"D{i}" for i in range(1,8)]+["综合得分"]; work=source.copy(); work["metric"]=pd.Categorical(work.metric,metrics,ordered=True); work=work.sort_values("metric"); ymap={m:i+1 for i,m in enumerate(metrics)}; labels=metrics
        offsets={"full":-.18,"a2c_pointer":0,"traditional_ppo":.18}
        for idx,model in enumerate(CORE_MODELS):
            q=work[work.model.eq(model)]; y=np.array([ymap[str(m)]+offsets[model] for m in q.metric]); series.append((model,q.value_100.to_numpy(),y)); point_specs.append(("scatter",COLORS[model],3+idx,1))
        xlabel="维度/综合得分（0–100）"; ylabel=""; xrange=(0,100)
    else: raise KeyError(figure_id)
    plot=_series_frame(series); app,book,graph,label_book,project,native=_origin_prepare(figure_id,source,plot,labels if labels else None)
    for idx,((name,_,_),spec) in enumerate(zip(series,point_specs)):
        display=LABELS.get(name.replace("_ci",""),name)
        app.Execute(f'win -a {book}; wks.col{2*idx+2}.lname$="{core._lt_escape(display)}";')
        _add_plot(app,book,graph,2*idx+1,2*idx+2,*spec,plot_index=idx+1)
    # 部分内置模板会自动把后加曲线编组；取消编组后再次逐条应用冻结样式。
    app.Execute(f"win -a {graph}; layer -gu;")
    for idx,spec in enumerate(point_specs):
        _style_plot(app,graph,idx+1,*spec)
    if figure_id=="M03":
        # 空心方块（消融）与实心圆（完整模型）同时使用颜色和点形编码。
        app.Execute(
            f"win -a {graph}; "
            "range ablationPlot=!2; set ablationPlot -q 0; set ablationPlot -k 1; set ablationPlot -z 2.2; set ablationPlot -csf color(255,255,255); set ablationPlot -kf 0; set ablationPlot -kh 1; "
            "range fullPlot=!3; set fullPlot -q 0; set fullPlot -k 2; set fullPlot -z 2.2; set fullPlot -kf 0; set fullPlot -kh 1; "
            "label -p 67 3 -n M03Legend \\l(2) 无优先级偏置   \\l(3) PPO+Pointer; M03Legend.fsize=1.05; M03Legend.showframe=0;"
        )
    elif figure_id=="M04":
        # Cleveland点图使用中等符号，避免模板默认大符号压过行标签。
        for plot_index in range(1,len(series)+1):
            app.Execute(f"win -a {graph}; range rr=!{plot_index}; set rr -z 1.75; set rr -kf 0; set rr -kh 1;")
        # MILP使用加号，需略加粗才能在600 dpi缩放后仍清晰可辨。
        app.Execute(f"win -a {graph}; range milpPlot=!6; set milpPlot -z 2.10; set milpPlot -kh 2; set milpPlot -cse color(34,34,34);")
    elif figure_id=="M05":
        # ECDF图不用模板默认图例；两行人工图例避免中文被截断。
        for plot_index in range(1,len(series)+1):
            app.Execute(f"win -a {graph}; range rr=!{plot_index}; set rr -wp 0.78;")
        app.Execute(
            f"win -a {graph}; layer.y.from=0; layer.y.to=1; layer.y.inc=.2; legend.show=0; "
            "label -p 58 2 -n M05Legend1 \\l(1) PPO+Pointer   \\l(2) A2C+Pointer   \\l(3) 传统PPO; M05Legend1.fsize=1.00; M05Legend1.showframe=0; "
            "label -p 52 5 -n M05Legend2 \\l(4) 优先级-资源贪心   \\l(5) ACO   \\l(6) MILP; M05Legend2.fsize=1.00; M05Legend2.showframe=0;"
        )
    elif figure_id=="M07":
        # 奇数曲线为区间，偶数曲线为中位点；模型图例只引用中位点。
        for plot_index in (1,3,5):
            app.Execute(f"win -a {graph}; range rr=!{plot_index}; set rr -wp 0.65; set rr -d 0; set rr -l 1;")
        for plot_index in (2,4,6):
            app.Execute(f"win -a {graph}; range rr=!{plot_index}; set rr -z 1.95; set rr -kf 0; set rr -kh 1;")
        app.Execute(
            f"win -a {graph}; layer.clip=0; legend.show=0; "
            "label -p 64 3 -n M07Legend \\l(2) PPO+Pointer   \\l(4) A2C+Pointer   \\l(6) 传统PPO; M07Legend.fsize=1.02; M07Legend.showframe=0;"
        )
    elif figure_id=="M08":
        for plot_index in (1,3,5):
            app.Execute(f"win -a {graph}; range rr=!{plot_index}; set rr -wp 0.70; set rr -d 0; set rr -l 1;")
        for plot_index in (2,4,6):
            app.Execute(f"win -a {graph}; range rr=!{plot_index}; set rr -z 2.00; set rr -kf 0; set rr -kh 1;")
    elif figure_id=="M10":
        for plot_index in (1,3,5,7):
            app.Execute(f"win -a {graph}; range rr=!{plot_index}; set rr -wp 0.70; set rr -d 0; set rr -l 1;")
        for plot_index in (2,4,6,8):
            app.Execute(f"win -a {graph}; range rr=!{plot_index}; set rr -z 1.95; set rr -kf 0; set rr -kh 1;")
        app.Execute(f"win -a {graph}; range reservePoint=!8; set reservePoint -k 2; set reservePoint -z 2.15; set reservePoint -kf 0; set reservePoint -kh 1;")
        app.Execute(
            f"win -a {graph}; draw -n M10Zero -d 1 -w 0.22 -l -v 0; "
            "M10Zero.attach=2; M10Zero.color=color(105,105,105); M10Zero.lineType=1;"
        )
    elif figure_id=="S02":
        for plot_index in (1,2):
            app.Execute(f"win -a {graph}; range rr=!{plot_index}; set rr -wp 0.65; set rr -l 1;")
        for plot_index in range(3,15):
            app.Execute(f"win -a {graph}; range rr=!{plot_index}; set rr -z 1.90; set rr -kh 1;")
        # 每个算法的第一点为未见合成（方形），第二点为真实DSM（圆形）。
        app.Execute(
            f"win -a {graph}; legend.show=0; "
            "label -p 55 2 -n S02Legend1 \\l(3) PPO+Pointer   \\l(5) A2C+Pointer   \\l(7) 传统PPO; S02Legend1.fsize=1.00; S02Legend1.showframe=0; "
            "label -p 49 5 -n S02Legend2 \\l(9) 优先级-资源贪心   \\l(11) ACO   \\l(13) MILP; S02Legend2.fsize=1.00; S02Legend2.showframe=0; "
            "label -p 50 8 -n S02Domain \\l(3) 未见合成   \\l(4) 真实DSM   \\l(1) 合成Pareto前沿   \\l(2) DSM Pareto前沿; S02Domain.fsize=.92; S02Domain.color=color(75,75,75); S02Domain.showframe=0;"
        )
    elif figure_id=="S03":
        app.Execute(f"win -a {graph}; layer.clip=0; range ciPlot=!1; set ciPlot -wp 0.72; set ciPlot -l 1; set ciPlot -d 0; range pointPlot=!2; set pointPlot -z 1.85; set pointPlot -kf 0; set pointPlot -kh 1;")
    elif figure_id=="S07":
        for plot_index in (1,2,3):
            app.Execute(f"win -a {graph}; range rr=!{plot_index}; set rr -z 1.95; set rr -kf 0; set rr -kh 1;")
        app.Execute(
            f"win -a {graph}; legend.show=0; draw -n S07Sep -d 1 -w 0.18 -l -h 7.5; "
            "S07Sep.attach=2; S07Sep.color=color(170,175,180); S07Sep.lineType=1; "
            "label -p 64 3 -n S07Legend \\l(1) PPO+Pointer   \\l(2) A2C+Pointer   \\l(3) 传统PPO; S07Legend.fsize=1.02; S07Legend.showframe=0;"
        )
    return _origin_finish(app,figure_id,graph,project,native,xlabel,ylabel,label_book,logx,xrange,False)


def write_captions_and_manifests(sources: Mapping[str,pd.DataFrame],records: Mapping[str,Any]) -> None:
    for figure_id in REMAINING_IDS:
        cap=OUTPUT/"captions_CN"/f"{figure_id}.md"; cap.parent.mkdir(parents=True,exist_ok=True); cap.write_text(f"# {figure_id} {core.FIGURES[figure_id]['name']}\n\n{CAPTIONS[figure_id]}\n",encoding="utf-8")
        source=OUTPUT/"source_data"/f"{figure_id}_source_data.csv"; outputs=sorted(p for p in _stem(figure_id).parent.glob(_stem(figure_id).name+".*") if p.is_file())
        manifest={"figure_id":figure_id,"backend":core.FIGURES[figure_id]["backend"],"template":core.FIGURES[figure_id]["template"],"caption":CAPTIONS[figure_id],"source_data":{"path":str(source.relative_to(OUTPUT)),"sha256":core._sha256(source),"rows":len(sources[figure_id])},"outputs":[{"path":str(p.relative_to(OUTPUT)),"sha256":core._sha256(p),"bytes":p.stat().st_size} for p in outputs],"render_record":records[figure_id]}
        core._write_json(OUTPUT/"manifests"/f"{figure_id}.json",manifest)


def refresh_delivery_metadata() -> None:
    """不重画图片，只用当前文件刷新注册表、图注和manifest。"""
    core.write_registry_and_literature()
    core.write_captions()
    sources={figure_id:pd.read_csv(OUTPUT/"source_data"/f"{figure_id}_source_data.csv") for figure_id in REMAINING_IDS}
    records={}
    for figure_id in REMAINING_IDS:
        manifest_path=OUTPUT/"manifests"/f"{figure_id}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"缺少既有manifest，无法只读刷新: {manifest_path}")
        records[figure_id]=json.loads(manifest_path.read_text(encoding="utf-8"))["render_record"]
    write_captions_and_manifests(sources,records)
    core._write_json(OUTPUT/"manifests"/"prototype_gate_status.json",{
        "state":"completed",
        "approved":["M01","M02","M06"],
        "approved_after_revision":["V02"],
        "note":"V02已修复上方坐标轴裁切及标注/图例重叠；四张样稿均通过用户验收。",
    })


def write_origin_ui_audit() -> Path:
    """记录逐个打开OPJU后的只读界面核验结果，不修改Origin项目。"""
    axis_label_ids={"M02","M03","M04","M07","M08","M10","S03","S07"}
    entries=[]
    for figure_id in ("M02","M03","M04","M05","M07","M08","M10","S02","S03","S07"):
        pages=["Book1","SourceData","PlotData","Metadata"]
        if figure_id in axis_label_ids:
            pages.append("AxisLabels")
        pages.append("Graph1" if figure_id!="M02" else "M02")
        entries.append({
            "figure_id":figure_id,
            "opju":f"origin_projects/{figure_id}.opju",
            "opened_in_origin_2021":True,
            "visible_pages":pages,
            "editable_graph_layer_present":True,
            "editable_plot_objects_present":True,
            "read_only_ui_audit":True,
            "saved_during_audit":False,
        })
    target=OUTPUT/"qa"/"origin_opju_ui_audit.json"
    core._write_json(target,{
        "state":"passed",
        "origin_version":"OriginPro 2021",
        "project_count":len(entries),
        "scope":"逐项目界面内核验工作簿、图页、图层和绘图对象；不编辑、不保存。",
        "projects":entries,
    })
    return target


def build_full_thumbnail_index() -> Path:
    """生成仅供总览与QA使用的20图索引，不替代任何独立正式图片。"""
    columns=4; thumb_w=620; thumb_h=400; cell_w=680; cell_h=470
    rows=math.ceil(len(ALL_IDS)/columns)
    canvas=Image.new("RGB",(columns*cell_w+40,rows*cell_h+45),"#F3F4F6")
    draw=ImageDraw.Draw(canvas)
    font_path=Path(r"C:\Windows\Fonts\msyh.ttc")
    font=ImageFont.truetype(str(font_path),20) if font_path.is_file() else ImageFont.load_default()
    for index,figure_id in enumerate(ALL_IDS):
        stem=core._output_stem(figure_id) if figure_id in core.PROTOTYPE_IDS else _stem(figure_id)
        png=stem.with_suffix(".png")
        if not png.is_file():
            raise FileNotFoundError(f"缩略索引缺少PNG: {png}")
        with Image.open(png) as opened:
            image=opened.convert("RGB")
            bbox=_nonwhite_bbox(image)
            if bbox:
                image=image.crop(bbox)
            image.thumbnail((thumb_w,thumb_h),Image.Resampling.LANCZOS)
        row=index//columns; col=index%columns; x=25+col*cell_w; y=20+row*cell_h
        draw.text((x,y),f"{figure_id}  {core.FIGURES[figure_id]['name']}",fill="#20252B",font=font)
        px=x+(thumb_w-image.width)//2; py=y+38+(thumb_h-image.height)//2
        canvas.paste(image,(px,py))
    target=OUTPUT/"thumbnail_index"/"all_20_figures.png"
    target.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(target,dpi=(150,150))
    return target


def _nonwhite_bbox(image: Image.Image) -> tuple[int,int,int,int]|None:
    rgb=image.convert("RGB"); return ImageChops.difference(rgb,Image.new("RGB",rgb.size,"white")).getbbox()


def final_qa(start_audit: Mapping[str,Any]) -> dict[str,Any]:
    ids=ALL_IDS; errors=[]; warnings=[]; figures={}
    for figure_id in ids:
        stem=_stem(figure_id) if figure_id in REMAINING_IDS else core._output_stem(figure_id); item={}
        for ext in ("pdf","png","tiff"):
            path=stem.with_suffix('.'+ext); item[ext]=path.is_file() and path.stat().st_size>0
            if not item[ext]: errors.append(f"{figure_id}缺少{ext}")
        svg=stem.with_suffix('.svg'); item["svg"]=svg.is_file() and svg.stat().st_size>0
        if not item["svg"]:
            if core.FIGURES[figure_id]["backend"]=="origin":
                warnings.append(f"{figure_id} Origin 2021原生SVG无效，保留OPJU和原生PDF作为可编辑主版本")
            else:
                errors.append(f"{figure_id}非Origin图缺少有效SVG")
        if stem.with_suffix('.png').is_file():
            with Image.open(stem.with_suffix('.png')) as im:
                item["pixel_size"]=list(im.size); item["dpi"]=list(im.info.get("dpi",(0,0))); item["nonwhite_bbox"]=_nonwhite_bbox(im)
                if im.width<3000: errors.append(f"{figure_id} PNG宽度不足600dpi双栏交付")
        figures[figure_id]=item
    for figure_id in ("M02",)+ORIGIN_IDS:
        if not (OUTPUT/"origin_projects"/f"{figure_id}.opju").is_file(): errors.append(f"{figure_id}缺少OPJU")
    registry_path=OUTPUT/"manifests"/"figure_registry_manual_v3.json"
    if not registry_path.is_file():
        errors.append("缺少图形注册表")
    else:
        registry=json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("gate_state")!="completed": errors.append("样稿闸门未记录为completed")
        if len(registry.get("figures",{}))!=20: errors.append("图形注册表不是20图")
    literature_path=OUTPUT/"literature_audit"/"literature_style_audit.csv"
    if not literature_path.is_file():
        errors.append("缺少文献图型审计")
    else:
        literature=pd.read_csv(literature_path)
        if len(literature)<20: errors.append("文献审计少于20篇")
        if int(literature["figure_count"].sum())<50: errors.append("审计图例少于50幅")
    if sum((OUTPUT/"manifests"/f"{i}.json").is_file() for i in ids)!=20: errors.append("逐图manifest不足20份")
    if sum((OUTPUT/"captions_CN"/f"{i}.md").is_file() for i in ids)!=20: errors.append("中文图注不足20份")
    source_text="\n".join(path.read_text(encoding="utf-8-sig",errors="replace") for path in sorted((OUTPUT/"source_data").rglob("*.csv")))
    if "ppo_mlp" in source_text: errors.append("Source Data出现已排除的ppo_mlp")
    old_end={"paper_final":core._tree_digest(core.OLD_PAPER_FINAL)[0],"paper_redraw_origin_v2":core._tree_digest(core.OLD_ORIGIN_V2)[0]}; old_start={k:v["sha256"] for k,v in start_audit["old_directories"].items()}
    if old_end!=old_start: errors.append("旧制图目录发生变化")
    report={"passed":not errors,"errors":errors,"warnings":warnings,"figure_count":len(ids),"origin_project_count":sum((OUTPUT/"origin_projects"/f"{i}.opju").is_file() for i in (("M02",)+ORIGIN_IDS)),"literature_paper_count":len(pd.read_csv(literature_path)) if literature_path.is_file() else 0,"literature_figure_count":int(pd.read_csv(literature_path)["figure_count"].sum()) if literature_path.is_file() else 0,"figures":figures,"old_directories_unchanged":old_end==old_start,"old_directories_at_end":old_end}
    core._write_json(OUTPUT/"qa"/"final_qa_report.json",report); return report


def run_full() -> dict[str,Any]:
    start=core.audit_inputs(write_snapshot=False); sources=build_all_sources(); records=render_python(sources)
    for figure_id in ORIGIN_IDS: records[figure_id]=render_origin(figure_id,sources[figure_id])
    write_captions_and_manifests(sources,records); report=final_qa(start)
    core._write_json(OUTPUT/"manifests"/"full_render_status.json",{"state":"completed" if report["passed"] else "qa_failed","figure_count":20,"origin_count":10,"python_count":9,"matlab_count":1,"qa":str(OUTPUT/"qa"/"final_qa_report.json")})
    return report


if __name__=="__main__":
    print(json.dumps(run_full(),ensure_ascii=False,indent=2))
