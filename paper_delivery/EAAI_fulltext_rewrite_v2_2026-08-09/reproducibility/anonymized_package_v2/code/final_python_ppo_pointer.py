#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""面向山区固定巡检点的 PPO + Pointer Network 路径规划内核。

本模块解决的是“资源约束定向越野”问题：无人机在电量、总航程和
任务时间三类硬约束下选择部分巡检点，并且必须返回起飞点。Pointer
Network 负责输出下一个合法动作的策略分布，PPO 负责训练策略。
论文实验还提供参数量可比的共享节点MLP、A2C更新及四项受控消融；
默认配置仍保持完整PPO+Pointer行为。

设计边界：
1. 巡检点位于公路段上，但无人机不需要沿道路飞行；
2. 安全约束由布尔动作掩码保证，奖励只比较合法路线的任务收益；
3. 训练时按策略概率采样，部署时默认使用确定性策略解码（argmax）；
4. 能耗模型是依据 DJI Matrice 3D 公开规格校准的工程代理，并非飞行认证模型。
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import random
import statistics
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

# CUDA矩阵乘法确定性所需；必须在首次CUDA上下文建立前设置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import scipy.io
except ImportError:  # pragma: no cover - 由命令行入口给出清晰错误
    scipy = None


SCHEMA_VERSION = 2
NODE_FEATURE_DIM = 15
UAV_FEATURE_DIM = 14
EPS = 1e-8

# v3.2 传统 PPO 的固定槽定义。16/20 点任务在网络内部补零到 24 点，
# 返航点始终放在最后一个固定槽；外部环境仍沿用原来的变长动作编号。
TRADITIONAL_PPO_MAX_NODES = 24
TRADITIONAL_PPO_FIXED_SLOTS = TRADITIONAL_PPO_MAX_NODES + 1
TRADITIONAL_PPO_HIDDEN_DIM = 256

# MILP 双界在数学上可相等，但二进制浮点序列化可能产生约 1e-16 的反序；
# 仅在该数值噪声范围内规范化为保守有序区间，较大的反序仍视为数据错误。
ORACLE_BOUND_ORDER_TOLERANCE = 1e-12

# 冻结验证实例只描述同一16点任务的环境条件；巡检点布局和优先级由场景哈希另行锁定。
FROZEN_DOMAIN_FIELDS = (
    "initial_soc",
    "distance_budget_scale",
    "time_budget_scale",
    "wind_scale",
    "wind_rotation_deg",
    "wind_vertical_bias_mps",
)


@dataclass(frozen=True)
class ExperimentVariant:
    """论文实验中的一个受控学习变体，避免把消融项任意组合。"""

    name: str
    policy_architecture: str
    training_algorithm: str
    domain_randomization: bool
    resource_shaping: bool
    return_reserve_mask: bool
    simulation_only: bool = False
    lambda_priority_override: Optional[float] = None


# 每个名字只改变论文中要验证的一个因素；full保持v2原始行为。
EXPERIMENT_VARIANTS: Dict[str, ExperimentVariant] = {
    "full": ExperimentVariant("full", "pointer", "ppo", True, True, True),
    "traditional_ppo": ExperimentVariant(
        "traditional_ppo", "flat_mlp_24", "ppo", True, True, True
    ),
    # 仅保留旧检查点兼容读取；v3.2 正式协议不会把该变体列入活跃模型。
    "ppo_mlp": ExperimentVariant(
        "ppo_mlp", "shared_node_mlp", "ppo", True, True, True
    ),
    "a2c_pointer": ExperimentVariant(
        "a2c_pointer", "pointer", "a2c", True, True, True
    ),
    "no_priority_bias": ExperimentVariant(
        "no_priority_bias", "pointer", "ppo", True, True, True,
        lambda_priority_override=0.0,
    ),
    "no_domain_randomization": ExperimentVariant(
        "no_domain_randomization", "pointer", "ppo", False, True, True
    ),
    "no_resource_shaping": ExperimentVariant(
        "no_resource_shaping", "pointer", "ppo", True, False, True
    ),
    "no_return_reserve": ExperimentVariant(
        "no_return_reserve", "pointer", "ppo", True, True, False,
        simulation_only=True,
    ),
}


def get_experiment_variant(name: str) -> ExperimentVariant:
    """按稳定ID取得实验变体；未知名字直接失败，防止实验标签写错。"""

    key = str(name).strip()
    try:
        return EXPERIMENT_VARIANTS[key]
    except KeyError as exc:
        choices = ", ".join(EXPERIMENT_VARIANTS)
        raise ValueError(f"未知experiment_variant={key!r}；可选值：{choices}") from exc

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# 保留模块级 device/torch，兼容现有项目的调用方式。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DJI Dock 2 / Matrice 3D 官方公开规格。
MATRICE_3D_SPECS: Dict[str, Any] = {
    "model": "DJI Matrice 3D",
    "weight_kg": 1.410,
    "max_takeoff_weight_kg": 1.610,
    "battery_model": "BPX220-7811-14.76",
    "battery_capacity_mah": 7811.0,
    "battery_voltage_v": 14.76,
    "battery_energy_wh": 115.2,
    "normal_max_horizontal_speed_mps": 15.0,
    "reference_cruise_speed_mps": 13.0,  # 官方 46.8 km/h 续航测试速度
    "normal_max_ascent_speed_mps": 6.0,
    "normal_max_descent_speed_mps": 6.0,
    "max_wind_resistance_mps": 12.0,
    "takeoff_landing_wind_resistance_mps": 8.0,
    "max_flight_time_min": 50.0,
    "max_hover_time_min": 40.0,
    "max_takeoff_altitude_m": 4000.0,
    "max_operating_radius_m": 10_000.0,
    "max_flight_distance_m": 43_000.0,
    # 以下功率由电池能量和官方续航时间反推，是可替换的工程代理参数。
    "hover_power_w": 172.8,
    "cruise_power_w": 138.24,
    "climb_power_w": 216.0,
    "descent_power_w": 138.24,
}


DEFAULT_CONFIG: Dict[str, Any] = {
    # 论文实验变体；其余派生字段由注册表锁定，不允许任意拼装消融。
    "experiment_variant": "full",
    "policy_architecture": "pointer",
    "training_algorithm": "ppo",
    "domain_randomization": True,
    "resource_shaping": True,
    "return_reserve_mask": True,
    "simulation_only": False,
    # legacy_v2保留旧实验复现；multimap_v3_1使用覆盖主导的层级奖励。
    "reward_schema": "legacy_v2",
    # 任务约束
    "return_to_start": True,
    "battery_capacity": MATRICE_3D_SPECS["battery_energy_wh"],
    "battery_reserve_ratio": 0.25,
    "initial_soc": 1.0,
    "max_route_distance": 8000.0,
    "max_mission_time_s": 2400.0,
    "inspection_service_time_s": 20.0,
    "coordinate_scale_m_per_unit": 1.0,
    "point_z_mode": "terrain",
    "terrain_clearance_m": 18.0,
    "terrain_sample_interval_m": 10.0,
    # 飞行和能耗代理模型
    "cruise_speed_mps": MATRICE_3D_SPECS["reference_cruise_speed_mps"],
    "max_horizontal_speed": MATRICE_3D_SPECS["normal_max_horizontal_speed_mps"],
    "max_ascent_speed": MATRICE_3D_SPECS["normal_max_ascent_speed_mps"],
    "max_descent_speed": MATRICE_3D_SPECS["normal_max_descent_speed_mps"],
    "max_wind_resistance": MATRICE_3D_SPECS["max_wind_resistance_mps"],
    "takeoff_landing_wind_resistance": MATRICE_3D_SPECS[
        "takeoff_landing_wind_resistance_mps"
    ],
    "max_takeoff_altitude_m": MATRICE_3D_SPECS["max_takeoff_altitude_m"],
    "min_ground_speed_mps": 1.0,
    "hover_power_w": MATRICE_3D_SPECS["hover_power_w"],
    "cruise_power_w": MATRICE_3D_SPECS["cruise_power_w"],
    "climb_power_w": MATRICE_3D_SPECS["climb_power_w"],
    "descent_power_w": MATRICE_3D_SPECS["descent_power_w"],
    "resource_safety_factor": 1.10,
    # 仅no_return_reserve仿真消融使用；真实环境不会执行该候选航段。
    "simulation_violation_penalty": -1.0,
    # 网络结构
    "d_model": 128,
    "n_heads": 4,
    "lambda_priority": 0.5,
    # PPO训练
    "max_episodes": 600,
    "episodes_per_update": 16,
    "ppo_epochs": 5,
    "minibatch_size": 128,
    # 真实16点小批次下3e-4首步KL容易显著越过0.02；1e-4更符合当前任务尺度。
    "lr": 1e-4,
    "clip_ratio": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "value_coef": 0.5,
    "entropy_coef_start": 0.02,
    "entropy_coef_end": 0.002,
    "target_kl": 0.02,
    "max_grad_norm": 1.0,
    "seed": 42,
    # 固定地图上的域随机化
    "initial_soc_min": 0.80,
    "initial_soc_max": 1.00,
    "distance_budget_scale_min": 0.85,
    "distance_budget_scale_max": 1.00,
    "time_budget_scale_min": 0.85,
    "time_budget_scale_max": 1.00,
    "wind_scale_min": 0.80,
    "wind_scale_max": 1.20,
    "wind_rotation_deg": 15.0,
    "wind_vertical_bias_mps": 1.0,
    # 训练内验证和检查点
    "validation_scenarios": 8,
    "validation_interval_updates": 10,
    "validation_mode": "legacy_seeded",
    "validation_instances": None,
    "validation_instances_hash": "",
    # 困难约束训练池只保存身份和计数；完整场景由外部冻结manifest重新提供。
    "training_mode": "legacy_seeded",
    "training_instances_hash": "",
    "training_instance_count": 0,
    "training_node_counts": [],
    "difficulty_protocol_hash": "",
    "monitor_episodes": [],
    "persist_monitor_checkpoints": False,
    "checkpoint_dir": None,
    # 归一化奖励；硬安全不放进奖励。
    "reward_weights": {
        "priority": 1.0,
        "coverage": 0.2,
        "energy": 0.15,
        "distance": 0.10,
        "time": 0.10,
    },
}


class ConstraintViolationError(RuntimeError):
    """环境硬约束不变量被破坏。"""


def setup_logging(
    log_dir: Optional[Union[os.PathLike, str]] = None,
) -> logging.Logger:
    """仅在显式运行训练/命令行时配置日志，避免导入模块就创建文件。"""

    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(logging.INFO)
    module_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    module_logger.addHandler(stream)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        filename = log_path / f"ppo_training_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        module_logger.addHandler(file_handler)

    return module_logger


def _deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并配置，保留未覆盖的默认字段。"""

    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            nested = dict(base[key])
            base[key] = _deep_update(nested, value)
        else:
            base[key] = value
    return base


def resolve_config(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """补齐并验证配置；所有关键魔法数字集中在 DEFAULT_CONFIG。"""

    raw = dict(cfg or {})
    merged = _deep_update(copy.deepcopy(DEFAULT_CONFIG), raw)

    variant = get_experiment_variant(
        str(raw.get("experiment_variant", DEFAULT_CONFIG["experiment_variant"]))
    )
    locked_fields: Dict[str, Any] = {
        "policy_architecture": variant.policy_architecture,
        "training_algorithm": variant.training_algorithm,
        "domain_randomization": variant.domain_randomization,
        "resource_shaping": variant.resource_shaping,
        "return_reserve_mask": variant.return_reserve_mask,
        "simulation_only": variant.simulation_only,
    }
    for field_name, expected in locked_fields.items():
        if field_name in raw and raw[field_name] != expected:
            raise ValueError(
                f"变体 {variant.name!r} 已锁定 {field_name}={expected!r}，"
                f"不能改为 {raw[field_name]!r}。"
            )
        merged[field_name] = expected
    merged["experiment_variant"] = variant.name
    if variant.lambda_priority_override is not None:
        expected_priority = float(variant.lambda_priority_override)
        if "lambda_priority" in raw and not math.isclose(
            float(raw["lambda_priority"]), expected_priority, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"变体 {variant.name!r} 要求lambda_priority={expected_priority}。"
            )
        merged["lambda_priority"] = expected_priority

    # 兼容旧配置中的单一熵系数。
    if "entropy_coef" in raw and "entropy_coef_start" not in raw:
        merged["entropy_coef_start"] = float(raw["entropy_coef"])
        merged["entropy_coef_end"] = min(
            float(raw["entropy_coef"]), float(DEFAULT_CONFIG["entropy_coef_end"])
        )

    if not bool(merged["return_to_start"]):
        raise ValueError("v2任务定义要求 return_to_start=True；不允许在巡检点结束任务。")

    positive_fields = (
        "battery_capacity",
        "max_route_distance",
        "max_mission_time_s",
        "coordinate_scale_m_per_unit",
        "terrain_clearance_m",
        "terrain_sample_interval_m",
        "cruise_speed_mps",
        "max_horizontal_speed",
        "max_ascent_speed",
        "max_descent_speed",
        "max_wind_resistance",
        "takeoff_landing_wind_resistance",
        "max_takeoff_altitude_m",
        "min_ground_speed_mps",
        "hover_power_w",
        "cruise_power_w",
        "climb_power_w",
        "descent_power_w",
        "resource_safety_factor",
        "d_model",
        "n_heads",
        "max_episodes",
        "episodes_per_update",
        "ppo_epochs",
        "minibatch_size",
        "lr",
        "target_kl",
        "max_grad_norm",
        "validation_scenarios",
        "validation_interval_updates",
    )
    finite_fields = set(positive_fields) | {
        "battery_reserve_ratio",
        "initial_soc",
        "inspection_service_time_s",
        "lambda_priority",
        "clip_ratio",
        "gamma",
        "gae_lambda",
        "value_coef",
        "entropy_coef_start",
        "entropy_coef_end",
        "initial_soc_min",
        "initial_soc_max",
        "distance_budget_scale_min",
        "distance_budget_scale_max",
        "time_budget_scale_min",
        "time_budget_scale_max",
        "wind_scale_min",
        "wind_scale_max",
        "wind_rotation_deg",
        "wind_vertical_bias_mps",
        "simulation_violation_penalty",
    }
    for field_name in finite_fields:
        try:
            is_finite = math.isfinite(float(merged[field_name]))
        except (TypeError, ValueError, KeyError):
            is_finite = False
        if not is_finite:
            raise ValueError(f"配置 {field_name} 必须是有限数值。")
    for field_name in positive_fields:
        if float(merged[field_name]) <= 0:
            raise ValueError(f"配置 {field_name} 必须大于0，当前为 {merged[field_name]!r}")

    if not 0.0 <= float(merged["battery_reserve_ratio"]) < 1.0:
        raise ValueError("battery_reserve_ratio 必须位于 [0, 1)。")
    if not 0.0 < float(merged["initial_soc"]) <= 1.0:
        raise ValueError("initial_soc 必须位于 (0, 1]。")
    if int(merged["d_model"]) % int(merged["n_heads"]) != 0:
        raise ValueError("d_model 必须能被 n_heads 整除。")
    if not 0.0 <= float(merged["gamma"]) <= 1.0:
        raise ValueError("gamma 必须位于 [0, 1]。")
    if not 0.0 <= float(merged["gae_lambda"]) <= 1.0:
        raise ValueError("gae_lambda 必须位于 [0, 1]。")
    if float(merged["clip_ratio"]) <= 0.0:
        raise ValueError("clip_ratio 必须大于0。")
    for field_name in (
        "inspection_service_time_s",
        "lambda_priority",
        "value_coef",
        "entropy_coef_start",
        "entropy_coef_end",
        "wind_rotation_deg",
        "wind_vertical_bias_mps",
    ):
        if float(merged[field_name]) < 0.0:
            raise ValueError(f"配置 {field_name} 不能为负数。")
    if float(merged["entropy_coef_end"]) > float(merged["entropy_coef_start"]):
        raise ValueError("entropy_coef_end 不能大于 entropy_coef_start。")
    if float(merged["simulation_violation_penalty"]) > 0.0:
        raise ValueError("simulation_violation_penalty 必须小于或等于0。")
    if str(merged["point_z_mode"]) not in {"terrain", "flight_altitude"}:
        raise ValueError("point_z_mode 只能是 'terrain' 或 'flight_altitude'。")
    if str(merged["reward_schema"]) not in {"legacy_v2", "multimap_v3_1"}:
        raise ValueError("reward_schema 只能是 legacy_v2 或 multimap_v3_1。")

    reward_weights = merged.get("reward_weights")
    if not isinstance(reward_weights, Mapping):
        raise ValueError("reward_weights 必须是包含五个奖励权重的映射。")
    for reward_name in ("priority", "coverage", "energy", "distance", "time"):
        try:
            reward_value = float(reward_weights[reward_name])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"reward_weights 缺少有效的 {reward_name} 权重。")
        if not math.isfinite(reward_value) or reward_value < 0.0:
            raise ValueError(f"reward_weights.{reward_name} 必须是有限非负数。")

    for low_key, high_key in (
        ("initial_soc_min", "initial_soc_max"),
        ("distance_budget_scale_min", "distance_budget_scale_max"),
        ("time_budget_scale_min", "time_budget_scale_max"),
        ("wind_scale_min", "wind_scale_max"),
    ):
        if float(merged[low_key]) > float(merged[high_key]):
            raise ValueError(f"{low_key} 不能大于 {high_key}。")
    if not (
        float(merged["battery_reserve_ratio"])
        < float(merged["initial_soc_min"])
        <= float(merged["initial_soc_max"])
        <= 1.0
    ):
        raise ValueError("域随机化SOC范围必须高于安全预留比例且不超过1。")
    for scale_key in (
        "distance_budget_scale_min",
        "distance_budget_scale_max",
        "time_budget_scale_min",
        "time_budget_scale_max",
        "wind_scale_min",
        "wind_scale_max",
    ):
        if float(merged[scale_key]) <= 0.0:
            raise ValueError(f"{scale_key} 必须大于0。")

    return merged


def set_global_seed(seed: int) -> None:
    """设置随机种子，并启用PyTorch确定性执行以支持实验复现。"""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _resolve_training_device(
    target_device: Optional[Union[str, torch.device]],
) -> torch.device:
    """解析训练设备；显式请求不可用CUDA时直接失败。"""

    if target_device is None or str(target_device).lower() == "auto":
        resolved = device
    else:
        resolved = torch.device(target_device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用CUDA训练，但当前PyTorch环境检测不到可用CUDA设备。")
    return resolved


def _restore_rng_state(rng_state: Mapping[str, Any]) -> np.random.Generator:
    """恢复全局RNG和域随机化Generator，供断点续训精确接续。"""

    required = {"python", "numpy_global", "torch", "training_generator"}
    missing = sorted(required.difference(rng_state))
    if missing:
        raise ValueError(f"恢复检查点缺少随机状态字段：{', '.join(missing)}。")

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy_global"])
    torch_state = rng_state["torch"]
    if isinstance(torch_state, torch.Tensor):
        torch_state = torch_state.detach().cpu()
    torch.set_rng_state(torch_state)

    cuda_states = rng_state.get("torch_cuda") or []
    if torch.cuda.is_available() and cuda_states:
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "检查点CUDA RNG数量与当前可见CUDA设备数量不一致，"
                "无法保证精确恢复。"
            )
        torch.cuda.set_rng_state_all(
            [
                state.detach().cpu() if isinstance(state, torch.Tensor) else state
                for state in cuda_states
            ]
        )

    generator = np.random.default_rng()
    generator_state = copy.deepcopy(rng_state["training_generator"])
    if generator_state is None:
        raise ValueError("恢复检查点没有域随机化Generator状态。")
    generator.bit_generator.state = generator_state
    return generator


def _unwrap_mat(value: Any) -> Any:
    """解开 scipy.io.loadmat 常见的单元素对象/结构数组。"""

    current = value
    for _ in range(12):
        if isinstance(current, np.ndarray) and current.size == 1:
            try:
                current = current.reshape(-1)[0]
                continue
            except Exception:
                break
        break
    return current


def _mat_field(container: Any, name: str, default: Any = None) -> Any:
    value = _unwrap_mat(container)
    try:
        if isinstance(value, Mapping):
            return value.get(name, default)
        if isinstance(value, np.void) and value.dtype.names and name in value.dtype.names:
            return value[name]
        if isinstance(value, np.ndarray) and value.dtype.names and name in value.dtype.names:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    except Exception:
        return default
    return default


def _to_scalar(value: Any, default: Any, cast: Any = float) -> Any:
    try:
        item = _unwrap_mat(value)
        if isinstance(item, np.ndarray):
            if item.size == 0:
                return default
            item = item.reshape(-1)[0]
        if hasattr(item, "item"):
            item = item.item()
        return cast(item)
    except Exception:
        return default


def _to_text(value: Any, default: str) -> str:
    """将MATLAB字符数组、NumPy字符串或bytes可靠转换为Python文本。"""

    try:
        item = _unwrap_mat(value)
        if isinstance(item, np.ndarray):
            if item.size == 0:
                return default
            if item.dtype.kind in {"U", "S"}:
                flattened = item.reshape(-1).tolist()
                return "".join(
                    part.decode("utf-8") if isinstance(part, bytes) else str(part)
                    for part in flattened
                ).strip()
            item = item.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode("utf-8").strip()
        if hasattr(item, "item"):
            item = item.item()
        return str(item).strip()
    except Exception:
        return default


def _to_array(value: Any, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if value is None:
        return default
    try:
        arr = np.asarray(_unwrap_mat(value), dtype=np.float32)
        if arr.size == 0:
            return default
        return arr
    except Exception:
        return default


def _first_array_field(container: Any, names: Iterable[str]) -> Optional[np.ndarray]:
    for name in names:
        arr = _to_array(_mat_field(container, name, None))
        if arr is not None:
            return arr
    return None


def _normalize_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"inspection_points 必须是二维数组，当前形状为 {arr.shape}")
    if arr.shape[1] not in (2, 3) and arr.shape[0] in (2, 3):
        arr = arr.T
    if arr.shape[1] not in (2, 3):
        raise ValueError(f"inspection_points 必须为 [N,2] 或 [N,3]，当前为 {arr.shape}")
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.full((arr.shape[0],), np.nan, dtype=np.float32)])
    if not np.all(np.isfinite(arr[:, :2])):
        raise ValueError("巡检点 x/y 坐标包含 NaN 或 Inf。")
    return arr.astype(np.float32, copy=False)


def extract_input(mat_data: Mapping[str, Any]) -> Tuple[
    List[float], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]
]:
    """解析旧版 `.mat` 六元组接口，并读取 v2 可选字段。"""

    if "input_data" not in mat_data:
        raise KeyError("输入 .mat 缺少 input_data。")
    input_data = _unwrap_mat(mat_data["input_data"])

    start_arr = _to_array(_mat_field(input_data, "start_pos"))
    points_arr = _to_array(_mat_field(input_data, "inspection_points"))
    terrain_arr = _to_array(_mat_field(input_data, "terrain_data"))
    if start_arr is None or points_arr is None or terrain_arr is None:
        raise ValueError("input_data 必须包含 start_pos、inspection_points 和 terrain_data。")

    start = np.asarray(start_arr, dtype=np.float32).reshape(-1)
    if start.size < 2:
        raise ValueError("start_pos 至少需要 x/y 两个分量。")
    start = start[:3]
    if not np.all(np.isfinite(start)):
        raise ValueError("start_pos 包含 NaN 或 Inf。")

    points = _normalize_points(points_arr)
    priorities = _to_array(_mat_field(input_data, "inspection_priorities"))
    if priorities is None:
        priorities = np.full((points.shape[0],), 2.0, dtype=np.float32)
    priorities = np.asarray(priorities, dtype=np.float32).reshape(-1)
    if priorities.size != points.shape[0]:
        raise ValueError("inspection_priorities 长度必须与巡检点数量一致。")

    terrain = np.asarray(terrain_arr, dtype=np.float32)
    if terrain.ndim != 2:
        raise ValueError("terrain_data 必须是二维 DEM。")

    wind_struct = _mat_field(input_data, "wind_data", {})
    wind_data: Dict[str, Any] = {
        "speed": _to_scalar(_mat_field(wind_struct, "speed", 0.0), 0.0, float),
        "direction": _to_scalar(_mat_field(wind_struct, "direction", 0.0), 0.0, float),
        "vertical_speed": _to_scalar(
            _mat_field(wind_struct, "vertical_speed", 0.0), 0.0, float
        ),
    }
    uniform_vector = _first_array_field(wind_struct, ("uniform_vector", "vector"))
    positions = _first_array_field(
        wind_struct, ("positions", "sample_positions", "wind_positions")
    )
    vectors = _first_array_field(wind_struct, ("vectors", "wind_vectors"))
    if uniform_vector is not None:
        wind_data["uniform_vector"] = uniform_vector.reshape(-1)[:3]
    if positions is not None and vectors is not None:
        wind_data["positions"] = positions
        wind_data["vectors"] = vectors

    cfg_struct = _mat_field(input_data, "config", {})
    raw_cfg: Dict[str, Any] = {}
    scalar_fields = {
        "max_episodes": int,
        "episodes_per_update": int,
        "ppo_epochs": int,
        "minibatch_size": int,
        "lr": float,
        "clip_ratio": float,
        "gamma": float,
        "gae_lambda": float,
        "value_coef": float,
        "entropy_coef": float,
        "entropy_coef_start": float,
        "entropy_coef_end": float,
        "target_kl": float,
        "max_grad_norm": float,
        "seed": int,
        "return_to_start": bool,
        "battery_capacity": float,
        "battery_reserve_ratio": float,
        "initial_soc": float,
        "max_route_distance": float,
        "max_mission_time_s": float,
        "inspection_service_time_s": float,
        "coordinate_scale_m_per_unit": float,
        "terrain_clearance_m": float,
        "terrain_sample_interval_m": float,
        "cruise_speed_mps": float,
        "max_horizontal_speed": float,
        "max_ascent_speed": float,
        "max_descent_speed": float,
        "max_wind_resistance": float,
        "takeoff_landing_wind_resistance": float,
        "resource_safety_factor": float,
        "simulation_violation_penalty": float,
        "hover_power_w": float,
        "cruise_power_w": float,
        "climb_power_w": float,
        "descent_power_w": float,
        "max_takeoff_altitude_m": float,
        "min_ground_speed_mps": float,
        "d_model": int,
        "n_heads": int,
        "lambda_priority": float,
        "validation_scenarios": int,
        "validation_interval_updates": int,
        "initial_soc_min": float,
        "initial_soc_max": float,
        "distance_budget_scale_min": float,
        "distance_budget_scale_max": float,
        "time_budget_scale_min": float,
        "time_budget_scale_max": float,
        "wind_scale_min": float,
        "wind_scale_max": float,
        "wind_rotation_deg": float,
        "wind_vertical_bias_mps": float,
    }
    sentinel = object()
    for field_name, caster in scalar_fields.items():
        raw_value = _mat_field(cfg_struct, field_name, sentinel)
        if raw_value is not sentinel:
            default_value = DEFAULT_CONFIG.get(field_name)
            raw_cfg[field_name] = _to_scalar(raw_value, default_value, caster)

    point_z_mode = _mat_field(cfg_struct, "point_z_mode", sentinel)
    if point_z_mode is not sentinel:
        raw_cfg["point_z_mode"] = _to_text(
            point_z_mode, str(DEFAULT_CONFIG["point_z_mode"])
        )
    experiment_variant = _mat_field(cfg_struct, "experiment_variant", sentinel)
    if experiment_variant is not sentinel:
        raw_cfg["experiment_variant"] = _to_text(
            experiment_variant, str(DEFAULT_CONFIG["experiment_variant"])
        )

    if "coordinate_scale_m_per_unit" not in raw_cfg:
        logger.warning(
            "输入未提供 coordinate_scale_m_per_unit，暂按1.0米/坐标单位处理；"
            "若坐标是DEM像素索引，请显式填写真实像元尺寸。"
        )

    service_times = _first_array_field(
        input_data, ("service_times_s", "inspection_service_times_s")
    )
    if service_times is not None:
        raw_cfg["service_times_s"] = service_times.reshape(-1)

    reward_struct = _mat_field(cfg_struct, "reward_weights", {})
    reward_weights: Dict[str, float] = {}
    for reward_name in ("priority", "coverage", "energy", "distance", "time"):
        raw_value = _mat_field(reward_struct, reward_name, sentinel)
        if raw_value is not sentinel:
            reward_weights[reward_name] = _to_scalar(
                raw_value, DEFAULT_CONFIG["reward_weights"][reward_name], float
            )
    if reward_weights:
        raw_cfg["reward_weights"] = reward_weights

    config = resolve_config(raw_cfg)
    return start.tolist(), points, priorities, terrain, wind_data, config


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """兼容旧接口的二维欧氏距离（坐标单位，不自动换算为米）。"""

    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(av[:2] - bv[:2]))


class TerrainModel:
    """DEM双线性采样及巡检点安全高度转换。"""

    def __init__(self, terrain: np.ndarray, cfg: Mapping[str, Any]) -> None:
        self.dem = np.asarray(terrain, dtype=np.float32)
        if self.dem.ndim != 2 or self.dem.size == 0:
            raise ValueError("terrain 必须是非空二维 DEM。")
        self.xy_scale = float(cfg["coordinate_scale_m_per_unit"])
        self.clearance = float(cfg["terrain_clearance_m"])
        self.point_z_mode = str(cfg["point_z_mode"])

    def height_at(self, x: float, y: float) -> float:
        rows, cols = self.dem.shape
        if not (0.0 <= x <= cols - 1 and 0.0 <= y <= rows - 1):
            return float("nan")
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = min(x0 + 1, cols - 1)
        y1 = min(y0 + 1, rows - 1)
        wx = float(x - x0)
        wy = float(y - y0)
        z00 = float(self.dem[y0, x0])
        z10 = float(self.dem[y0, x1])
        z01 = float(self.dem[y1, x0])
        z11 = float(self.dem[y1, x1])
        if not np.all(np.isfinite([z00, z10, z01, z11])):
            return float("nan")
        z0 = z00 * (1.0 - wx) + z10 * wx
        z1 = z01 * (1.0 - wx) + z11 * wx
        return float(z0 * (1.0 - wy) + z1 * wy)

    def target_position(self, point: Sequence[float]) -> np.ndarray:
        arr = np.asarray(point, dtype=np.float32).reshape(-1)
        ground = self.height_at(float(arr[0]), float(arr[1]))
        if not np.isfinite(ground):
            raise ValueError(f"巡检点 ({arr[0]:.3f}, {arr[1]:.3f}) 超出DEM或高程无效。")
        safe_z = ground + self.clearance
        if self.point_z_mode == "flight_altitude" and arr.size >= 3 and np.isfinite(arr[2]):
            safe_z = max(float(arr[2]), safe_z)
        return np.array([arr[0], arr[1], safe_z], dtype=np.float32)


def height_at(terrain: np.ndarray, x: float, y: float) -> float:
    """兼容旧接口：超出DEM时返回边界裁剪位置的高程。"""

    dem = np.asarray(terrain, dtype=np.float32)
    if dem.ndim != 2 or dem.size == 0:
        return 0.0
    x_clip = float(np.clip(x, 0, dem.shape[1] - 1))
    y_clip = float(np.clip(y, 0, dem.shape[0] - 1))
    model = TerrainModel(dem, {**DEFAULT_CONFIG, "terrain_clearance_m": 0.0})
    value = model.height_at(x_clip, y_clip)
    return float(value if np.isfinite(value) else 0.0)


def _meteorological_vector(speed: float, direction_deg: float, vertical_speed: float) -> np.ndarray:
    """气象风向（从何处吹来，正北顺时针）转为模型东-南-上矢量。"""

    theta = math.radians(float(direction_deg))
    return np.array(
        [
            -float(speed) * math.sin(theta),
            float(speed) * math.cos(theta),
            float(vertical_speed),
        ],
        dtype=np.float32,
    )


class WindField:
    """稀疏三维风矢量场；空间数据无效时回退到均匀风。"""

    def __init__(
        self,
        positions: Optional[np.ndarray],
        vectors: Optional[np.ndarray],
        uniform_vector: Sequence[float],
        *,
        xy_scale_m_per_unit: float,
        scale: float = 1.0,
        rotation_deg: float = 0.0,
        vertical_bias_mps: float = 0.0,
    ) -> None:
        self.xy_scale = float(xy_scale_m_per_unit)
        self.uniform_vector = self._pad_vector(uniform_vector)
        self.scale = float(scale)
        self.rotation_deg = float(rotation_deg)
        self.vertical_bias_mps = float(vertical_bias_mps)
        self.fallback_count = 0

        self.positions: Optional[np.ndarray] = None
        self.vectors: Optional[np.ndarray] = None
        if positions is not None and vectors is not None:
            p = np.asarray(positions, dtype=np.float32)
            v = np.asarray(vectors, dtype=np.float32)
            if p.ndim == 1:
                p = p.reshape(1, -1)
            if v.ndim == 1:
                v = v.reshape(1, -1)
            if p.ndim == 2 and v.ndim == 2 and p.shape[0] == v.shape[0]:
                if p.shape[1] == 2:
                    p = np.column_stack([p, np.zeros((p.shape[0],), dtype=np.float32)])
                if v.shape[1] == 2:
                    v = np.column_stack([v, np.zeros((v.shape[0],), dtype=np.float32)])
                valid = np.all(np.isfinite(p[:, :3]), axis=1) & np.all(
                    np.isfinite(v[:, :3]), axis=1
                )
                if np.any(valid):
                    self.positions = p[valid, :3].astype(np.float32)
                    self.vectors = v[valid, :3].astype(np.float32)

    @staticmethod
    def _pad_vector(vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        out = np.zeros((3,), dtype=np.float32)
        out[: min(3, arr.size)] = arr[:3]
        if not np.all(np.isfinite(out)):
            return np.zeros((3,), dtype=np.float32)
        return out

    @classmethod
    def from_data(
        cls,
        wind_data: Optional[Mapping[str, Any]],
        cfg: Mapping[str, Any],
        *,
        rng: Optional[np.random.Generator] = None,
        randomize: bool = False,
    ) -> "WindField":
        data = dict(wind_data or {})
        if "uniform_vector" in data:
            uniform = cls._pad_vector(data["uniform_vector"])
        else:
            uniform = _meteorological_vector(
                float(data.get("speed", 0.0)),
                float(data.get("direction", 0.0)),
                float(data.get("vertical_speed", 0.0)),
            )

        scale = 1.0
        rotation = 0.0
        vertical_bias = 0.0
        if randomize:
            random_source = rng or np.random.default_rng(int(cfg["seed"]))
            scale = float(
                random_source.uniform(cfg["wind_scale_min"], cfg["wind_scale_max"])
            )
            rotation_limit = float(cfg["wind_rotation_deg"])
            rotation = float(random_source.uniform(-rotation_limit, rotation_limit))
            vertical_limit = float(cfg["wind_vertical_bias_mps"])
            vertical_bias = float(random_source.uniform(-vertical_limit, vertical_limit))

        return cls(
            _to_array(data.get("positions")),
            _to_array(data.get("vectors")),
            uniform,
            xy_scale_m_per_unit=float(cfg["coordinate_scale_m_per_unit"]),
            scale=scale,
            rotation_deg=rotation,
            vertical_bias_mps=vertical_bias,
        )

    def _episode_transform(self, vector: np.ndarray) -> np.ndarray:
        theta = math.radians(self.rotation_deg)
        c, s = math.cos(theta), math.sin(theta)
        x, y, z = np.asarray(vector, dtype=np.float32)
        transformed = np.array([c * x - s * y, s * x + c * y, z], dtype=np.float32)
        transformed *= self.scale
        transformed[2] += self.vertical_bias_mps
        return transformed

    def _fallback(self) -> np.ndarray:
        self.fallback_count += 1
        return self._episode_transform(self.uniform_vector)

    def vector_at(self, position: Sequence[float]) -> np.ndarray:
        query = self._pad_vector(position)
        if self.positions is None or self.vectors is None:
            # 仅配置均匀风时，它是正式输入而不是“空间查询失败”。
            return self._episode_transform(self.uniform_vector)

        p = self.positions
        if not (
            float(np.min(p[:, 0])) <= query[0] <= float(np.max(p[:, 0]))
            and float(np.min(p[:, 1])) <= query[1] <= float(np.max(p[:, 1]))
        ):
            return self._fallback()

        deltas = p - query.reshape(1, 3)
        deltas[:, :2] *= self.xy_scale
        distances = np.linalg.norm(deltas, axis=1)
        if not np.all(np.isfinite(distances)):
            return self._fallback()
        nearest = np.argsort(distances)[: min(4, distances.size)]
        if nearest.size == 0:
            return self._fallback()
        if distances[nearest[0]] <= 1e-6:
            return self._episode_transform(self.vectors[nearest[0]])
        weights = 1.0 / np.maximum(distances[nearest], 1e-3) ** 2
        vector = np.average(self.vectors[nearest], axis=0, weights=weights)
        if not np.all(np.isfinite(vector)):
            return self._fallback()
        return self._episode_transform(vector)

    def vectors_along(self, start: Sequence[float], end: Sequence[float], count: int) -> np.ndarray:
        start_arr = self._pad_vector(start)
        end_arr = self._pad_vector(end)
        fractions = np.linspace(0.0, 1.0, max(2, int(count)), dtype=np.float32)
        return np.stack(
            [self.vector_at(start_arr + float(t) * (end_arr - start_arr)) for t in fractions]
        )


def transform_wind_for_domain_instance(
    wind_data: Optional[Mapping[str, Any]],
    instance: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """按冻结实例变换风场，顺序与 ``WindField._episode_transform`` 一致。"""

    scale = float(instance["wind_scale"])
    rotation_deg = float(instance["wind_rotation_deg"])
    vertical_bias = float(instance["wind_vertical_bias_mps"])
    if not all(math.isfinite(value) for value in (scale, rotation_deg, vertical_bias)):
        raise ValueError("冻结实例的风场参数必须是有限数值。")
    if scale <= 0.0:
        raise ValueError("冻结实例的wind_scale必须大于0。")

    transformed: Dict[str, Any] = copy.deepcopy(dict(wind_data or {}))
    if "uniform_vector" not in transformed:
        # 冻结验证关闭了WindField内部随机化，因此先把同样受支持的气象
        # 风速/风向转为“X向东、Y向南、Z向上”矢量，再统一施加域变换。
        meteorological = _meteorological_vector(
            float(transformed.get("speed", 0.0)),
            float(transformed.get("direction", 0.0)),
            float(transformed.get("vertical_speed", 0.0)),
        )
        transformed["uniform_vector"] = WindField._pad_vector(meteorological)
    theta = math.radians(rotation_deg)
    cosine, sine = math.cos(theta), math.sin(theta)
    for field_name in ("vectors", "uniform_vector"):
        if field_name not in transformed or transformed[field_name] is None:
            continue
        original = np.asarray(transformed[field_name], dtype=np.float32)
        if original.size == 0:
            transformed[field_name] = original.copy()
            continue
        if original.size % 3 != 0:
            raise ValueError(f"风场字段{field_name}的元素数必须是3的倍数。")
        vectors = original.reshape(-1, 3).copy()
        x = vectors[:, 0].copy()
        y = vectors[:, 1].copy()
        z = vectors[:, 2].copy()
        vectors[:, 0] = scale * (cosine * x - sine * y)
        vectors[:, 1] = scale * (sine * x + cosine * y)
        vectors[:, 2] = scale * z + vertical_bias
        transformed[field_name] = vectors.reshape(original.shape)
    return transformed


def apply_frozen_domain_instance(
    cfg: Mapping[str, Any],
    wind_data: Optional[Mapping[str, Any]],
    instance: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """把冻结SOC、预算、功率和风场条件精确应用到一个名义任务。"""

    missing = [field for field in FROZEN_DOMAIN_FIELDS if field not in instance]
    if missing:
        raise ValueError("冻结实例缺少字段：" + ", ".join(missing))
    config = copy.deepcopy(dict(cfg))
    initial_soc = float(instance["initial_soc"])
    distance_scale = float(instance["distance_budget_scale"])
    time_scale = float(instance["time_budget_scale"])
    power_scale = float(instance.get("power_scale", 1.0))
    if not all(
        math.isfinite(value)
        for value in (initial_soc, distance_scale, time_scale, power_scale)
    ):
        raise ValueError("冻结实例的SOC、预算比例和功率比例必须是有限数值。")
    reserve_ratio = float(
        config.get("battery_reserve_ratio", DEFAULT_CONFIG["battery_reserve_ratio"])
    )
    if not reserve_ratio < initial_soc <= 1.0:
        raise ValueError(
            "冻结实例的initial_soc必须高于battery_reserve_ratio="
            f"{reserve_ratio:.6g}，且不超过1。"
        )
    if min(distance_scale, time_scale, power_scale) <= 0.0:
        raise ValueError("冻结实例的预算比例和power_scale必须大于0。")

    config["initial_soc"] = initial_soc
    config["max_route_distance"] = float(config["max_route_distance"]) * distance_scale
    config["max_mission_time_s"] = (
        float(config["max_mission_time_s"]) * time_scale
    )
    if "service_times_s" in instance:
        config["service_times_s"] = [
            float(value) for value in instance["service_times_s"]
        ]
    for field_name in (
        "hover_power_w",
        "cruise_power_w",
        "climb_power_w",
        "descent_power_w",
    ):
        config[field_name] = float(config[field_name]) * power_scale
    return config, transform_wind_for_domain_instance(wind_data, instance)


@dataclass
class SegmentEstimate:
    feasible: bool
    reason: str
    distance_m: float
    time_s: float
    energy_wh: float
    cruise_altitude_m: float
    mean_wind_mps: np.ndarray
    flight_path: np.ndarray


class SegmentEstimator:
    """统一计算地形净空、风场可达性、时间和能耗。"""

    def __init__(self, terrain: TerrainModel, wind: WindField, cfg: Mapping[str, Any]) -> None:
        self.terrain = terrain
        self.wind = wind
        self.cfg = cfg

    def _infeasible(
        self,
        reason: str,
        start: np.ndarray,
        target: np.ndarray,
        *,
        cruise_altitude: float = float("nan"),
        mean_wind: Optional[np.ndarray] = None,
    ) -> SegmentEstimate:
        return SegmentEstimate(
            feasible=False,
            reason=reason,
            distance_m=float("inf"),
            time_s=float("inf"),
            energy_wh=float("inf"),
            cruise_altitude_m=float(cruise_altitude),
            mean_wind_mps=np.asarray(
                mean_wind if mean_wind is not None else np.zeros(3), dtype=np.float32
            ),
            flight_path=np.stack([start, target]).astype(np.float32),
        )

    @staticmethod
    def _deduplicate_path(points: List[np.ndarray]) -> np.ndarray:
        result: List[np.ndarray] = []
        for point in points:
            arr = np.asarray(point, dtype=np.float32)
            if not result or not np.allclose(result[-1], arr, atol=1e-6):
                result.append(arr)
        return np.stack(result).astype(np.float32)

    def estimate(
        self,
        start: Sequence[float],
        target: Sequence[float],
        *,
        is_takeoff: bool = False,
        is_landing: bool = False,
    ) -> SegmentEstimate:
        start_arr = np.asarray(start, dtype=np.float32).reshape(-1)[:3]
        target_arr = np.asarray(target, dtype=np.float32).reshape(-1)[:3]
        if not np.all(np.isfinite(start_arr)) or not np.all(np.isfinite(target_arr)):
            return self._infeasible("invalid_coordinate", start_arr, target_arr)

        xy_delta_m = (target_arr[:2] - start_arr[:2]) * self.terrain.xy_scale
        horizontal_distance = float(np.linalg.norm(xy_delta_m))

        # 位于机场且立即返航时，保持零资源的合法结束动作。
        if horizontal_distance <= 1e-8 and abs(float(target_arr[2] - start_arr[2])) <= 1e-8:
            return SegmentEstimate(
                feasible=True,
                reason="ok",
                distance_m=0.0,
                time_s=0.0,
                energy_wh=0.0,
                cruise_altitude_m=float(start_arr[2]),
                mean_wind_mps=self.wind.vector_at(start_arr),
                flight_path=np.stack([start_arr]).astype(np.float32),
            )

        # 安全采样不能跨过一个DEM网格，否则窄山脊可能恰好落在两个10米采样点之间。
        sample_interval = min(
            float(self.cfg["terrain_sample_interval_m"]), self.terrain.xy_scale
        )
        sample_count = max(2, int(math.ceil(horizontal_distance / sample_interval)) + 1)
        fractions = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
        terrain_heights: List[float] = []
        for fraction in fractions:
            xy = start_arr[:2] + float(fraction) * (target_arr[:2] - start_arr[:2])
            height = self.terrain.height_at(float(xy[0]), float(xy[1]))
            if not np.isfinite(height):
                return self._infeasible("terrain_out_of_bounds", start_arr, target_arr)
            terrain_heights.append(float(height))

        cruise_altitude = max(
            float(start_arr[2]),
            float(target_arr[2]),
            max(terrain_heights) + float(self.cfg["terrain_clearance_m"]),
        )
        if cruise_altitude > float(self.cfg["max_takeoff_altitude_m"]):
            return self._infeasible(
                "altitude_limit", start_arr, target_arr, cruise_altitude=cruise_altitude
            )

        wind_start = self.wind.vector_at(start_arr)
        wind_target = self.wind.vector_at(target_arr)
        wind_samples = self.wind.vectors_along(
            np.array([start_arr[0], start_arr[1], cruise_altitude], dtype=np.float32),
            np.array([target_arr[0], target_arr[1], cruise_altitude], dtype=np.float32),
            min(max(3, sample_count), 17),
        )
        mean_wind = np.mean(wind_samples, axis=0).astype(np.float32)
        max_wind = float(np.max(np.linalg.norm(wind_samples, axis=1)))
        if max_wind > float(self.cfg["max_wind_resistance"]) + 1e-6:
            return self._infeasible(
                "wind_resistance", start_arr, target_arr,
                cruise_altitude=cruise_altitude, mean_wind=mean_wind
            )
        takeoff_landing_limit = float(self.cfg["takeoff_landing_wind_resistance"])
        if is_takeoff and float(np.linalg.norm(wind_start)) > takeoff_landing_limit + 1e-6:
            return self._infeasible(
                "takeoff_wind", start_arr, target_arr,
                cruise_altitude=cruise_altitude, mean_wind=mean_wind
            )
        if is_landing and float(np.linalg.norm(wind_target)) > takeoff_landing_limit + 1e-6:
            return self._infeasible(
                "landing_wind", start_arr, target_arr,
                cruise_altitude=cruise_altitude, mean_wind=mean_wind
            )

        horizontal_time = 0.0
        if horizontal_distance > EPS:
            track = xy_delta_m / horizontal_distance
            wind_xy = mean_wind[:2]
            parallel = float(np.dot(wind_xy, track))
            cross_vec = wind_xy - parallel * track
            cross_speed = float(np.linalg.norm(cross_vec))
            airspeed = min(
                float(self.cfg["cruise_speed_mps"]),
                float(self.cfg["max_horizontal_speed"]),
            )
            if cross_speed >= airspeed - 1e-6:
                return self._infeasible(
                    "crosswind_tracking", start_arr, target_arr,
                    cruise_altitude=cruise_altitude, mean_wind=mean_wind
                )
            ground_speed = parallel + math.sqrt(max(airspeed * airspeed - cross_speed**2, 0.0))
            if ground_speed < float(self.cfg["min_ground_speed_mps"]):
                return self._infeasible(
                    "headwind_no_progress", start_arr, target_arr,
                    cruise_altitude=cruise_altitude, mean_wind=mean_wind
                )
            horizontal_time = horizontal_distance / ground_speed

        climb_distance = max(0.0, cruise_altitude - float(start_arr[2]))
        descent_distance = max(0.0, cruise_altitude - float(target_arr[2]))
        climb_ground_speed = float(self.cfg["max_ascent_speed"]) + float(wind_start[2])
        descent_ground_speed = float(self.cfg["max_descent_speed"]) - float(wind_target[2])
        if climb_distance > EPS and climb_ground_speed <= 0.1:
            return self._infeasible(
                "vertical_wind_climb", start_arr, target_arr,
                cruise_altitude=cruise_altitude, mean_wind=mean_wind
            )
        if descent_distance > EPS and descent_ground_speed <= 0.1:
            return self._infeasible(
                "vertical_wind_descent", start_arr, target_arr,
                cruise_altitude=cruise_altitude, mean_wind=mean_wind
            )

        climb_time = climb_distance / max(climb_ground_speed, EPS)
        descent_time = descent_distance / max(descent_ground_speed, EPS)
        total_time = horizontal_time + climb_time + descent_time
        total_distance = horizontal_distance + climb_distance + descent_distance
        raw_energy = (
            float(self.cfg["cruise_power_w"]) * horizontal_time
            + float(self.cfg["climb_power_w"]) * climb_time
            + float(self.cfg["descent_power_w"]) * descent_time
        ) / 3600.0
        total_energy = raw_energy * float(self.cfg["resource_safety_factor"])

        flight_path = self._deduplicate_path(
            [
                start_arr,
                np.array([start_arr[0], start_arr[1], cruise_altitude], dtype=np.float32),
                np.array([target_arr[0], target_arr[1], cruise_altitude], dtype=np.float32),
                target_arr,
            ]
        )
        return SegmentEstimate(
            feasible=True,
            reason="ok",
            distance_m=float(total_distance),
            time_s=float(total_time),
            energy_wh=float(total_energy),
            cruise_altitude_m=float(cruise_altitude),
            mean_wind_mps=mean_wind,
            flight_path=flight_path,
        )


def calculate_energy_consumption(
    start_pos: Sequence[float],
    end_pos: Sequence[float],
    terrain: np.ndarray,
    cfg: Optional[Mapping[str, Any]] = None,
    wind_data: Optional[Mapping[str, Any]] = None,
) -> float:
    """兼容旧接口；权威实现由 SegmentEstimator 统一计算。"""

    config = resolve_config(cfg)
    terrain_model = TerrainModel(terrain, config)
    wind = WindField.from_data(wind_data, config)
    estimate = SegmentEstimator(terrain_model, wind, config).estimate(start_pos, end_pos)
    return float(estimate.energy_wh if estimate.feasible else float("inf"))


@dataclass
class ActionEstimate:
    action_idx: int
    outgoing: SegmentEstimate
    return_segment: Optional[SegmentEstimate]
    service_time_s: float
    service_energy_wh: float


def _service_times(points_count: int, cfg: Mapping[str, Any]) -> np.ndarray:
    if "service_times_s" in cfg and cfg["service_times_s"] is not None:
        values = np.asarray(cfg["service_times_s"], dtype=np.float32).reshape(-1)
        if values.size != points_count:
            raise ValueError("service_times_s 长度必须与巡检点数量一致。")
        if np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError("service_times_s 必须是有限非负数。")
        return values
    return np.full(
        (points_count,), float(cfg["inspection_service_time_s"]), dtype=np.float32
    )


def build_episode(
    start_pos: Sequence[float],
    points: np.ndarray,
    terrain: np.ndarray,
    cfg: Mapping[str, Any],
    wind_data: Optional[Mapping[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
    *,
    randomize: bool = False,
) -> Dict[str, Any]:
    """构建一个必须返航的资源约束巡检回合。"""

    config = resolve_config(cfg)
    randomize = bool(randomize and config["domain_randomization"])
    points_arr = _normalize_points(points)
    terrain_model = TerrainModel(terrain, config)
    targets = np.stack([terrain_model.target_position(point) for point in points_arr])

    start = np.asarray(start_pos, dtype=np.float32).reshape(-1)
    if start.size < 2:
        raise ValueError("start_pos 至少需要 x/y 两个分量。")
    ground_at_start = terrain_model.height_at(float(start[0]), float(start[1]))
    if not np.isfinite(ground_at_start):
        raise ValueError("起点超出DEM范围或起点地面高程无效。")
    if start.size == 2:
        # 二维输入表示地面起飞点；航段模型随后显式计算爬升阶段。
        start = np.append(start, ground_at_start)
    else:
        start = start[:3]
        if not np.all(np.isfinite(start)):
            raise ValueError("start_pos 包含 NaN 或 Inf。")
        if float(start[2]) < float(ground_at_start) - 1e-6:
            raise ValueError(
                f"起点高度 {float(start[2]):.3f} m 低于DEM地面高程 "
                f"{float(ground_at_start):.3f} m。"
            )
    start = start.astype(np.float32)

    random_source = rng or np.random.default_rng(int(config["seed"]))
    if randomize:
        initial_soc = float(
            random_source.uniform(config["initial_soc_min"], config["initial_soc_max"])
        )
        distance_scale = float(
            random_source.uniform(
                config["distance_budget_scale_min"], config["distance_budget_scale_max"]
            )
        )
        time_scale = float(
            random_source.uniform(config["time_budget_scale_min"], config["time_budget_scale_max"])
        )
    else:
        initial_soc = float(config["initial_soc"])
        distance_scale = 1.0
        time_scale = 1.0

    capacity = float(config["battery_capacity"])
    initial_energy = capacity * initial_soc
    reserve_energy = capacity * float(config["battery_reserve_ratio"])
    usable_energy = initial_energy - reserve_energy
    if usable_energy <= 0:
        raise ValueError("初始电量低于或等于安全预留电量，任务不可启动。")

    wind = WindField.from_data(
        wind_data, config, rng=random_source, randomize=randomize
    )
    estimator = SegmentEstimator(terrain_model, wind, config)
    n_nodes = points_arr.shape[0]
    all_positions = np.vstack([targets, start.reshape(1, 3)]).astype(np.float32)

    return {
        "cfg": config,
        "terrain_model": terrain_model,
        "wind_field": wind,
        "segment_estimator": estimator,
        "points": points_arr,
        "target_positions": targets,
        "all_positions": all_positions,
        "service_times_s": _service_times(n_nodes, config),
        "start_pos": start.copy(),
        "current": start.copy(),
        "current_node_index": n_nodes,  # N 表示机场token
        "remaining_idx": list(range(n_nodes)),
        "visited": [],
        "last_direction": np.zeros((3,), dtype=np.float32),
        "total_energy_consumed": 0.0,
        "total_distance": 0.0,
        "total_time_s": 0.0,
        "battery_capacity": capacity,
        "initial_energy_wh": initial_energy,
        "energy_reserve_wh": reserve_energy,
        "energy_budget_wh": usable_energy,
        "max_route_distance": float(config["max_route_distance"]) * distance_scale,
        "max_mission_time_s": float(config["max_mission_time_s"]) * time_scale,
        "path_history": [start.copy()],
        "flight_path": [start.copy()],
        "executed_segments": [],
        "edge_cache": {},
        "done": False,
        "termination_reason": None,
        "last_info": {},
        "min_remaining_soc": initial_soc,
        "constraint_violation_count": 0,
        "constraint_violations": [],
        "episode_randomization": {
            "initial_soc": initial_soc,
            "distance_scale": distance_scale,
            "time_scale": time_scale,
            "wind_scale": wind.scale,
            "wind_rotation_deg": wind.rotation_deg,
            "wind_vertical_bias_mps": wind.vertical_bias_mps,
        },
    }


def _get_segment(
    state: Dict[str, Any],
    from_idx: int,
    to_idx: int,
    *,
    is_takeoff: bool = False,
    is_landing: bool = False,
) -> SegmentEstimate:
    key = (int(from_idx), int(to_idx), bool(is_takeoff), bool(is_landing))
    cache: Dict[Tuple[int, int, bool, bool], SegmentEstimate] = state["edge_cache"]
    if key not in cache:
        positions = state["all_positions"]
        cache[key] = state["segment_estimator"].estimate(
            positions[from_idx], positions[to_idx],
            is_takeoff=is_takeoff, is_landing=is_landing
        )
    return cache[key]


def _compute_action_context(
    state: Dict[str, Any], priorities: np.ndarray, *, reserve_return: bool = True
) -> Tuple[List[ActionEstimate], Dict[str, np.ndarray]]:
    """计算动作合法性；正常策略会把候选点返航资源一并预留。"""

    priorities_arr = np.asarray(priorities, dtype=np.float32).reshape(-1)
    n_nodes = state["target_positions"].shape[0]
    if priorities_arr.size != n_nodes:
        raise ValueError("priorities 长度必须与巡检点数量一致。")
    depot_idx = n_nodes
    current_idx = int(state["current_node_index"])
    is_first_takeoff = current_idx == depot_idx and len(state["visited"]) == 0

    estimates: List[ActionEstimate] = []
    m_visit = np.ones((n_nodes + 1,), dtype=bool)
    m_energy = np.ones((n_nodes + 1,), dtype=bool)
    m_distance = np.ones((n_nodes + 1,), dtype=bool)
    m_time = np.ones((n_nodes + 1,), dtype=bool)
    m_dynamics = np.ones((n_nodes + 1,), dtype=bool)
    visited_set = set(int(i) for i in state["visited"])

    safety_factor = float(state["cfg"]["resource_safety_factor"])
    hover_power = float(state["cfg"]["hover_power_w"])
    for idx in range(n_nodes):
        outgoing = _get_segment(
            state, current_idx, idx, is_takeoff=is_first_takeoff, is_landing=False
        )
        return_segment = _get_segment(
            state, idx, depot_idx, is_takeoff=False, is_landing=True
        )
        service_time = float(state["service_times_s"][idx])
        service_energy = hover_power * service_time / 3600.0 * safety_factor
        estimates.append(
            ActionEstimate(idx, outgoing, return_segment, service_time, service_energy)
        )

        m_visit[idx] = idx not in visited_set
        m_dynamics[idx] = outgoing.feasible and (
            return_segment.feasible if reserve_return else True
        )
        if m_dynamics[idx]:
            projected_energy = (
                state["total_energy_consumed"]
                + outgoing.energy_wh
                + service_energy
                + (return_segment.energy_wh if reserve_return else 0.0)
            )
            projected_distance = (
                state["total_distance"]
                + outgoing.distance_m
                + (return_segment.distance_m if reserve_return else 0.0)
            )
            projected_time = (
                state["total_time_s"]
                + outgoing.time_s
                + service_time
                + (return_segment.time_s if reserve_return else 0.0)
            )
            m_energy[idx] = projected_energy <= state["energy_budget_wh"] + 1e-6
            m_distance[idx] = projected_distance <= state["max_route_distance"] + 1e-6
            m_time[idx] = projected_time <= state["max_mission_time_s"] + 1e-6
        # 动力学不可行时，航段资源估计是 Inf，不能因此伪造三类资源超限。
        # m_dynamics 已经会单独屏蔽该动作；资源掩码保持 True 表示“未评估”。

    return_segment = _get_segment(
        state, current_idx, depot_idx, is_takeoff=False, is_landing=True
    )
    estimates.append(ActionEstimate(depot_idx, return_segment, None, 0.0, 0.0))
    m_visit[depot_idx] = True
    m_dynamics[depot_idx] = return_segment.feasible
    if return_segment.feasible:
        m_energy[depot_idx] = (
            state["total_energy_consumed"] + return_segment.energy_wh
            <= state["energy_budget_wh"] + 1e-6
        )
        m_distance[depot_idx] = (
            state["total_distance"] + return_segment.distance_m
            <= state["max_route_distance"] + 1e-6
        )
        m_time[depot_idx] = (
            state["total_time_s"] + return_segment.time_s
            <= state["max_mission_time_s"] + 1e-6
        )
    # 返航动力学不可行时同理只由 m_dynamics 标记失败。

    legal = m_visit & m_energy & m_distance & m_time & m_dynamics
    if not np.any(legal):
        raise ConstraintViolationError(
            "没有任何合法动作，且返航动作也不可行；此前动作掩码未保持返航不变量。"
        )
    return estimates, {
        "m_visit": m_visit,
        "m_energy": m_energy,
        "m_distance": m_distance,
        "m_time": m_time,
        "m_dynamics": m_dynamics,
        "legal": legal,
    }


def _append_flight_path(state: Dict[str, Any], segment: SegmentEstimate) -> None:
    for point in segment.flight_path[1:]:
        arr = np.asarray(point, dtype=np.float32)
        if not np.allclose(state["flight_path"][-1], arr, atol=1e-6):
            state["flight_path"].append(arr.copy())


def _resource_cost_reward(
    delta_energy: float, delta_distance: float, delta_time: float,
    state: Mapping[str, Any],
    priorities: Optional[np.ndarray] = None,
) -> float:
    if not bool(state["cfg"]["resource_shaping"]):
        return 0.0
    if str(state["cfg"].get("reward_schema", "legacy_v2")) == "multimap_v3_1":
        priorities_arr = np.clip(
            np.asarray(priorities, dtype=np.float64).reshape(-1),
            0.0,
            None,
        )
        total_priority = float(np.sum(priorities_arr))
        minimum_priority = (
            float(np.min(priorities_arr))
            if priorities_arr.size and total_priority > EPS
            else 0.0
        )
        secondary_scale = 0.25 * minimum_priority / max(total_priority, EPS)
        mean_incremental_utilization = (
            delta_energy / max(state["energy_budget_wh"], EPS)
            + delta_distance / max(state["max_route_distance"], EPS)
            + delta_time / max(state["max_mission_time_s"], EPS)
        ) / 3.0
        return -secondary_scale * mean_incremental_utilization
    weights = state["cfg"]["reward_weights"]
    return -(
        float(weights["energy"]) * delta_energy / max(state["energy_budget_wh"], EPS)
        + float(weights["distance"]) * delta_distance / max(state["max_route_distance"], EPS)
        + float(weights["time"]) * delta_time / max(state["max_mission_time_s"], EPS)
    )


def step_env_improved(
    state: Dict[str, Any],
    action_idx: int,
    points: np.ndarray,
    priorities: np.ndarray,
    terrain: np.ndarray,
    cfg: Mapping[str, Any],
    wind_data: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], float, bool]:
    """执行一次巡检或返航动作；非法动作直接报错而不是靠负奖励学习。"""

    del points, terrain, cfg, wind_data  # 环境所需对象已冻结在 state 中
    if state.get("done", False):
        raise RuntimeError("回合已经终止，不能继续执行动作。")

    priorities_arr = np.asarray(priorities, dtype=np.float32).reshape(-1)
    estimates, masks = _compute_action_context(state, priorities_arr)
    n_nodes = priorities_arr.size
    action_idx = int(action_idx)
    if action_idx < 0 or action_idx > n_nodes:
        raise IndexError(f"动作 {action_idx} 超出 [0, {n_nodes}]。")
    if not bool(masks["legal"][action_idx]):
        failed = [name for name, mask in masks.items() if name != "legal" and not mask[action_idx]]
        if not bool(state["cfg"]["return_reserve_mask"]):
            relaxed_masks = _compute_action_context(
                state, priorities_arr, reserve_return=False
            )[1]
            if bool(relaxed_masks["legal"][action_idx]):
                # 仿真消融允许策略“提出”未预留返航的动作，但环境不移动无人机。
                # 这样能测量安全掩码的贡献，同时绝不会执行真实约束下不可行的航段。
                violation = {
                    "attempted_action": action_idx,
                    "failed_constraints": failed,
                    "position": np.asarray(
                        state["current"], dtype=np.float32
                    ).tolist(),
                }
                state["constraint_violation_count"] += 1
                state["constraint_violations"].append(violation)
                state["done"] = True
                state["termination_reason"] = "stranded"
                reward = float(state["cfg"]["simulation_violation_penalty"])
                state["last_info"] = {
                    "action_idx": action_idx,
                    "attempted_only": True,
                    "failed_constraints": failed,
                    "reward": reward,
                    "done": True,
                    "termination_reason": "stranded",
                }
                return state, reward, True
        raise ConstraintViolationError(f"动作 {action_idx} 非法，失败约束：{failed}")

    estimate = estimates[action_idx]
    old_current = np.asarray(state["current"], dtype=np.float32).copy()
    old_node_index = int(state["current_node_index"])
    delta_energy = estimate.outgoing.energy_wh + estimate.service_energy_wh
    delta_distance = estimate.outgoing.distance_m
    delta_time = estimate.outgoing.time_s + estimate.service_time_s

    state["total_energy_consumed"] += float(delta_energy)
    state["total_distance"] += float(delta_distance)
    state["total_time_s"] += float(delta_time)
    _append_flight_path(state, estimate.outgoing)
    state["executed_segments"].append(
        {
            "from_node_index": old_node_index,
            "to_node_index": action_idx,
            "is_return": bool(action_idx == n_nodes),
            "feasible": bool(estimate.outgoing.feasible),
            "reason": estimate.outgoing.reason,
            "flight_distance_m": float(estimate.outgoing.distance_m),
            "flight_time_s": float(estimate.outgoing.time_s),
            "flight_energy_wh": float(estimate.outgoing.energy_wh),
            "service_time_s": float(estimate.service_time_s),
            "service_energy_wh": float(estimate.service_energy_wh),
            "total_time_s": float(delta_time),
            "total_energy_wh": float(delta_energy),
            "cruise_altitude_m": float(estimate.outgoing.cruise_altitude_m),
            "mean_wind_mps": estimate.outgoing.mean_wind_mps.copy(),
            "flight_path": estimate.outgoing.flight_path.copy(),
        }
    )

    reward = _resource_cost_reward(
        delta_energy,
        delta_distance,
        delta_time,
        state,
        priorities_arr,
    )
    if action_idx == n_nodes:
        state["current"] = state["start_pos"].copy()
        state["current_node_index"] = n_nodes
        if not np.allclose(state["path_history"][-1], state["start_pos"], atol=1e-6):
            state["path_history"].append(state["start_pos"].copy())
        state["done"] = True
        state["termination_reason"] = (
            "returned_full" if len(state["visited"]) == n_nodes else "returned_partial"
        )
    else:
        target = state["target_positions"][action_idx].copy()
        state["current"] = target
        state["current_node_index"] = action_idx
        state["visited"].append(action_idx)
        state["remaining_idx"].remove(action_idx)
        state["path_history"].append(target.copy())

        total_priority = float(np.sum(np.clip(priorities_arr, 0.0, None)))
        if total_priority <= EPS:
            priority_gain = 1.0 / max(n_nodes, 1)
        else:
            priority_gain = max(float(priorities_arr[action_idx]), 0.0) / total_priority
        if str(state["cfg"].get("reward_schema", "legacy_v2")) == "multimap_v3_1":
            minimum_priority = (
                float(np.min(np.clip(priorities_arr, 0.0, None)))
                if priorities_arr.size and total_priority > EPS
                else 0.0
            )
            secondary_scale = 0.25 * minimum_priority / max(total_priority, EPS)
            reward += priority_gain + secondary_scale * 0.25 / max(n_nodes, 1)
        else:
            reward += (
                float(state["cfg"]["reward_weights"]["priority"]) * priority_gain
                + float(state["cfg"]["reward_weights"]["coverage"]) / max(n_nodes, 1)
            )

    physical_delta = np.array(
        [
            (state["current"][0] - old_current[0])
            * float(state["cfg"]["coordinate_scale_m_per_unit"]),
            (state["current"][1] - old_current[1])
            * float(state["cfg"]["coordinate_scale_m_per_unit"]),
            state["current"][2] - old_current[2],
        ],
        dtype=np.float32,
    )
    direction_norm = float(np.linalg.norm(physical_delta))
    if direction_norm > EPS:
        state["last_direction"] = physical_delta / direction_norm

    remaining_soc = (
        state["initial_energy_wh"] - state["total_energy_consumed"]
    ) / max(state["battery_capacity"], EPS)
    state["min_remaining_soc"] = min(float(state["min_remaining_soc"]), remaining_soc)

    if state["total_energy_consumed"] > state["energy_budget_wh"] + 1e-5:
        raise ConstraintViolationError("执行后能耗超过可用电量预算。")
    if state["total_distance"] > state["max_route_distance"] + 1e-5:
        raise ConstraintViolationError("执行后航程超过距离预算。")
    if state["total_time_s"] > state["max_mission_time_s"] + 1e-5:
        raise ConstraintViolationError("执行后时间超过任务预算。")

    state["last_info"] = {
        "action_idx": action_idx,
        "delta_energy_wh": float(delta_energy),
        "delta_distance_m": float(delta_distance),
        "delta_time_s": float(delta_time),
        "reward": float(reward),
        "done": bool(state["done"]),
        "termination_reason": state["termination_reason"],
    }
    return state, float(reward), bool(state["done"])


def _safe_ratio(value: float, denominator: float, *, upper: float = 2.0) -> float:
    if not np.isfinite(value):
        return float(upper)
    return float(np.clip(value / max(denominator, EPS), -upper, upper))


def _build_priority_bias(priorities: np.ndarray) -> np.ndarray:
    priorities_arr = np.asarray(priorities, dtype=np.float32).reshape(-1)
    nonnegative = np.clip(priorities_arr, 0.0, None)
    max_priority = float(np.max(nonnegative)) if nonnegative.size else 0.0
    normalized = nonnegative / max(max_priority, EPS) if max_priority > EPS else nonnegative
    normalized = np.append(normalized, 0.0).astype(np.float32)  # 返航token优先级为0
    m = normalized.size
    return np.broadcast_to(normalized.reshape(1, 1, -1), (1, m, m)).copy()


def _build_observation(
    state: Dict[str, Any], priorities: np.ndarray
) -> Dict[str, np.ndarray]:
    priorities_arr = np.asarray(priorities, dtype=np.float32).reshape(-1)
    n_nodes = priorities_arr.size
    depot_idx = n_nodes
    # no_return_reserve仅放宽“策略看见的掩码”；环境执行仍使用完整返航预留约束。
    estimates, masks = _compute_action_context(
        state,
        priorities_arr,
        reserve_return=bool(state["cfg"]["return_reserve_mask"]),
    )

    features = np.zeros((n_nodes + 1, NODE_FEATURE_DIM), dtype=np.float32)
    current = np.asarray(state["current"], dtype=np.float32)
    scale_xy = float(state["cfg"]["coordinate_scale_m_per_unit"])
    position_scale = max(float(state["max_route_distance"]), 1.0)
    priority_max = max(float(np.max(np.clip(priorities_arr, 0.0, None))), EPS)
    visited_set = set(state["visited"])

    for idx, estimate in enumerate(estimates):
        target = state["all_positions"][idx]
        rel = np.array(
            [
                (target[0] - current[0]) * scale_xy,
                (target[1] - current[1]) * scale_xy,
                target[2] - current[2],
            ],
            dtype=np.float32,
        ) / position_scale
        features[idx, 0:3] = np.clip(rel, -2.0, 2.0)
        features[idx, 3] = (
            max(float(priorities_arr[idx]), 0.0) / priority_max if idx < n_nodes else 0.0
        )
        features[idx, 4] = 1.0 if idx in visited_set else 0.0
        features[idx, 5] = 1.0 if idx == depot_idx else 0.0

        edge_energy = estimate.outgoing.energy_wh + estimate.service_energy_wh
        edge_time = estimate.outgoing.time_s + estimate.service_time_s
        features[idx, 6] = _safe_ratio(
            estimate.outgoing.distance_m, state["max_route_distance"]
        )
        features[idx, 7] = _safe_ratio(edge_energy, state["energy_budget_wh"])
        features[idx, 8] = _safe_ratio(edge_time, state["max_mission_time_s"])
        if estimate.return_segment is not None:
            features[idx, 9] = _safe_ratio(
                estimate.return_segment.distance_m, state["max_route_distance"]
            )
            features[idx, 10] = _safe_ratio(
                estimate.return_segment.energy_wh, state["energy_budget_wh"]
            )
            features[idx, 11] = _safe_ratio(
                estimate.return_segment.time_s, state["max_mission_time_s"]
            )
        max_wind = max(float(state["cfg"]["max_wind_resistance"]), EPS)
        features[idx, 12:15] = np.clip(
            estimate.outgoing.mean_wind_mps / max_wind, -2.0, 2.0
        )

    total_priority = float(np.sum(np.clip(priorities_arr, 0.0, None)))
    visited_priority = float(
        np.sum([max(float(priorities_arr[i]), 0.0) for i in state["visited"]])
    )
    priority_coverage = (
        visited_priority / total_priority
        if total_priority > EPS
        else len(state["visited"]) / max(n_nodes, 1)
    )
    current_wind = state["wind_field"].vector_at(current)
    relative_current = np.array(
        [
            (current[0] - state["start_pos"][0]) * scale_xy,
            (current[1] - state["start_pos"][1]) * scale_xy,
            current[2] - state["start_pos"][2],
        ],
        dtype=np.float32,
    ) / position_scale
    s_uav = np.concatenate(
        [
            np.clip(relative_current, -2.0, 2.0),
            np.asarray(state["last_direction"], dtype=np.float32),
            np.array(
                [
                    max(
                        0.0,
                        (state["energy_budget_wh"] - state["total_energy_consumed"])
                        / max(state["energy_budget_wh"], EPS),
                    ),
                    max(
                        0.0,
                        (state["max_route_distance"] - state["total_distance"])
                        / max(state["max_route_distance"], EPS),
                    ),
                    max(
                        0.0,
                        (state["max_mission_time_s"] - state["total_time_s"])
                        / max(state["max_mission_time_s"], EPS),
                    ),
                    len(state["visited"]) / max(n_nodes, 1),
                    priority_coverage,
                ],
                dtype=np.float32,
            ),
            np.clip(
                current_wind / max(float(state["cfg"]["max_wind_resistance"]), EPS),
                -2.0,
                2.0,
            ),
        ]
    ).astype(np.float32)
    if s_uav.size != UAV_FEATURE_DIM:
        raise AssertionError(f"UAV特征维度应为{UAV_FEATURE_DIM}，当前为{s_uav.size}")

    remaining_mask = np.zeros((n_nodes + 1,), dtype=bool)
    for idx in state["remaining_idx"]:
        remaining_mask[int(idx)] = True

    return {
        "s_env": features,
        "s_uav": s_uav,
        "m_priority": _build_priority_bias(priorities_arr),
        "m_visit": masks["m_visit"],
        "m_energy": masks["m_energy"],
        "m_distance": masks["m_distance"],
        "m_time": masks["m_time"],
        "m_dynamics": masks["m_dynamics"],
        "m_remaining": remaining_mask,
        "legal_mask": masks["legal"],
    }


def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, length, model_dim = x.shape
    head_dim = model_dim // n_heads
    return x.view(batch, length, n_heads, head_dim).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, n_heads, length, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, length, n_heads * head_dim)


def _valid_mask(mask: torch.Tensor) -> torch.Tensor:
    """同时接受v2布尔掩码和旧版0/-1e9浮点掩码。"""

    if mask.dtype == torch.bool:
        return mask
    return mask > -1e8


class PriorityEncoder(nn.Module):
    """带巡检优先级加性偏置的集合自注意力编码器。"""

    def __init__(
        self, d_env: int, d_model: int, n_heads: int, lambda_priority: float
    ) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model) // int(n_heads)
        self.lambda_priority = float(lambda_priority)
        self.input_proj = nn.Linear(int(d_env), int(d_model))
        self.q_proj = nn.Linear(int(d_model), int(d_model), bias=False)
        self.k_proj = nn.Linear(int(d_model), int(d_model), bias=False)
        self.v_proj = nn.Linear(int(d_model), int(d_model), bias=False)
        self.out_proj = nn.Linear(int(d_model), int(d_model))
        self.norm1 = nn.LayerNorm(int(d_model))
        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) * 2),
            nn.GELU(),
            nn.Linear(int(d_model) * 2, int(d_model)),
        )
        self.norm2 = nn.LayerNorm(int(d_model))

    def forward(self, s_env: torch.Tensor, m_priority: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(s_env)
        q = _split_heads(self.q_proj(x), self.n_heads)
        k = _split_heads(self.k_proj(x), self.n_heads)
        v = _split_heads(self.v_proj(x), self.n_heads)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = m_priority
        if bias.ndim == 3:
            bias = bias.unsqueeze(1)
        scores = scores + self.lambda_priority * bias
        attention = torch.softmax(scores, dim=-1)
        attended = self.out_proj(_merge_heads(torch.matmul(attention, v)))
        x = self.norm1(x + attended)
        return self.norm2(x + self.ffn(x))


class DecoderActor(nn.Module):
    """Pointer解码器：glimpse和最终logits均应用同一合法动作掩码。"""

    def __init__(self, d_uav: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model) // int(n_heads)
        self.uav_proj = nn.Sequential(
            nn.Linear(int(d_uav), int(d_model)), nn.Tanh(), nn.Linear(int(d_model), int(d_model))
        )
        self.q_glimpse = nn.Linear(int(d_model), int(d_model), bias=False)
        self.k_glimpse = nn.Linear(int(d_model), int(d_model), bias=False)
        self.v_glimpse = nn.Linear(int(d_model), int(d_model), bias=False)
        self.glimpse_out = nn.Linear(int(d_model), int(d_model))
        self.gate = nn.Linear(int(d_model) * 2, int(d_model))
        self.query_final = nn.Linear(int(d_model), int(d_model), bias=False)
        self.key_pointer = nn.Linear(int(d_model), int(d_model), bias=False)
        self.logit_scale = 10.0

    def forward(
        self, x_bar: torch.Tensor, s_uav: torch.Tensor, legal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.all(torch.any(legal_mask, dim=-1)):
            raise ConstraintViolationError("Actor收到全False合法动作掩码。")
        q_raw = self.uav_proj(s_uav).unsqueeze(1)
        q = _split_heads(self.q_glimpse(q_raw), self.n_heads)
        k = _split_heads(self.k_glimpse(x_bar), self.n_heads)
        v = _split_heads(self.v_glimpse(x_bar), self.n_heads)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~legal_mask[:, None, None, :], -torch.inf)
        context = self.glimpse_out(_merge_heads(torch.matmul(torch.softmax(scores, -1), v)))
        gate = torch.sigmoid(self.gate(torch.cat([q_raw, context], dim=-1)))
        q_final = gate * context + (1.0 - gate) * q_raw

        pointer_q = self.query_final(q_final)
        pointer_k = self.key_pointer(x_bar)
        logits = self.logit_scale * torch.tanh(
            torch.matmul(pointer_q, pointer_k.transpose(-2, -1)).squeeze(1)
            / math.sqrt(pointer_k.shape[-1])
        )
        logits = logits.masked_fill(~legal_mask, -torch.inf)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities, logits, q_final


class SharedNodeMLPActor(nn.Module):
    """参数量接近Pointer的逐节点共享MLP；天然支持可变节点数量。"""

    def __init__(self, d_uav: int, d_model: int) -> None:
        super().__init__()
        model_dim = int(d_model)
        hidden_dim = max(model_dim, model_dim * 7 // 4)
        self.uav_proj = nn.Sequential(
            nn.Linear(int(d_uav), model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, model_dim),
        )
        # 同一个打分器应用到每个节点，不能依赖固定N或节点槽位编号。
        self.node_scorer = nn.Sequential(
            nn.Linear(model_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x_bar: torch.Tensor, s_uav: torch.Tensor, legal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.all(torch.any(legal_mask, dim=-1)):
            raise ConstraintViolationError("Actor收到全False合法动作掩码。")
        q_final = self.uav_proj(s_uav).unsqueeze(1)
        global_context = q_final.expand(-1, x_bar.shape[1], -1)
        interactions = torch.cat(
            [x_bar, global_context, x_bar * global_context], dim=-1
        )
        logits = self.node_scorer(interactions).squeeze(-1)
        logits = logits.masked_fill(~legal_mask, -torch.inf)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities, logits, q_final


class Critic(nn.Module):
    """同时观察未访问集合和当前合法动作集合的状态价值网络。"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(int(d_model) * 3, int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), int(d_model) // 2),
            nn.GELU(),
            nn.Linear(int(d_model) // 2, 1),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=x.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denominator

    def forward(
        self,
        x_bar: torch.Tensor,
        q_final: torch.Tensor,
        remaining_mask: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        remaining_context = self._masked_mean(x_bar, remaining_mask)
        legal_context = self._masked_mean(x_bar, legal_mask)
        critic_input = torch.cat(
            [q_final.squeeze(1), remaining_context, legal_context], dim=-1
        )
        return self.value_head(critic_input)


class FlatMLPActorCritic(nn.Module):
    """v3.2 传统 PPO：固定 24 点加返航槽的纯 MLP Actor-Critic。"""

    def __init__(
        self,
        d_env: int = NODE_FEATURE_DIM,
        d_uav: int = UAV_FEATURE_DIM,
        hidden_dim: int = TRADITIONAL_PPO_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        actor_input_dim = TRADITIONAL_PPO_FIXED_SLOTS * int(d_env) + int(d_uav)
        critic_input_dim = actor_input_dim + 2 * TRADITIONAL_PPO_FIXED_SLOTS
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), TRADITIONAL_PPO_FIXED_SLOTS),
        )
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    @staticmethod
    def _fixed_slots(
        s_env: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if s_env.ndim != 3 or legal_mask.ndim != 2:
            raise ValueError("传统PPO要求批次化节点特征和合法动作掩码。")
        batch_size, variable_slots, feature_dim = s_env.shape
        node_count = int(variable_slots) - 1
        if node_count < 1 or node_count > TRADITIONAL_PPO_MAX_NODES:
            raise ValueError(
                f"传统PPO仅支持1到{TRADITIONAL_PPO_MAX_NODES}个巡检点，"
                f"当前为{node_count}。"
            )
        depot_mask = s_env[..., 5] > 0.5
        if not torch.all(depot_mask.sum(dim=-1) == 1):
            raise ValueError("传统PPO要求每个观测恰有一个返航点。")
        if not torch.all(depot_mask[:, -1]):
            raise ValueError("外部变长观测必须把返航点放在最后一个位置。")

        fixed_env = s_env.new_zeros(
            (batch_size, TRADITIONAL_PPO_FIXED_SLOTS, feature_dim)
        )
        fixed_env[:, :node_count] = s_env[:, :node_count]
        fixed_env[:, -1] = s_env[:, -1]
        valid_slots = torch.zeros(
            (batch_size, TRADITIONAL_PPO_FIXED_SLOTS),
            dtype=torch.bool,
            device=s_env.device,
        )
        valid_slots[:, :node_count] = True
        valid_slots[:, -1] = True
        fixed_legal = torch.zeros_like(valid_slots)
        fixed_legal[:, :node_count] = legal_mask[:, :node_count]
        fixed_legal[:, -1] = legal_mask[:, -1]
        if not torch.all(torch.any(fixed_legal, dim=-1)):
            raise ConstraintViolationError("传统PPO收到全False合法动作掩码。")
        return fixed_env, valid_slots, fixed_legal, node_count

    def forward(
        self,
        s_env: torch.Tensor,
        s_uav: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fixed_env, valid_slots, fixed_legal, node_count = self._fixed_slots(
            s_env, legal_mask
        )
        actor_input = torch.cat([fixed_env.flatten(start_dim=1), s_uav], dim=-1)
        fixed_logits = self.actor(actor_input).masked_fill(~fixed_legal, -torch.inf)
        fixed_probabilities = torch.softmax(fixed_logits, dim=-1)

        critic_input = torch.cat(
            [
                actor_input,
                valid_slots.to(dtype=s_env.dtype),
                fixed_legal.to(dtype=s_env.dtype),
            ],
            dim=-1,
        )
        value = self.critic(critic_input)

        # 保持旧环境动作接口：节点仍为 0..N-1，返航仍为 N。
        logits = torch.cat(
            [fixed_logits[:, :node_count], fixed_logits[:, -1:]], dim=-1
        )
        probabilities = torch.cat(
            [
                fixed_probabilities[:, :node_count],
                fixed_probabilities[:, -1:],
            ],
            dim=-1,
        )
        return probabilities, logits, value


class PPO_PtrNet(nn.Module):
    """可变节点数的共享编码器Actor-Critic；默认仍为PPO+Pointer。"""

    def __init__(
        self,
        *,
        batch_size: int = 1,
        n_nodes: int = 29,
        d_env: int = NODE_FEATURE_DIM,
        d_uav: int = UAV_FEATURE_DIM,
        d_model: int = 128,
        n_heads: int = 4,
        lambda_priority: float = 0.5,
        policy_architecture: str = "pointer",
        training_algorithm: str = "ppo",
        experiment_variant: str = "full",
    ) -> None:
        super().__init__()
        self.batch_size = int(batch_size)  # 仅保留兼容，不限制实际前向批量
        self.N = int(n_nodes)
        self.d_env = int(d_env)
        self.d_uav = int(d_uav)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.lambda_priority = float(lambda_priority)
        self.policy_architecture = str(policy_architecture)
        self.training_algorithm = str(training_algorithm)
        self.experiment_variant = str(experiment_variant)
        if self.policy_architecture not in {
            "pointer",
            "shared_node_mlp",
            "flat_mlp_24",
        }:
            raise ValueError(
                "policy_architecture只能是'pointer'、'shared_node_mlp'或"
                "'flat_mlp_24'。"
            )
        if self.training_algorithm not in {"ppo", "a2c"}:
            raise ValueError("training_algorithm只能是'ppo'或'a2c'。")
        if self.policy_architecture == "flat_mlp_24":
            self.flat_actor_critic = FlatMLPActorCritic(d_env=d_env, d_uav=d_uav)
            self.encoder = None
            self.actor = None
            self.critic = None
        else:
            self.encoder = PriorityEncoder(d_env, d_model, n_heads, lambda_priority)
            if self.policy_architecture == "pointer":
                self.actor = DecoderActor(d_uav, d_model, n_heads)
            else:
                self.actor = SharedNodeMLPActor(d_uav, d_model)
            self.critic = Critic(d_model)

    def forward(
        self,
        s_env: torch.Tensor,
        s_uav: torch.Tensor,
        m_priority: torch.Tensor,
        m_visit: torch.Tensor,
        m_energy: torch.Tensor,
        m_distance: Optional[torch.Tensor] = None,
        m_time: Optional[torch.Tensor] = None,
        m_dynamics: Optional[torch.Tensor] = None,
        m_remaining: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = [_valid_mask(m_visit), _valid_mask(m_energy)]
        for optional_mask in (m_distance, m_time, m_dynamics):
            if optional_mask is not None:
                masks.append(_valid_mask(optional_mask))
        legal_mask = masks[0]
        for valid in masks[1:]:
            legal_mask = legal_mask & valid
        if m_remaining is None:
            is_depot = s_env[..., 5] > 0.5
            m_remaining = _valid_mask(m_visit) & ~is_depot
        else:
            m_remaining = _valid_mask(m_remaining)

        if self.policy_architecture == "flat_mlp_24":
            return self.flat_actor_critic(s_env, s_uav, legal_mask)

        x_bar = self.encoder(s_env, m_priority)
        probabilities, logits, q_final = self.actor(x_bar, s_uav, legal_mask)
        value = self.critic(x_bar, q_final, m_remaining, legal_mask)
        return probabilities, logits, value


def count_trainable_parameters(model: nn.Module) -> int:
    """返回参与梯度更新的参数量，供论文公平性核对。"""

    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _experiment_metadata(
    model: PPO_PtrNet,
    cfg: Mapping[str, Any],
    environment_interactions: int = 0,
    interaction_count_complete: bool = True,
) -> Dict[str, Any]:
    metadata = {
        "variant": str(
            getattr(model, "experiment_variant", cfg.get("experiment_variant", "full"))
        ),
        "policy_architecture": str(
            getattr(
                model, "policy_architecture", cfg.get("policy_architecture", "pointer")
            )
        ),
        "training_algorithm": str(
            getattr(model, "training_algorithm", cfg.get("training_algorithm", "ppo"))
        ),
        "parameter_count": count_trainable_parameters(model),
        "environment_interactions": int(environment_interactions),
        "interaction_count_complete": bool(interaction_count_complete),
        "lambda_priority": float(
            getattr(model, "lambda_priority", cfg.get("lambda_priority", 0.5))
        ),
        "domain_randomization": bool(cfg.get("domain_randomization", True)),
        "resource_shaping": bool(cfg.get("resource_shaping", True)),
        "return_reserve_mask": bool(cfg.get("return_reserve_mask", True)),
        "simulation_only": bool(cfg.get("simulation_only", False)),
        "reward_schema": str(cfg.get("reward_schema", "legacy_v2")),
        "scenario_mode": str(cfg.get("scenario_mode", "legacy_fixed_map")),
        "scenario_provider_hash": str(cfg.get("scenario_provider_hash", "")),
    }
    if metadata["policy_architecture"] == "flat_mlp_24":
        metadata["fixed_slot_schema"] = {
            "max_inspection_nodes": TRADITIONAL_PPO_MAX_NODES,
            "inspection_slots": [0, TRADITIONAL_PPO_MAX_NODES - 1],
            "depot_slot": TRADITIONAL_PPO_FIXED_SLOTS - 1,
            "padding": "zero_features_and_illegal_mask",
        }
    return metadata


@dataclass
class PPOBatch:
    s_env: torch.Tensor
    s_uav: torch.Tensor
    m_priority: torch.Tensor
    m_visit: torch.Tensor
    m_energy: torch.Tensor
    m_distance: torch.Tensor
    m_time: torch.Tensor
    m_dynamics: torch.Tensor
    m_remaining: torch.Tensor
    action: torch.Tensor
    logp_old: torch.Tensor
    value_old: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    def to(self, target_device: torch.device) -> "PPOBatch":
        return PPOBatch(
            **{
                field_name: getattr(self, field_name).to(target_device)
                for field_name in self.__dataclass_fields__
            }
        )


class PPOAgent:
    """对冻结旧策略批次执行多轮minibatch PPO更新。"""

    def __init__(self, model: PPO_PtrNet, cfg: Mapping[str, Any]) -> None:
        self.model = model
        self.cfg = cfg
        self.opt = optim.Adam(self.model.parameters(), lr=float(cfg["lr"]), eps=1e-5)

    def update(self, batch: PPOBatch, entropy_coef: float) -> Dict[str, float]:
        self.model.train()
        batch_size = int(batch.action.shape[0])
        minibatch_size = min(int(self.cfg["minibatch_size"]), batch_size)
        clip_eps = float(self.cfg["clip_ratio"])
        target_kl = float(self.cfg["target_kl"])
        aggregates: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "gradient_norm": [],
            "gradient_norm_pre_clip": [],
            "ratio_deviation": [],
        }
        epochs_completed = 0
        post_update_kl = 0.0
        post_update_clip_fraction = 0.0
        post_update_ratio_deviation = 0.0

        for _epoch in range(int(self.cfg["ppo_epochs"])):
            permutation = torch.randperm(batch_size, device=batch.action.device)
            epoch_kls: List[float] = []
            for start in range(0, batch_size, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                _, logits, value = self.model(
                    batch.s_env[idx],
                    batch.s_uav[idx],
                    batch.m_priority[idx],
                    batch.m_visit[idx],
                    batch.m_energy[idx],
                    batch.m_distance[idx],
                    batch.m_time[idx],
                    batch.m_dynamics[idx],
                    batch.m_remaining[idx],
                )
                distribution = torch.distributions.Categorical(logits=logits)
                logp = distribution.log_prob(batch.action[idx])
                entropy = distribution.entropy().mean()
                log_ratio = logp - batch.logp_old[idx]
                ratio = torch.exp(log_ratio)
                advantage = batch.advantages[idx]

                surrogate_1 = ratio * advantage
                surrogate_2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                value_pred = value.squeeze(-1)
                value_clipped = batch.value_old[idx] + torch.clamp(
                    value_pred - batch.value_old[idx], -clip_eps, clip_eps
                )
                value_loss_unclipped = (value_pred - batch.returns[idx]) ** 2
                value_loss_clipped = (value_clipped - batch.returns[idx]) ** 2
                value_loss = 0.5 * torch.max(
                    value_loss_unclipped, value_loss_clipped
                ).mean()
                total_loss = (
                    policy_loss
                    + float(self.cfg["value_coef"]) * value_loss
                    - float(entropy_coef) * entropy
                )
                if not bool(torch.isfinite(total_loss).item()):
                    raise FloatingPointError(
                        "PPO总损失出现NaN/Inf；已停止更新，未执行optimizer.step()。"
                    )

                self.opt.zero_grad(set_to_none=True)
                total_loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.cfg["max_grad_norm"])
                )
                if not bool(torch.isfinite(gradient_norm).item()):
                    self.opt.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "PPO梯度范数出现NaN/Inf；已清空梯度，未执行optimizer.step()。"
                    )
                self.opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
                    ratio_deviation = (ratio - 1.0).abs().max()
                aggregates["policy_loss"].append(float(policy_loss.item()))
                aggregates["value_loss"].append(float(value_loss.item()))
                aggregates["entropy"].append(float(entropy.item()))
                aggregates["approx_kl"].append(float(approx_kl.item()))
                aggregates["clip_fraction"].append(float(clip_fraction.item()))
                pre_clip_norm = float(gradient_norm.item())
                aggregates["gradient_norm_pre_clip"].append(pre_clip_norm)
                aggregates["gradient_norm"].append(
                    min(pre_clip_norm, float(self.cfg["max_grad_norm"]))
                )
                aggregates["ratio_deviation"].append(float(ratio_deviation.item()))
                epoch_kls.append(float(approx_kl.item()))

            epochs_completed += 1
            # 用“本轮全部参数更新完成后的策略”重新计算完整批次KL，避免漏测最后一步更新。
            with torch.no_grad():
                _, diagnostic_logits, _ = self.model(
                    batch.s_env,
                    batch.s_uav,
                    batch.m_priority,
                    batch.m_visit,
                    batch.m_energy,
                    batch.m_distance,
                    batch.m_time,
                    batch.m_dynamics,
                    batch.m_remaining,
                )
                diagnostic_distribution = torch.distributions.Categorical(
                    logits=diagnostic_logits
                )
                diagnostic_logp = diagnostic_distribution.log_prob(batch.action)
                diagnostic_log_ratio = diagnostic_logp - batch.logp_old
                diagnostic_ratio = torch.exp(diagnostic_log_ratio)
                post_update_kl = float(
                    (((diagnostic_ratio - 1.0) - diagnostic_log_ratio).mean()).item()
                )
                post_update_clip_fraction = float(
                    ((diagnostic_ratio - 1.0).abs() > clip_eps).float().mean().item()
                )
                post_update_ratio_deviation = float(
                    (diagnostic_ratio - 1.0).abs().max().item()
                )
            if not np.all(
                np.isfinite(
                    [
                        post_update_kl,
                        post_update_clip_fraction,
                        post_update_ratio_deviation,
                    ]
                )
            ):
                raise FloatingPointError("PPO更新后诊断指标出现NaN/Inf。")
            if post_update_kl > target_kl:
                break

        with torch.no_grad():
            _, _, value_after = self.model(
                batch.s_env,
                batch.s_uav,
                batch.m_priority,
                batch.m_visit,
                batch.m_energy,
                batch.m_distance,
                batch.m_time,
                batch.m_dynamics,
                batch.m_remaining,
            )
            targets = batch.returns
            residual_variance = torch.var(targets - value_after.squeeze(-1), unbiased=False)
            target_variance = torch.var(targets, unbiased=False)
            explained_variance = (
                1.0 - residual_variance / target_variance
                if target_variance > 1e-8
                else torch.tensor(0.0, device=targets.device)
            )

        result = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in aggregates.items()
        }
        result["epochs_completed"] = float(epochs_completed)
        result["explained_variance"] = float(explained_variance.item())
        result["entropy_coef"] = float(entropy_coef)
        result["approx_kl"] = post_update_kl
        result["clip_fraction"] = post_update_clip_fraction
        result["ratio_deviation"] = post_update_ratio_deviation
        result["kl_early_stopped"] = bool(
            post_update_kl > target_kl
            and epochs_completed < int(self.cfg["ppo_epochs"])
        )
        return result


class A2CAgent:
    """同步Advantage Actor-Critic：每批轨迹仅使用一次，不做PPO裁剪。"""

    def __init__(self, model: PPO_PtrNet, cfg: Mapping[str, Any]) -> None:
        self.model = model
        self.cfg = cfg
        self.opt = optim.Adam(self.model.parameters(), lr=float(cfg["lr"]), eps=1e-5)

    def update(self, batch: PPOBatch, entropy_coef: float) -> Dict[str, float]:
        self.model.train()
        batch_size = int(batch.action.shape[0])
        minibatch_size = min(int(self.cfg["minibatch_size"]), batch_size)
        permutation = torch.randperm(batch_size, device=batch.action.device)
        aggregates: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "gradient_norm": [],
            "gradient_norm_pre_clip": [],
        }
        for start in range(0, batch_size, minibatch_size):
            idx = permutation[start : start + minibatch_size]
            _, logits, value = self.model(
                batch.s_env[idx],
                batch.s_uav[idx],
                batch.m_priority[idx],
                batch.m_visit[idx],
                batch.m_energy[idx],
                batch.m_distance[idx],
                batch.m_time[idx],
                batch.m_dynamics[idx],
                batch.m_remaining[idx],
            )
            distribution = torch.distributions.Categorical(logits=logits)
            logp = distribution.log_prob(batch.action[idx])
            entropy = distribution.entropy().mean()
            policy_loss = -(logp * batch.advantages[idx]).mean()
            value_loss = 0.5 * (
                value.squeeze(-1) - batch.returns[idx]
            ).pow(2).mean()
            total_loss = (
                policy_loss
                + float(self.cfg["value_coef"]) * value_loss
                - float(entropy_coef) * entropy
            )
            if not bool(torch.isfinite(total_loss).item()):
                raise FloatingPointError(
                    "A2C总损失出现NaN/Inf；未执行optimizer.step()。"
                )
            self.opt.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(self.cfg["max_grad_norm"])
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                self.opt.zero_grad(set_to_none=True)
                raise FloatingPointError("A2C梯度范数出现NaN/Inf。")
            self.opt.step()
            aggregates["policy_loss"].append(float(policy_loss.item()))
            aggregates["value_loss"].append(float(value_loss.item()))
            aggregates["entropy"].append(float(entropy.item()))
            pre_clip_norm = float(gradient_norm.item())
            aggregates["gradient_norm_pre_clip"].append(pre_clip_norm)
            aggregates["gradient_norm"].append(
                min(pre_clip_norm, float(self.cfg["max_grad_norm"]))
            )

        with torch.no_grad():
            _, diagnostic_logits, value_after = self.model(
                batch.s_env,
                batch.s_uav,
                batch.m_priority,
                batch.m_visit,
                batch.m_energy,
                batch.m_distance,
                batch.m_time,
                batch.m_dynamics,
                batch.m_remaining,
            )
            diagnostic_logp = torch.distributions.Categorical(
                logits=diagnostic_logits
            ).log_prob(batch.action)
            log_ratio = diagnostic_logp - batch.logp_old
            ratio = torch.exp(log_ratio)
            approx_kl = float((((ratio - 1.0) - log_ratio).mean()).item())
            clip_eps = float(self.cfg["clip_ratio"])
            clip_fraction = float(
                ((ratio - 1.0).abs() > clip_eps).float().mean().item()
            )
            ratio_deviation = float((ratio - 1.0).abs().max().item())
            targets = batch.returns
            residual_variance = torch.var(
                targets - value_after.squeeze(-1), unbiased=False
            )
            target_variance = torch.var(targets, unbiased=False)
            explained_variance = (
                1.0 - residual_variance / target_variance
                if target_variance > 1e-8
                else torch.tensor(0.0, device=targets.device)
            )

        diagnostics = [
            approx_kl,
            clip_fraction,
            ratio_deviation,
            float(explained_variance.item()),
        ]
        if not np.all(np.isfinite(diagnostics)):
            raise FloatingPointError("A2C更新后诊断指标出现NaN/Inf。")
        result = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in aggregates.items()
        }
        result.update(
            {
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "ratio_deviation": ratio_deviation,
                "epochs_completed": 1.0,
                "explained_variance": float(explained_variance.item()),
                "entropy_coef": float(entropy_coef),
                "kl_early_stopped": False,
            }
        )
        return result


def _make_training_agent(
    model: PPO_PtrNet, cfg: Mapping[str, Any]
) -> Union[PPOAgent, A2CAgent]:
    algorithm = str(cfg["training_algorithm"])
    if algorithm == "ppo":
        return PPOAgent(model, cfg)
    if algorithm == "a2c":
        return A2CAgent(model, cfg)
    raise ValueError(f"不支持的training_algorithm={algorithm!r}。")


def _observation_tensors(
    observation: Mapping[str, np.ndarray], target_device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        "s_env": torch.as_tensor(
            observation["s_env"], dtype=torch.float32, device=target_device
        ).unsqueeze(0),
        "s_uav": torch.as_tensor(
            observation["s_uav"], dtype=torch.float32, device=target_device
        ).unsqueeze(0),
        "m_priority": torch.as_tensor(
            observation["m_priority"], dtype=torch.float32, device=target_device
        ).unsqueeze(0),
        "m_visit": torch.as_tensor(
            observation["m_visit"], dtype=torch.bool, device=target_device
        ).unsqueeze(0),
        "m_energy": torch.as_tensor(
            observation["m_energy"], dtype=torch.bool, device=target_device
        ).unsqueeze(0),
        "m_distance": torch.as_tensor(
            observation["m_distance"], dtype=torch.bool, device=target_device
        ).unsqueeze(0),
        "m_time": torch.as_tensor(
            observation["m_time"], dtype=torch.bool, device=target_device
        ).unsqueeze(0),
        "m_dynamics": torch.as_tensor(
            observation["m_dynamics"], dtype=torch.bool, device=target_device
        ).unsqueeze(0),
        "m_remaining": torch.as_tensor(
            observation["m_remaining"], dtype=torch.bool, device=target_device
        ).unsqueeze(0),
    }


def _model_forward(model: PPO_PtrNet, tensors: Mapping[str, torch.Tensor]):
    return model(
        tensors["s_env"],
        tensors["s_uav"],
        tensors["m_priority"],
        tensors["m_visit"],
        tensors["m_energy"],
        tensors["m_distance"],
        tensors["m_time"],
        tensors["m_dynamics"],
        tensors["m_remaining"],
    )


def _episode_metrics(state: Mapping[str, Any], priorities: np.ndarray) -> Dict[str, Any]:
    priorities_arr = np.asarray(priorities, dtype=np.float32).reshape(-1)
    total_priority = float(np.sum(np.clip(priorities_arr, 0.0, None)))
    visited_priority = float(
        np.sum([max(float(priorities_arr[i]), 0.0) for i in state["visited"]])
    )
    weighted_coverage = (
        visited_priority / total_priority
        if total_priority > EPS
        else len(state["visited"]) / max(priorities_arr.size, 1)
    )
    violation_records = list(state.get("constraint_violations", []))
    failed_constraints = {
        str(name)
        for record in violation_records
        if isinstance(record, Mapping)
        for name in record.get("failed_constraints", [])
    }
    # 三个 violation 字段只表示已执行路线的真实超限。仿真消融中被拒绝的
    # 候选动作没有移动无人机，应由 constraint/dynamics 字段记录，不计成实际资源超限。
    energy_violation = bool(
        float(state["total_energy_consumed"]) > float(state["energy_budget_wh"]) + 1e-6
    )
    distance_violation = bool(
        float(state["total_distance"]) > float(state["max_route_distance"]) + 1e-6
    )
    time_violation = bool(
        float(state["total_time_s"]) > float(state["max_mission_time_s"]) + 1e-6
    )
    dynamics_violation = "m_dynamics" in failed_constraints
    return {
        "coverage": len(state["visited"]) / max(priorities_arr.size, 1),
        "weighted_coverage": float(weighted_coverage),
        "visited_count": int(len(state["visited"])),
        "visited_order": list(state["visited"]),
        "energy_wh": float(state["total_energy_consumed"]),
        "distance_m": float(state["total_distance"]),
        "time_s": float(state["total_time_s"]),
        "energy_utilization": float(
            state["total_energy_consumed"] / max(state["energy_budget_wh"], EPS)
        ),
        "distance_utilization": float(
            state["total_distance"] / max(state["max_route_distance"], EPS)
        ),
        "time_utilization": float(
            state["total_time_s"] / max(state["max_mission_time_s"], EPS)
        ),
        "min_remaining_soc": float(state["min_remaining_soc"]),
        "energy_budget_wh": float(state["energy_budget_wh"]),
        "distance_budget_m": float(state["max_route_distance"]),
        "time_budget_s": float(state["max_mission_time_s"]),
        "energy_violation": energy_violation,
        "distance_violation": distance_violation,
        "time_violation": time_violation,
        "dynamics_violation": dynamics_violation,
        "returned": bool(
            state.get("done")
            and state.get("termination_reason") in {"returned_full", "returned_partial"}
        ),
        "termination_reason": state.get("termination_reason"),
        "constraint_violation_count": int(state.get("constraint_violation_count", 0)),
        "constraint_violations": copy.deepcopy(violation_records),
        "experiment_variant": str(state["cfg"]["experiment_variant"]),
        "simulation_only": bool(state["cfg"]["simulation_only"]),
        "wind_fallback_count": int(state["wind_field"].fallback_count),
        "episode_randomization": dict(state["episode_randomization"]),
    }


def rollout_episode_improved(
    model: PPO_PtrNet,
    start_pos: Sequence[float],
    points: np.ndarray,
    priorities: np.ndarray,
    terrain: np.ndarray,
    cfg: Mapping[str, Any],
    wind_data: Optional[Mapping[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
    *,
    decode_mode: str = "stochastic",
    randomize: bool = True,
):
    """采集完整回合；保持旧版六项返回结构。"""

    if decode_mode not in {"stochastic", "deterministic"}:
        raise ValueError("decode_mode 只能是 'stochastic' 或 'deterministic'。")
    state = build_episode(
        start_pos, points, terrain, cfg, wind_data, rng, randomize=randomize
    )
    model_device = next(model.parameters()).device
    actions: List[int] = []
    logps: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    rewards: List[torch.Tensor] = []
    snapshots: List[Dict[str, Any]] = []

    previous_mode = model.training
    model.eval()
    max_steps = int(np.asarray(points).shape[0]) + 1
    try:
        for _step in range(max_steps):
            observation = _build_observation(state, priorities)
            tensors = _observation_tensors(observation, model_device)
            with torch.no_grad():
                _, logits, value = _model_forward(model, tensors)
                distribution = torch.distributions.Categorical(logits=logits)
                if decode_mode == "deterministic":
                    action = torch.argmax(logits, dim=-1)
                else:
                    action = distribution.sample()
                logp = distribution.log_prob(action)

            snapshot = {
                key: tensor.squeeze(0).detach().cpu()
                for key, tensor in tensors.items()
            }
            action_idx = int(action.item())
            state, reward, done = step_env_improved(
                state, action_idx, points, priorities, terrain, cfg, wind_data
            )
            snapshot["terminated"] = bool(done)
            snapshot["episode_metrics"] = _episode_metrics(state, priorities)
            snapshots.append(snapshot)
            actions.append(action_idx)
            logps.append(logp.squeeze(0).detach())
            values.append(value.squeeze(0).squeeze(-1).detach())
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=model_device))
            if done:
                break
        if not state["done"]:
            raise ConstraintViolationError(
                "回合在 N+1 步内未返航；访问掩码或返航动作实现存在错误。"
            )
    finally:
        model.train(previous_mode)

    return (
        [np.asarray(p, dtype=np.float32).tolist() for p in state["path_history"]],
        actions,
        torch.stack(logps),
        torch.stack(values),
        torch.stack(rewards),
        snapshots,
    )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    next_value: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算GAE；终态不bootstrap，防御性截断可传入next_value。"""

    rewards = rewards.reshape(-1)
    values = values.reshape(-1)
    terminated = terminated.to(dtype=torch.bool).reshape(-1)
    if not (rewards.numel() == values.numel() == terminated.numel()):
        raise ValueError("rewards、values、terminated 长度必须一致。")
    if rewards.numel() == 0:
        return rewards.clone(), rewards.clone()
    if not bool(torch.all(torch.isfinite(rewards)).item()) or not bool(
        torch.all(torch.isfinite(values)).item()
    ):
        raise FloatingPointError("GAE输入的reward或value包含NaN/Inf。")
    bootstrap = (
        torch.zeros((), dtype=values.dtype, device=values.device)
        if next_value is None
        else next_value.to(device=values.device, dtype=values.dtype).reshape(())
    )
    if not bool(torch.isfinite(bootstrap).item()):
        raise FloatingPointError("GAE的bootstrap value包含NaN/Inf。")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=values.dtype, device=values.device)
    for index in reversed(range(rewards.numel())):
        next_v = bootstrap if index == rewards.numel() - 1 else values[index + 1]
        nonterminal = (~terminated[index]).to(dtype=values.dtype)
        delta = rewards[index] + float(gamma) * nonterminal * next_v - values[index]
        gae = delta + float(gamma) * float(gae_lambda) * nonterminal * gae
        advantages[index] = gae
    returns = advantages + values
    if not bool(torch.all(torch.isfinite(advantages)).item()) or not bool(
        torch.all(torch.isfinite(returns)).item()
    ):
        raise FloatingPointError("GAE输出的advantage或return包含NaN/Inf。")
    return advantages, returns


def _batch_from_rollouts(
    rollouts: List[Tuple[Any, ...]], cfg: Mapping[str, Any], target_device: torch.device
) -> PPOBatch:
    tensor_lists: Dict[str, List[torch.Tensor]] = {
        key: []
        for key in (
            "s_env",
            "s_uav",
            "m_priority",
            "m_visit",
            "m_energy",
            "m_distance",
            "m_time",
            "m_dynamics",
            "m_remaining",
        )
    }
    actions: List[torch.Tensor] = []
    old_logps: List[torch.Tensor] = []
    old_values: List[torch.Tensor] = []
    all_advantages: List[torch.Tensor] = []
    all_returns: List[torch.Tensor] = []

    for _path, rollout_actions, logps, values, rewards, snapshots in rollouts:
        terminated = torch.tensor(
            [bool(snapshot["terminated"]) for snapshot in snapshots],
            dtype=torch.bool,
            device=values.device,
        )
        advantages, returns = compute_gae(
            rewards,
            values,
            terminated,
            gamma=float(cfg["gamma"]),
            gae_lambda=float(cfg["gae_lambda"]),
        )
        for key in tensor_lists:
            tensor_lists[key].extend(snapshot[key] for snapshot in snapshots)
        actions.append(torch.tensor(rollout_actions, dtype=torch.long))
        old_logps.append(logps.detach().cpu())
        old_values.append(values.detach().cpu())
        all_advantages.append(advantages.detach().cpu())
        all_returns.append(returns.detach().cpu())

    advantages_flat = torch.cat(all_advantages)
    old_logps_flat = torch.cat(old_logps)
    old_values_flat = torch.cat(old_values)
    returns_flat = torch.cat(all_returns)
    for tensor_name, tensor_value in (
        ("logp_old", old_logps_flat),
        ("value_old", old_values_flat),
        ("advantages", advantages_flat),
        ("returns", returns_flat),
    ):
        if not bool(torch.all(torch.isfinite(tensor_value)).item()):
            raise FloatingPointError(f"PPO轨迹批次 {tensor_name} 包含NaN/Inf。")
    advantage_mean = advantages_flat.mean()
    advantage_std = advantages_flat.std(unbiased=False)
    if float(advantage_std.item()) > 1e-8:
        advantages_flat = (advantages_flat - advantage_mean) / advantage_std
    else:
        advantages_flat = advantages_flat - advantage_mean

    batch = PPOBatch(
        s_env=torch.stack(tensor_lists["s_env"]).float(),
        s_uav=torch.stack(tensor_lists["s_uav"]).float(),
        m_priority=torch.stack(tensor_lists["m_priority"]).float(),
        m_visit=torch.stack(tensor_lists["m_visit"]).bool(),
        m_energy=torch.stack(tensor_lists["m_energy"]).bool(),
        m_distance=torch.stack(tensor_lists["m_distance"]).bool(),
        m_time=torch.stack(tensor_lists["m_time"]).bool(),
        m_dynamics=torch.stack(tensor_lists["m_dynamics"]).bool(),
        m_remaining=torch.stack(tensor_lists["m_remaining"]).bool(),
        action=torch.cat(actions),
        logp_old=old_logps_flat,
        value_old=old_values_flat,
        returns=returns_flat,
        advantages=advantages_flat,
    )
    for tensor_name in ("s_env", "s_uav", "m_priority"):
        if not bool(torch.all(torch.isfinite(getattr(batch, tensor_name))).item()):
            raise FloatingPointError(f"PPO轨迹批次 {tensor_name} 包含NaN/Inf。")
    return batch.to(target_device)


def normalize_validation_instances(
    instances: Sequence[Mapping[str, Any]],
    points: np.ndarray,
    priorities: np.ndarray,
    cfg: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    """规范化并哈希冻结验证条件；输入顺序不影响身份。"""

    points_arr = _normalize_points(points)
    priorities_arr = np.asarray(priorities, dtype=np.float32).reshape(-1)
    if priorities_arr.size != points_arr.shape[0]:
        raise ValueError("冻结验证集校验时points与priorities数量不一致。")
    base_service_times = _service_times(points_arr.shape[0], cfg)
    normalized: List[Dict[str, Any]] = []
    seen_ids = set()
    for raw_index, raw_instance in enumerate(instances):
        if not isinstance(raw_instance, Mapping):
            raise TypeError("每个冻结验证实例都必须是映射。")
        instance = dict(raw_instance)
        identifier = str(instance.get("id", "")).strip()
        if not identifier:
            raise ValueError(f"第{raw_index}个冻结验证实例缺少非空id。")
        if identifier in seen_ids:
            raise ValueError(f"冻结验证实例id重复：{identifier}")
        seen_ids.add(identifier)
        missing = [field for field in FROZEN_DOMAIN_FIELDS if field not in instance]
        if missing:
            raise ValueError(f"冻结验证实例{identifier}缺少字段：{', '.join(missing)}")

        record: Dict[str, Any] = {"id": identifier}
        for field_name in FROZEN_DOMAIN_FIELDS:
            value = float(instance[field_name])
            if not math.isfinite(value):
                raise ValueError(f"冻结验证实例{identifier}的{field_name}不是有限数。")
            record[field_name] = value
        reserve_ratio = float(
            cfg.get("battery_reserve_ratio", DEFAULT_CONFIG["battery_reserve_ratio"])
        )
        if not reserve_ratio < record["initial_soc"] <= 1.0:
            raise ValueError(
                f"冻结验证实例{identifier}的initial_soc必须高于"
                f"battery_reserve_ratio={reserve_ratio:.6g}，且不超过1。"
            )
        if min(
            record["distance_budget_scale"],
            record["time_budget_scale"],
            record["wind_scale"],
        ) <= 0.0:
            raise ValueError(f"冻结验证实例{identifier}的比例参数必须大于0。")

        power_scale = float(instance.get("power_scale", 1.0))
        if not math.isclose(power_scale, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("选模验证固定power_scale=1.0；功率敏感性只能在独立测试中进行。")
        record["power_scale"] = 1.0
        node_count = int(instance.get("node_count", points_arr.shape[0]))
        if node_count != points_arr.shape[0]:
            raise ValueError(
                f"冻结验证实例{identifier}的node_count={node_count}，"
                f"但训练场景为{points_arr.shape[0]}点。"
            )
        record["node_count"] = node_count

        if "inspection_points_xyz" in instance:
            frozen_points = _normalize_points(instance["inspection_points_xyz"])
            if frozen_points.shape != points_arr.shape or not np.allclose(
                frozen_points, points_arr, rtol=0.0, atol=1e-6
            ):
                raise ValueError(f"冻结验证实例{identifier}的巡检点与训练场景不一致。")
        if "priorities" in instance:
            frozen_priorities = np.asarray(
                instance["priorities"], dtype=np.float32
            ).reshape(-1)
            if frozen_priorities.shape != priorities_arr.shape or not np.allclose(
                frozen_priorities, priorities_arr, rtol=0.0, atol=1e-6
            ):
                raise ValueError(f"冻结验证实例{identifier}的优先级与训练场景不一致。")

        service_times = np.asarray(
            instance.get("service_times_s", base_service_times), dtype=np.float32
        ).reshape(-1)
        if service_times.size != points_arr.shape[0] or np.any(service_times < 0.0):
            raise ValueError(f"冻结验证实例{identifier}的service_times_s无效。")
        if not np.all(np.isfinite(service_times)):
            raise ValueError(f"冻结验证实例{identifier}的service_times_s包含NaN/Inf。")
        record["service_times_s"] = [float(value) for value in service_times]
        if "instance_seed" in instance:
            record["instance_seed"] = int(instance["instance_seed"])
        normalized.append(record)

    if not normalized:
        raise ValueError("冻结验证实例不能为空。")
    normalized.sort(key=lambda item: item["id"])
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return normalized, digest


def normalize_variable_instances(
    instances: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    require_map_identity: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """规范化可变节点冻结场景；可选地绑定多地图身份。"""

    normalized: List[Dict[str, Any]] = []
    seen_ids = set()
    reserve_ratio = float(
        cfg.get("battery_reserve_ratio", DEFAULT_CONFIG["battery_reserve_ratio"])
    )
    for raw_index, raw_instance in enumerate(instances):
        if not isinstance(raw_instance, Mapping):
            raise TypeError("每个可变节点冻结实例都必须是映射。")
        instance = dict(raw_instance)
        identifier = str(instance.get("id", "")).strip()
        if not identifier:
            raise ValueError(f"第{raw_index}个可变节点冻结实例缺少非空id。")
        if identifier in seen_ids:
            raise ValueError(f"可变节点冻结实例id重复：{identifier}")
        seen_ids.add(identifier)
        missing = [
            field
            for field in (*FROZEN_DOMAIN_FIELDS, "inspection_points_xyz", "priorities")
            if field not in instance
        ]
        if missing:
            raise ValueError(
                f"可变节点冻结实例{identifier}缺少字段：{', '.join(missing)}"
            )
        if require_map_identity:
            map_id = str(instance.get("map_id", "")).strip()
            map_hash = str(instance.get("map_hash", "")).strip()
            if not map_id or len(map_hash) != 64:
                raise ValueError(
                    f"多地图实例{identifier}必须提供map_id和64位map_hash。"
                )

        points_arr = _normalize_points(instance["inspection_points_xyz"])
        priorities_arr = np.asarray(
            instance["priorities"], dtype=np.float32
        ).reshape(-1)
        if (
            priorities_arr.size != points_arr.shape[0]
            or not np.all(np.isfinite(priorities_arr))
        ):
            raise ValueError(f"可变节点冻结实例{identifier}的优先级无效。")
        node_count = int(instance.get("node_count", points_arr.shape[0]))
        if node_count != points_arr.shape[0]:
            raise ValueError(
                f"可变节点冻结实例{identifier}的node_count与点位数量不一致。"
            )

        record: Dict[str, Any] = {
            "id": identifier,
            "node_count": node_count,
            "inspection_points_xyz": points_arr.tolist(),
            "priorities": priorities_arr.tolist(),
        }
        for field_name in ("map_id", "map_hash"):
            if field_name in instance:
                record[field_name] = str(instance[field_name])
        if "start_xy" in instance:
            start_xy = np.asarray(instance["start_xy"], dtype=np.float64).reshape(-1)
            if start_xy.size != 2 or not np.all(np.isfinite(start_xy)):
                raise ValueError(
                    f"多地图实例{identifier}的start_xy必须是两个有限坐标。"
                )
            record["start_xy"] = [float(value) for value in start_xy]
        for field_name in FROZEN_DOMAIN_FIELDS:
            value = float(instance[field_name])
            if not math.isfinite(value):
                raise ValueError(
                    f"可变节点冻结实例{identifier}的{field_name}不是有限数。"
                )
            record[field_name] = value
        if not reserve_ratio < record["initial_soc"] <= 1.0:
            raise ValueError(
                f"可变节点冻结实例{identifier}的initial_soc必须高于"
                f"battery_reserve_ratio={reserve_ratio:.6g}，且不超过1。"
            )
        if min(
            record["distance_budget_scale"],
            record["time_budget_scale"],
            record["wind_scale"],
        ) <= 0.0:
            raise ValueError(
                f"可变节点冻结实例{identifier}的比例参数必须大于0。"
            )
        power_scale = float(instance.get("power_scale", 1.0))
        if not math.isclose(power_scale, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("困难训练和选模固定power_scale=1.0。")
        record["power_scale"] = 1.0

        default_services = _service_times(node_count, cfg)
        service_times = np.asarray(
            instance.get("service_times_s", default_services), dtype=np.float32
        ).reshape(-1)
        if (
            service_times.size != node_count
            or np.any(service_times < 0.0)
            or not np.all(np.isfinite(service_times))
        ):
            raise ValueError(
                f"可变节点冻结实例{identifier}的service_times_s无效。"
            )
        record["service_times_s"] = [
            float(value) for value in service_times
        ]
        if "instance_seed" in instance:
            record["instance_seed"] = int(instance["instance_seed"])
        for field_name in (
            "split",
            "difficulty",
            "constraint_type",
            "priority_layout",
            "replicate_id",
        ):
            if field_name in instance:
                record[field_name] = instance[field_name]
        certificate = instance.get("certificate")
        if isinstance(certificate, Mapping):
            lower = certificate.get("weighted_coverage_lower_bound")
            upper = certificate.get("weighted_coverage_upper_bound")
            if lower is not None and upper is not None:
                lower_value, upper_value = float(lower), float(upper)
                bounds_finite_and_unit = (
                    math.isfinite(lower_value)
                    and math.isfinite(upper_value)
                    and 0.0 <= min(lower_value, upper_value)
                    and max(lower_value, upper_value) <= 1.0
                )
                if (
                    not bounds_finite_and_unit
                    or lower_value - upper_value
                    > ORACLE_BOUND_ORDER_TOLERANCE
                ):
                    raise ValueError(
                        f"可变节点冻结实例{identifier}的oracle上下界无效。"
                    )
                if lower_value > upper_value:
                    # 只处理认证最优时的舍入级反序，保持区间而不改变原始场景记录。
                    lower_value, upper_value = upper_value, lower_value
                record["oracle_weighted_coverage_lower_bound"] = lower_value
                record["oracle_weighted_coverage_upper_bound"] = upper_value
        normalized.append(record)

    if not normalized:
        raise ValueError("可变节点冻结实例不能为空。")
    normalized.sort(key=lambda item: item["id"])
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return normalized, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_instance_map_context(
    instance: Mapping[str, Any],
    *,
    default_start_pos: Sequence[float],
    default_terrain: np.ndarray,
    default_wind_data: Optional[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    scenario_provider: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]],
) -> Tuple[np.ndarray, np.ndarray, Optional[Mapping[str, Any]], Dict[str, Any]]:
    """从只读提供器解析单个地图；旧实验未提供时保持原固定地图。"""

    if scenario_provider is None:
        return (
            np.asarray(default_start_pos, dtype=np.float32),
            np.asarray(default_terrain, dtype=np.float32),
            default_wind_data,
            dict(cfg),
        )
    payload = dict(scenario_provider(instance))
    required = ("start_pos", "terrain", "map_id", "map_hash")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError("地图提供器缺少字段：" + ", ".join(missing))
    if str(payload["map_id"]) != str(instance.get("map_id", "")):
        raise ValueError("地图提供器返回的map_id与冻结实例不一致。")
    if str(payload["map_hash"]) != str(instance.get("map_hash", "")):
        raise ValueError("地图提供器返回的map_hash与冻结实例不一致。")
    start = np.asarray(payload["start_pos"], dtype=np.float32).reshape(-1)
    terrain = np.asarray(payload["terrain"], dtype=np.float32)
    if start.size != 3 or not np.all(np.isfinite(start)):
        raise ValueError("地图提供器返回的start_pos必须是三个有限坐标。")
    if terrain.ndim != 2 or terrain.size == 0 or not np.all(np.isfinite(terrain)):
        raise ValueError("地图提供器返回的terrain必须是有限非空二维DEM。")
    provider_cfg = _deep_update(
        copy.deepcopy(dict(cfg)), dict(payload.get("cfg_overrides") or {})
    )
    provider_cfg = resolve_config(provider_cfg)
    return start, terrain, payload.get("wind_data"), provider_cfg


def _validation_summary(
    model: PPO_PtrNet,
    start_pos: Sequence[float],
    points: np.ndarray,
    priorities: np.ndarray,
    terrain: np.ndarray,
    cfg: Mapping[str, Any],
    wind_data: Optional[Mapping[str, Any]],
    validation_instances: Optional[Sequence[Mapping[str, Any]]] = None,
    scenario_provider: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] = None,
) -> Dict[str, Any]:
    metrics: List[Dict[str, Any]] = []
    failures = 0
    frozen_instances = list(validation_instances or [])
    scenario_count = (
        len(frozen_instances) if frozen_instances else int(cfg["validation_scenarios"])
    )
    for scenario_idx in range(scenario_count):
        validation_rng = np.random.default_rng(int(cfg["seed"]) + 10_000 + scenario_idx)
        scenario_cfg: Mapping[str, Any] = cfg
        scenario_wind = wind_data
        scenario_start = np.asarray(start_pos, dtype=np.float32)
        scenario_terrain = np.asarray(terrain, dtype=np.float32)
        randomize = bool(cfg["domain_randomization"])
        if frozen_instances:
            (
                scenario_start,
                scenario_terrain,
                scenario_wind,
                provider_cfg,
            ) = _resolve_instance_map_context(
                frozen_instances[scenario_idx],
                default_start_pos=start_pos,
                default_terrain=terrain,
                default_wind_data=wind_data,
                cfg=cfg,
                scenario_provider=scenario_provider,
            )
            scenario_cfg, scenario_wind = apply_frozen_domain_instance(
                provider_cfg, scenario_wind, frozen_instances[scenario_idx]
            )
            # 所有学习变体必须面对同一组显式条件，不能受训练随机化开关影响。
            randomize = False
        validation_points = points
        validation_priorities = priorities
        if frozen_instances and "inspection_points_xyz" in frozen_instances[scenario_idx]:
            validation_points = np.asarray(
                frozen_instances[scenario_idx]["inspection_points_xyz"],
                dtype=np.float32,
            )
            validation_priorities = np.asarray(
                frozen_instances[scenario_idx]["priorities"],
                dtype=np.float32,
            )
        try:
            rollout = rollout_episode_improved(
                model,
                scenario_start,
                validation_points,
                validation_priorities,
                scenario_terrain,
                scenario_cfg,
                scenario_wind,
                validation_rng,
                decode_mode="deterministic",
                randomize=randomize,
            )
            episode_metrics = dict(rollout[-1][-1]["episode_metrics"])
            if frozen_instances:
                certificate = dict(
                    frozen_instances[scenario_idx].get("certificate") or {}
                )
                episode_metrics["oracle_weighted_coverage_lower_bound"] = (
                    frozen_instances[scenario_idx].get(
                        "oracle_weighted_coverage_lower_bound",
                        certificate.get("weighted_coverage_lower_bound"),
                    )
                )
                episode_metrics["oracle_weighted_coverage_upper_bound"] = (
                    frozen_instances[scenario_idx].get(
                        "oracle_weighted_coverage_upper_bound",
                        certificate.get("weighted_coverage_upper_bound"),
                    )
                )
            metrics.append(episode_metrics)
        except ConstraintViolationError:
            failures += 1

    if frozen_instances and scenario_provider is not None:
        validation_mode = "external_multimap_v3_1"
    elif frozen_instances:
        validation_mode = "external_fixed_v1"
        for instance in frozen_instances:
            if "inspection_points_xyz" not in instance:
                continue
            instance_points = _normalize_points(instance["inspection_points_xyz"])
            instance_priorities = np.asarray(
                instance.get("priorities", priorities), dtype=np.float32
            ).reshape(-1)
            if (
                instance_points.shape != np.asarray(points).shape
                or instance_priorities.shape != np.asarray(priorities).reshape(-1).shape
                or not np.allclose(
                    instance_points, np.asarray(points), rtol=0.0, atol=1e-6
                )
                or not np.allclose(
                    instance_priorities,
                    np.asarray(priorities).reshape(-1),
                    rtol=0.0,
                    atol=1e-6,
                )
            ):
                validation_mode = "external_variable_v2"
                break
    else:
        validation_mode = "legacy_seeded"
    validation_hash = str(cfg.get("validation_instances_hash", ""))
    if not metrics:
        return {
            "constraint_failures": float(failures),
            "return_rate": 0.0,
            "weighted_coverage": 0.0,
            "coverage": 0.0,
            "resource_utilization": 3.0,
            "validation_mode": validation_mode,
            "validation_instances_hash": validation_hash,
            "validation_instance_count": scenario_count,
        }
    recorded_violations = sum(
        int(m.get("constraint_violation_count", 0)) for m in metrics
    )
    safe_weighted_values = [
        float(m["weighted_coverage"]) if bool(m["returned"]) else 0.0
        for m in metrics
    ]
    oracle_attainment_lower = [
        min(
            1.0,
            safe_value
            / max(
                float(metric["oracle_weighted_coverage_upper_bound"]),
                EPS,
            ),
        )
        for safe_value, metric in zip(safe_weighted_values, metrics)
        if metric.get("oracle_weighted_coverage_upper_bound") is not None
    ]
    return {
        "constraint_failures": float(failures + recorded_violations),
        # 失败场景必须进入分母，不能把失败样本从“返航率”中悄悄排除。
        "return_rate": float(
            sum(float(bool(m["returned"])) for m in metrics) / scenario_count
        ),
        # 抛出约束故障的样本按零覆盖计入分母，诊断值不会因删掉失败样本而虚高。
        "weighted_coverage": float(
            sum(float(m["weighted_coverage"]) for m in metrics) / scenario_count
        ),
        "coverage": float(sum(float(m["coverage"]) for m in metrics) / scenario_count),
        "safe_weighted_coverage": float(
            sum(safe_weighted_values) / scenario_count
        ),
        "zero_visit_rate": float(
            (
                sum(int(m.get("visited_count", 0)) == 0 for m in metrics)
                + failures
            )
            / scenario_count
        ),
        "partial_return_rate": float(
            sum(
                bool(m.get("returned"))
                and str(m.get("termination_reason")) == "returned_partial"
                for m in metrics
            )
            / scenario_count
        ),
        "median_visited_count": float(
            statistics.median(
                [int(m.get("visited_count", 0)) for m in metrics]
                + [0] * failures
            )
        ),
        "median_oracle_attainment_lower": (
            float(statistics.median(oracle_attainment_lower))
            if oracle_attainment_lower
            else None
        ),
        "resource_utilization": float(
            (
                sum(
                    float(m["energy_utilization"])
                    + float(m["distance_utilization"])
                    + float(m["time_utilization"])
                    for m in metrics
                )
                + 3.0 * failures
            )
            / scenario_count
        ),
        "validation_mode": validation_mode,
        "validation_instances_hash": validation_hash,
        "validation_instance_count": scenario_count,
    }


def _validation_key(summary: Mapping[str, float]) -> Tuple[float, ...]:
    """按安全、加权覆盖、普通覆盖、资源消耗的字典序选择最佳模型。"""

    return (
        -float(summary["constraint_failures"]),
        float(summary["return_rate"]),
        float(summary["weighted_coverage"]),
        float(summary["coverage"]),
        -float(summary["resource_utilization"]),
    )


def save_checkpoint(
    checkpoint_path: Union[os.PathLike, str],
    model: PPO_PtrNet,
    cfg: Mapping[str, Any],
    *,
    episode_returns: Optional[Sequence[float]] = None,
    optimizer_state_dict: Optional[Mapping[str, Any]] = None,
    training_summary: Optional[Mapping[str, Any]] = None,
    training_state: Optional[Mapping[str, Any]] = None,
    checkpoint_kind: str = "generic",
) -> Path:
    """原子保存schema v2检查点，避免中断留下半写入文件。"""

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(training_summary or {})
    resumable = copy.deepcopy(dict(training_state or {}))
    environment_interactions = int(
        resumable.get(
            "environment_interactions", summary.get("environment_interactions", 0)
        )
    )
    interaction_count_complete = bool(
        resumable.get(
            "interaction_count_complete",
            summary.get("interaction_count_complete", True),
        )
    )
    experiment_metadata = _experiment_metadata(
        model,
        cfg,
        environment_interactions=environment_interactions,
        interaction_count_complete=interaction_count_complete,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": str(checkpoint_kind),
        "feature_schema": {"d_env": NODE_FEATURE_DIM, "d_uav": UAV_FEATURE_DIM},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state_dict,
        "returns": list(episode_returns or []),
        "cfg": copy.deepcopy(dict(cfg)),
        "n_nodes": int(model.N),
        "d_model": int(model.d_model),
        "n_heads": int(model.n_heads),
        "lambda_priority": float(model.lambda_priority),
        "seed": int(cfg["seed"]),
        "training_summary": summary,
        "training_state": resumable,
        "experiment_metadata": experiment_metadata,
        "experiment_variant": experiment_metadata["variant"],
        "policy_architecture": experiment_metadata["policy_architecture"],
        "training_algorithm": experiment_metadata["training_algorithm"],
        "parameter_count": experiment_metadata["parameter_count"],
        "environment_interactions": experiment_metadata["environment_interactions"],
        "interaction_count_complete": experiment_metadata[
            "interaction_count_complete"
        ],
        # 保存完整随机状态，便于后续从检查点继续训练并复现实验。
        "rng_state": {
            "python": random.getstate(),
            "numpy_global": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "training_generator": copy.deepcopy(
                getattr(model, "training_rng_state", None)
            ),
        },
    }
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path


def load_checkpoint(
    checkpoint_path: Union[os.PathLike, str],
    *,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Tuple[PPO_PtrNet, Dict[str, Any]]:
    target_location = map_location or device
    try:
        payload = torch.load(
            checkpoint_path, map_location=target_location, weights_only=False
        )
    except TypeError:  # 兼容尚未提供 weights_only 参数的旧版PyTorch
        payload = torch.load(checkpoint_path, map_location=target_location)
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            "检查点不是schema_version=2，旧版5/7维特征权重不能静默加载。"
        )
    schema = payload.get("feature_schema", {})
    if int(schema.get("d_env", -1)) != NODE_FEATURE_DIM or int(
        schema.get("d_uav", -1)
    ) != UAV_FEATURE_DIM:
        raise ValueError("检查点特征维度与当前v2模型不一致。")
    stored_metadata = payload.get("experiment_metadata")
    metadata = dict(stored_metadata) if isinstance(stored_metadata, Mapping) else {}
    variant_name = str(
        payload.get("experiment_variant", metadata.get("variant", "full"))
    )
    variant = get_experiment_variant(variant_name)
    policy_architecture = str(
        payload.get(
            "policy_architecture",
            metadata.get("policy_architecture", variant.policy_architecture),
        )
    )
    training_algorithm = str(
        payload.get(
            "training_algorithm",
            metadata.get("training_algorithm", variant.training_algorithm),
        )
    )
    if policy_architecture != variant.policy_architecture:
        raise ValueError("检查点的变体与policy_architecture元数据不一致。")
    if training_algorithm != variant.training_algorithm:
        raise ValueError("检查点的变体与training_algorithm元数据不一致。")
    if variant.lambda_priority_override is not None and not math.isclose(
        float(payload["lambda_priority"]),
        float(variant.lambda_priority_override),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("检查点的变体与lambda_priority元数据不一致。")
    model = PPO_PtrNet(
        batch_size=1,
        n_nodes=int(payload["n_nodes"]),
        d_env=NODE_FEATURE_DIM,
        d_uav=UAV_FEATURE_DIM,
        d_model=int(payload["d_model"]),
        n_heads=int(payload["n_heads"]),
        lambda_priority=float(payload["lambda_priority"]),
        policy_architecture=policy_architecture,
        training_algorithm=training_algorithm,
        experiment_variant=variant_name,
    ).to(target_location)
    model.load_state_dict(payload["model_state_dict"])
    actual_parameter_count = count_trainable_parameters(model)
    stored_parameter_count = payload.get(
        "parameter_count", metadata.get("parameter_count")
    )
    if stored_parameter_count is not None and int(stored_parameter_count) != actual_parameter_count:
        raise ValueError("检查点记录的参数量与重建模型不一致。")
    has_interaction_count = (
        "environment_interactions" in payload
        or "environment_interactions" in metadata
    )
    environment_interactions = int(
        payload.get(
            "environment_interactions", metadata.get("environment_interactions", 0)
        )
    )
    interaction_count_complete = bool(
        payload.get(
            "interaction_count_complete",
            metadata.get("interaction_count_complete", has_interaction_count),
        )
    )
    # 旧full schema v2没有实验元数据；内存中补齐，但不改写原文件。
    payload["experiment_metadata"] = _experiment_metadata(
        model,
        dict(payload.get("cfg") or {}),
        environment_interactions=environment_interactions,
        interaction_count_complete=interaction_count_complete,
    )
    payload.setdefault("experiment_variant", variant_name)
    payload.setdefault("policy_architecture", policy_architecture)
    payload.setdefault("training_algorithm", training_algorithm)
    payload.setdefault("parameter_count", actual_parameter_count)
    payload.setdefault("environment_interactions", environment_interactions)
    payload.setdefault("interaction_count_complete", interaction_count_complete)
    model.eval()
    return model, payload


def _resume_config_mismatches(
    stored_cfg: Mapping[str, Any], current_cfg: Mapping[str, Any]
) -> List[str]:
    """返回不允许在续训时变化的配置项。

    ``max_episodes`` 是累计训练目标，允许向后延长；``checkpoint_dir`` 只决定
    新检查点落盘位置。其余训练、动力学、奖励和域随机化参数一旦变化，就不再是
    对同一实验的严格续训，必须显式拒绝而不是悄悄混合两套配置。
    """

    allowed_changes = {"max_episodes", "checkpoint_dir"}

    def values_equal(left: Any, right: Any) -> bool:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if set(left) != set(right):
                return False
            return all(values_equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple, np.ndarray)) or isinstance(
            right, (list, tuple, np.ndarray)
        ):
            try:
                return bool(
                    np.array_equal(
                        np.asarray(left), np.asarray(right), equal_nan=True
                    )
                )
            except (TypeError, ValueError):
                return False
        if isinstance(left, (float, np.floating)) or isinstance(
            right, (float, np.floating)
        ):
            try:
                return math.isclose(
                    float(left), float(right), rel_tol=0.0, abs_tol=1e-12
                )
            except (TypeError, ValueError):
                return False
        return left == right

    mismatches: List[str] = []
    for key in sorted((set(stored_cfg) | set(current_cfg)) - allowed_changes):
        if key not in stored_cfg or key not in current_cfg:
            mismatches.append(key)
        elif not values_equal(stored_cfg[key], current_cfg[key]):
            mismatches.append(key)
    return mismatches


def train_policy_improved(
    start_pos: Sequence[float],
    points: np.ndarray,
    priorities: np.ndarray,
    terrain: np.ndarray,
    cfg: Mapping[str, Any],
    wind_data: Optional[Mapping[str, Any]] = None,
    *,
    resume_from: Optional[Union[os.PathLike, str]] = None,
    metrics_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
    target_device: Optional[Union[str, torch.device]] = None,
    validation_instances: Optional[Sequence[Mapping[str, Any]]] = None,
    training_instances: Optional[Sequence[Mapping[str, Any]]] = None,
    scenario_provider: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] = None,
) -> Tuple[PPO_PtrNet, List[float]]:
    """训练或恢复注册变体；max_episodes始终表示累计目标回合数。"""

    config = resolve_config(cfg)
    if scenario_provider is not None:
        provider_hash = str(getattr(scenario_provider, "provider_hash", "")).strip()
        if len(provider_hash) != 64:
            raise ValueError("多地图scenario_provider必须公开64位provider_hash。")
        supplied_provider_hash = str(config.get("scenario_provider_hash", ""))
        if supplied_provider_hash and supplied_provider_hash != provider_hash:
            raise ValueError("配置中的scenario_provider_hash与地图提供器不一致。")
        config["scenario_provider_hash"] = provider_hash
        config["scenario_mode"] = "frozen_multimap_v3_1"
    else:
        config["scenario_mode"] = "legacy_fixed_map"
    training_device = _resolve_training_device(target_device)
    points_arr = _normalize_points(points)
    priorities_arr = np.asarray(priorities, dtype=np.float32).reshape(-1)
    if priorities_arr.size != points_arr.shape[0]:
        raise ValueError("priorities 长度必须与巡检点数量一致。")
    if not np.all(np.isfinite(priorities_arr)):
        raise ValueError("priorities 包含 NaN 或 Inf。")

    requested_validation = (
        validation_instances
        if validation_instances is not None
        else config.get("validation_instances")
    )
    normalized_validation: Optional[List[Dict[str, Any]]] = None
    if requested_validation is not None:
        variable_validation = False
        for instance in requested_validation:
            if "inspection_points_xyz" not in instance:
                continue
            frozen_points = _normalize_points(instance["inspection_points_xyz"])
            frozen_priorities = np.asarray(
                instance.get("priorities", priorities_arr), dtype=np.float32
            ).reshape(-1)
            if (
                frozen_points.shape != points_arr.shape
                or frozen_priorities.shape != priorities_arr.shape
                or not np.allclose(
                    frozen_points, points_arr, rtol=0.0, atol=1e-6
                )
                or not np.allclose(
                    frozen_priorities, priorities_arr, rtol=0.0, atol=1e-6
                )
            ):
                variable_validation = True
                break
        if variable_validation:
            if not all(
                "inspection_points_xyz" in instance
                and "priorities" in instance
                for instance in requested_validation
            ):
                raise ValueError("可变节点validation必须为每个实例提供点位和优先级。")
            normalized_validation, validation_hash = normalize_variable_instances(
                requested_validation,
                config,
                require_map_identity=scenario_provider is not None,
            )
        else:
            normalized_validation, validation_hash = normalize_validation_instances(
                requested_validation, points_arr, priorities_arr, config
            )
        supplied_hash = str(config.get("validation_instances_hash", ""))
        if supplied_hash and supplied_hash != validation_hash:
            raise ValueError(
                "冻结验证实例内容与validation_instances_hash不一致；"
                "拒绝在不同验证集上续训或选模。"
            )
        # 旧固定点验证保留内嵌内容；可变节点清单只保存哈希，避免每个检查点重复存档。
        config["validation_instances"] = (
            None
            if variable_validation
            else copy.deepcopy(normalized_validation)
        )
        config["validation_instances_hash"] = validation_hash
        config["validation_scenarios"] = len(normalized_validation)
        config["validation_mode"] = (
            "external_multimap_v3_1"
            if scenario_provider is not None
            else "external_variable_v2"
            if variable_validation
            else "external_fixed_v1"
        )
    elif config.get("validation_instances_hash"):
        raise ValueError("配置声明了validation_instances_hash，但没有提供冻结验证实例。")
    else:
        config["validation_mode"] = "legacy_seeded"

    requested_training = (
        training_instances
        if training_instances is not None
        else config.get("training_instances")
    )
    normalized_training: Optional[List[Dict[str, Any]]] = None
    training_groups: Dict[int, List[Dict[str, Any]]] = {}
    if requested_training is not None:
        normalized_training, training_hash = normalize_variable_instances(
            requested_training,
            config,
            require_map_identity=scenario_provider is not None,
        )
        supplied_training_hash = str(config.get("training_instances_hash", ""))
        if supplied_training_hash and supplied_training_hash != training_hash:
            raise ValueError(
                "困难训练池内容与training_instances_hash不一致；拒绝混合训练分布。"
            )
        for record in normalized_training:
            training_groups.setdefault(int(record["node_count"]), []).append(record)
        config["training_mode"] = (
            "external_multimap_pool_v3_1"
            if scenario_provider is not None
            else "external_variable_pool_v2"
        )
        config["training_instances_hash"] = training_hash
        config["training_instance_count"] = len(normalized_training)
        config["training_node_counts"] = sorted(training_groups)
        config.pop("training_instances", None)
    elif config.get("training_instances_hash"):
        raise ValueError("配置声明了training_instances_hash，但没有提供困难训练池。")
    else:
        config["training_mode"] = "legacy_seeded"
        config["training_instance_count"] = 0
        config["training_node_counts"] = []

    episode_returns: List[float] = []
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_validation: Optional[Dict[str, float]] = None
    best_key: Optional[Tuple[float, ...]] = None
    best_optimizer_state: Optional[Dict[str, Any]] = None
    best_candidate_validation: Optional[Dict[str, float]] = None
    best_candidate_key: Optional[Tuple[float, ...]] = None
    best_candidate_state: Optional[Dict[str, torch.Tensor]] = None
    best_candidate_optimizer_state: Optional[Dict[str, Any]] = None
    best_candidate_episode = 0
    best_episode = 0
    episodes_seen = 0
    update_index = 0
    environment_interactions = 0
    interaction_count_complete = True
    entropy_progress = 0.0
    max_episodes = int(config["max_episodes"])
    model_node_count = (
        max(training_groups) if training_groups else points_arr.shape[0]
    )
    training_sampler_state: Dict[str, Any] = {}

    if resume_from is None:
        set_global_seed(int(config["seed"]))
        model = PPO_PtrNet(
            batch_size=1,
            n_nodes=model_node_count,
            d_env=NODE_FEATURE_DIM,
            d_uav=UAV_FEATURE_DIM,
            d_model=int(config["d_model"]),
            n_heads=int(config["n_heads"]),
            lambda_priority=float(config["lambda_priority"]),
            policy_architecture=str(config["policy_architecture"]),
            training_algorithm=str(config["training_algorithm"]),
            experiment_variant=str(config["experiment_variant"]),
        ).to(training_device)
        agent = _make_training_agent(model, config)
        rng = np.random.default_rng(int(config["seed"]))
        if training_groups:
            training_sampler_state = {
                "node_counts": sorted(training_groups),
                "groups": {
                    str(node_count): {
                        "order": rng.permutation(
                            len(training_groups[node_count])
                        ).tolist(),
                        "cursor": 0,
                    }
                    for node_count in sorted(training_groups)
                },
            }
    else:
        model, resume_payload = load_checkpoint(
            resume_from, map_location=training_device
        )
        if str(resume_payload.get("checkpoint_kind", "")) != "latest":
            raise ValueError(
                "恢复训练只接受checkpoint_kind=latest的断点；"
                "best_safe仅用于确定性推断/评估。"
            )
        if int(model.N) != model_node_count:
            raise ValueError(
                f"恢复检查点的节点容量身份为{model.N}，当前训练协议为"
                f"{model_node_count}。"
            )
        architecture = {
            "d_model": int(model.d_model),
            "n_heads": int(model.n_heads),
            "lambda_priority": float(model.lambda_priority),
            "policy_architecture": str(model.policy_architecture),
            "training_algorithm": str(model.training_algorithm),
            "experiment_variant": str(model.experiment_variant),
        }
        requested_architecture = {
            "d_model": int(config["d_model"]),
            "n_heads": int(config["n_heads"]),
            "lambda_priority": float(config["lambda_priority"]),
            "policy_architecture": str(config["policy_architecture"]),
            "training_algorithm": str(config["training_algorithm"]),
            "experiment_variant": str(config["experiment_variant"]),
        }
        if architecture != requested_architecture:
            raise ValueError(
                "恢复检查点的网络结构与当前配置不一致："
                f"checkpoint={architecture}, cfg={requested_architecture}。"
            )
        if int(resume_payload.get("seed", -1)) != int(config["seed"]):
            raise ValueError("恢复训练必须沿用检查点中的seed。")
        stored_cfg = dict(resume_payload.get("cfg") or {})
        stored_max_episodes = int(stored_cfg.get("max_episodes", max_episodes))
        if max_episodes < stored_max_episodes:
            raise ValueError(
                "续训累计目标不能缩短原实验计划："
                f"checkpoint={stored_max_episodes}, requested={max_episodes}。"
            )
        # 旧full schema v2缺少新增实验字段时按full默认值补齐后再比较。
        stored_cfg_for_compare = resolve_config(stored_cfg)
        config_mismatches = _resume_config_mismatches(
            stored_cfg_for_compare, config
        )
        if config_mismatches:
            raise ValueError(
                "恢复训练不允许改变除max_episodes/checkpoint_dir以外的配置；"
                "不一致项：" + ", ".join(config_mismatches)
            )
        optimizer_state = resume_payload.get("optimizer_state_dict")
        if optimizer_state is None:
            raise ValueError("该检查点没有optimizer_state_dict，不能用于恢复训练。")

        agent = _make_training_agent(model, config)
        agent.opt.load_state_dict(optimizer_state)
        summary = dict(resume_payload.get("training_summary") or {})
        resumable_state = dict(resume_payload.get("training_state") or {})
        episode_returns = [float(value) for value in resume_payload.get("returns", [])]
        episodes_seen = int(
            resumable_state.get(
                "episodes_seen", summary.get("episodes_seen", len(episode_returns))
            )
        )
        if episodes_seen != len(episode_returns):
            raise ValueError(
                "恢复检查点的episodes_seen与回报历史长度不一致，文件可能不完整。"
            )
        if max_episodes < episodes_seen:
            raise ValueError(
                f"max_episodes={max_episodes}小于检查点已完成回合数{episodes_seen}。"
            )
        history = copy.deepcopy(
            resumable_state.get("history", summary.get("history", []))
        )
        update_index = int(
            resumable_state.get("update_index", summary.get("updates", len(history)))
        )
        environment_interactions = int(
            resumable_state.get(
                "environment_interactions",
                summary.get(
                    "environment_interactions",
                    resume_payload.get("environment_interactions", 0),
                ),
            )
        )
        interaction_count_complete = bool(
            resumable_state.get(
                "interaction_count_complete",
                summary.get(
                    "interaction_count_complete",
                    resume_payload.get("interaction_count_complete", False),
                ),
            )
        )
        best_state = copy.deepcopy(resumable_state.get("best_model_state_dict"))
        best_optimizer_state = copy.deepcopy(
            resumable_state.get("best_optimizer_state_dict")
        )
        best_validation = copy.deepcopy(
            resumable_state.get("best_validation", summary.get("best_validation"))
        )
        stored_best_key = resumable_state.get("best_key")
        best_key = tuple(stored_best_key) if stored_best_key is not None else None
        best_candidate_validation = copy.deepcopy(
            resumable_state.get("best_candidate_validation")
        )
        stored_candidate_key = resumable_state.get("best_candidate_key")
        best_candidate_key = (
            tuple(stored_candidate_key) if stored_candidate_key is not None else None
        )
        best_candidate_state = copy.deepcopy(
            resumable_state.get("best_candidate_model_state_dict")
        )
        best_candidate_optimizer_state = copy.deepcopy(
            resumable_state.get("best_candidate_optimizer_state_dict")
        )
        best_candidate_episode = int(
            resumable_state.get("best_candidate_episode", 0)
        )
        best_episode = int(
            resumable_state.get("best_episode", summary.get("best_episode", 0))
        )

        # 兼容增强前生成的v2 best_safe检查点；latest仍会在下一次验证重建最佳状态。
        if best_state is None and best_validation is not None:
            is_safe = (
                float(best_validation.get("constraint_failures", 1.0)) == 0.0
                and float(best_validation.get("return_rate", 0.0)) >= 1.0 - 1e-12
            )
            if is_safe:
                best_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(agent.opt.state_dict())
                best_key = _validation_key(best_validation)
        if best_candidate_state is None and best_candidate_validation is not None:
            best_candidate_state = copy.deepcopy(model.state_dict())
            best_candidate_optimizer_state = copy.deepcopy(agent.opt.state_dict())
            best_candidate_episode = episodes_seen

        inferred_entropy_progress = episodes_seen / max(stored_max_episodes - 1, 1)
        entropy_progress = float(
            resumable_state.get("entropy_progress", inferred_entropy_progress)
        )
        rng_state = resume_payload.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ValueError("恢复检查点缺少完整rng_state。")
        # 必须在模型和优化器构造完成后恢复，抵消构造过程对全局RNG的消耗。
        rng = _restore_rng_state(rng_state)
        if training_groups:
            raw_sampler = resumable_state.get("training_sampler_state")
            if not isinstance(raw_sampler, Mapping):
                raise ValueError("困难训练断点缺少training_sampler_state。")
            training_sampler_state = copy.deepcopy(dict(raw_sampler))
            if list(training_sampler_state.get("node_counts", ())) != sorted(
                training_groups
            ):
                raise ValueError("困难训练断点的节点分层与当前训练池不一致。")
            stored_groups = dict(training_sampler_state.get("groups") or {})
            for node_count, records in training_groups.items():
                state = dict(stored_groups.get(str(node_count)) or {})
                order = [int(value) for value in state.get("order", ())]
                cursor = int(state.get("cursor", -1))
                if (
                    sorted(order) != list(range(len(records)))
                    or cursor < 0
                    or cursor > len(order)
                ):
                    raise ValueError(
                        f"困难训练断点的{node_count}节点采样状态无效。"
                    )
        logger.info(
            "已从 %s 恢复：回合 %d/%d，PPO更新 %d。",
            resume_from,
            episodes_seen,
            max_episodes,
            update_index,
        )

    checkpoint_dir = config.get("checkpoint_dir")

    def _snapshot_training_state() -> Dict[str, Any]:
        return {
            "episodes_seen": episodes_seen,
            "update_index": update_index,
            "environment_interactions": environment_interactions,
            "interaction_count_complete": interaction_count_complete,
            "history": copy.deepcopy(history),
            "entropy_progress": float(entropy_progress),
            "best_episode": best_episode,
            "best_validation": copy.deepcopy(best_validation),
            "best_key": copy.deepcopy(best_key),
            "best_model_state_dict": copy.deepcopy(best_state),
            "best_optimizer_state_dict": copy.deepcopy(best_optimizer_state),
            "best_candidate_validation": copy.deepcopy(best_candidate_validation),
            "best_candidate_key": copy.deepcopy(best_candidate_key),
            "best_candidate_model_state_dict": copy.deepcopy(best_candidate_state),
            "best_candidate_optimizer_state_dict": copy.deepcopy(
                best_candidate_optimizer_state
            ),
            "best_candidate_episode": best_candidate_episode,
            "validation_mode": str(config["validation_mode"]),
            "validation_instances_hash": str(
                config.get("validation_instances_hash", "")
            ),
            "validation_instance_count": int(config["validation_scenarios"]),
            "training_mode": str(config["training_mode"]),
            "training_instances_hash": str(
                config.get("training_instances_hash", "")
            ),
            "training_instance_count": int(
                config.get("training_instance_count", 0)
            ),
            "training_sampler_state": copy.deepcopy(training_sampler_state),
        }

    def _draw_training_record(node_count: int) -> Dict[str, Any]:
        group = training_groups[int(node_count)]
        group_state = training_sampler_state["groups"][str(node_count)]
        order = [int(value) for value in group_state["order"]]
        cursor = int(group_state["cursor"])
        if cursor >= len(order):
            order = rng.permutation(len(group)).tolist()
            cursor = 0
            group_state["order"] = order
        record = group[order[cursor]]
        group_state["cursor"] = cursor + 1
        return record

    monitor_episodes = sorted(
        {int(value) for value in config.get("monitor_episodes", ())}
    )
    if any(value <= 0 or value > max_episodes for value in monitor_episodes):
        raise ValueError("monitor_episodes必须位于(0, max_episodes]。")

    while episodes_seen < max_episodes:
        batch_episodes = min(
            int(config["episodes_per_update"]), max_episodes - episodes_seen
        )
        next_monitor = next(
            (value for value in monitor_episodes if value > episodes_seen),
            None,
        )
        if next_monitor is not None:
            batch_episodes = min(batch_episodes, next_monitor - episodes_seen)
        rollouts: List[Tuple[Any, ...]] = []
        batch_metrics: List[Dict[str, Any]] = []
        batch_scenario_ids: List[str] = []
        batch_node_count: Optional[int] = None
        if training_groups:
            node_counts = sorted(training_groups)
            batch_node_count = int(node_counts[update_index % len(node_counts)])
        for _ in range(batch_episodes):
            training_points = points_arr
            training_priorities = priorities_arr
            training_cfg: Mapping[str, Any] = config
            training_wind = wind_data
            training_start = np.asarray(start_pos, dtype=np.float32)
            training_terrain = np.asarray(terrain, dtype=np.float32)
            training_randomize = bool(config["domain_randomization"])
            if training_groups:
                record = _draw_training_record(int(batch_node_count))
                training_points = np.asarray(
                    record["inspection_points_xyz"], dtype=np.float32
                )
                training_priorities = np.asarray(
                    record["priorities"], dtype=np.float32
                )
                batch_scenario_ids.append(str(record["id"]))
                (
                    training_start,
                    training_terrain,
                    training_wind,
                    provider_cfg,
                ) = _resolve_instance_map_context(
                    record,
                    default_start_pos=start_pos,
                    default_terrain=terrain,
                    default_wind_data=wind_data,
                    cfg=config,
                    scenario_provider=scenario_provider,
                )
                if bool(config["domain_randomization"]):
                    training_cfg, training_wind = apply_frozen_domain_instance(
                        provider_cfg, training_wind, record
                    )
                else:
                    # 域随机化消融仍使用同一几何池，但不应用SOC、预算和风扰动。
                    training_cfg = copy.deepcopy(dict(provider_cfg))
                    training_cfg["service_times_s"] = list(
                        record["service_times_s"]
                    )
                    training_wind = wind_data
                training_randomize = False
            rollout = rollout_episode_improved(
                model,
                training_start,
                training_points,
                training_priorities,
                training_terrain,
                training_cfg,
                training_wind,
                rng,
                decode_mode="stochastic",
                randomize=training_randomize,
            )
            rollouts.append(rollout)
            episode_return = float(rollout[4].sum().item())
            episode_returns.append(episode_return)
            batch_metrics.append(rollout[-1][-1]["episode_metrics"])

        batch_interactions = int(sum(len(rollout[1]) for rollout in rollouts))
        environment_interactions += batch_interactions
        batch = _batch_from_rollouts(rollouts, config, training_device)
        scheduled_progress = episodes_seen / max(max_episodes - 1, 1)
        # 更新按完整回合批次进行；最后一个批次必须精确使用entropy_coef_end，
        # 不能因为计算时尚未累加本批回合而停在终点之前。
        if episodes_seen + batch_episodes >= max_episodes:
            scheduled_progress = 1.0
        # 扩展总回合数时熵系数不能反弹；恢复后只沿既有衰减进度继续向下。
        entropy_progress = min(1.0, max(entropy_progress, scheduled_progress))
        entropy_coef = float(config["entropy_coef_start"]) + entropy_progress * (
            float(config["entropy_coef_end"]) - float(config["entropy_coef_start"])
        )
        update_stats = agent.update(batch, entropy_coef)
        episodes_seen += batch_episodes
        update_index += 1
        update_stats.update(
            {
                "update": float(update_index),
                "episodes_seen": float(episodes_seen),
                "environment_interactions": int(environment_interactions),
                "batch_environment_interactions": batch_interactions,
                "training_node_count": batch_node_count,
                "training_scenario_ids": batch_scenario_ids,
                "interaction_count_complete": bool(interaction_count_complete),
                "experiment": _experiment_metadata(
                    model,
                    config,
                    environment_interactions,
                    interaction_count_complete,
                ),
                "mean_return": float(np.mean(episode_returns[-batch_episodes:])),
                "mean_coverage": float(np.mean([m["coverage"] for m in batch_metrics])),
                "mean_weighted_coverage": float(
                    np.mean([m["weighted_coverage"] for m in batch_metrics])
                ),
                "return_rate": float(np.mean([m["returned"] for m in batch_metrics])),
                "mean_energy_utilization": float(
                    np.mean([m["energy_utilization"] for m in batch_metrics])
                ),
                "mean_distance_utilization": float(
                    np.mean([m["distance_utilization"] for m in batch_metrics])
                ),
                "mean_time_utilization": float(
                    np.mean([m["time_utilization"] for m in batch_metrics])
                ),
                "termination_reason_counts": {
                    reason: int(
                        sum(m.get("termination_reason") == reason for m in batch_metrics)
                    )
                    for reason in (
                        "returned_full",
                        "returned_partial",
                        "constraint_failure",
                        "stranded",
                    )
                },
            }
        )

        should_validate = (
            update_index % int(config["validation_interval_updates"]) == 0
            or episodes_seen == max_episodes
            or episodes_seen in monitor_episodes
            # 尚未得到任何候选时提前验证一次；仿真安全消融可能始终
            # 没有best_safe，但不应因此绕过validation_interval_updates每轮重复验证。
            or best_candidate_state is None
        )
        validation: Optional[Dict[str, float]] = None
        is_new_best_safe = False
        if should_validate:
            validation = _validation_summary(
                model,
                start_pos,
                points_arr,
                priorities_arr,
                terrain,
                config,
                wind_data,
                normalized_validation,
                scenario_provider,
            )
            candidate_key = _validation_key(validation)
            if best_candidate_key is None or candidate_key > best_candidate_key:
                best_candidate_key = candidate_key
                best_candidate_validation = copy.deepcopy(validation)
                best_candidate_state = copy.deepcopy(model.state_dict())
                best_candidate_optimizer_state = copy.deepcopy(agent.opt.state_dict())
                best_candidate_episode = episodes_seen
                is_new_best_candidate = True
            else:
                is_new_best_candidate = False

            is_safe_candidate = (
                float(validation["constraint_failures"]) == 0.0
                and float(validation["return_rate"]) >= 1.0 - 1e-12
            )
            if is_safe_candidate and (best_key is None or candidate_key > best_key):
                best_key = candidate_key
                best_validation = validation
                best_episode = episodes_seen
                best_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(agent.opt.state_dict())
                is_new_best_safe = True

        update_stats["validation"] = copy.deepcopy(validation)
        update_stats["is_best_safe"] = bool(is_new_best_safe)
        update_stats["is_best_candidate"] = bool(
            should_validate and is_new_best_candidate
        )
        update_stats["best_episode"] = float(best_episode)
        update_stats["best_validation"] = copy.deepcopy(best_validation)
        history.append(copy.deepcopy(update_stats))
        logger.info(
            "更新 %d | 回合 %d/%d | 回报 %.4f | 加权覆盖 %.3f | "
            "%s KL %.5f | clip %.3f | entropy %.3f",
            update_index,
            episodes_seen,
            max_episodes,
            update_stats["mean_return"],
            update_stats["mean_weighted_coverage"],
            str(config["training_algorithm"]).upper(),
            update_stats["approx_kl"],
            update_stats["clip_fraction"],
            update_stats["entropy"],
        )

        model.training_rng_state = copy.deepcopy(  # type: ignore[attr-defined]
            rng.bit_generator.state
        )
        current_summary = {
            "schema_version": SCHEMA_VERSION,
            "episodes_seen": episodes_seen,
            "updates": update_index,
            "environment_interactions": environment_interactions,
            "interaction_count_complete": interaction_count_complete,
            "experiment": _experiment_metadata(
                model,
                config,
                environment_interactions,
                interaction_count_complete,
            ),
            "best_episode": best_episode,
            "best_validation": copy.deepcopy(best_validation) or {},
            "validation_mode": str(config["validation_mode"]),
            "validation_instances_hash": str(
                config.get("validation_instances_hash", "")
            ),
            "validation_instance_count": int(config["validation_scenarios"]),
            "training_mode": str(config["training_mode"]),
            "training_instances_hash": str(
                config.get("training_instances_hash", "")
            ),
            "training_instance_count": int(
                config.get("training_instance_count", 0)
            ),
            "entropy_progress": float(entropy_progress),
            "history": copy.deepcopy(history),
        }
        training_state = _snapshot_training_state()
        latest_path: Optional[Path] = None
        best_safe_path: Optional[Path] = None
        best_candidate_path: Optional[Path] = None
        monitor_path: Optional[Path] = None
        if checkpoint_dir:
            checkpoint_root = Path(checkpoint_dir)
            best_safe_path = checkpoint_root / "best_safe.pt"
            best_candidate_path = checkpoint_root / "best_candidate.pt"
            latest_path = checkpoint_root / "latest.pt"
            if is_new_best_safe:
                # 当前模型就是刚通过验证的新最佳模型，立即形成可恢复的安全快照。
                save_checkpoint(
                    best_safe_path,
                    model,
                    config,
                    episode_returns=episode_returns,
                    optimizer_state_dict=agent.opt.state_dict(),
                    training_summary=current_summary,
                    training_state=training_state,
                    checkpoint_kind="best_safe",
                )
            if should_validate and is_new_best_candidate:
                save_checkpoint(
                    best_candidate_path,
                    model,
                    config,
                    episode_returns=episode_returns,
                    optimizer_state_dict=agent.opt.state_dict(),
                    training_summary=current_summary,
                    training_state=training_state,
                    checkpoint_kind="best_candidate",
                )
            save_checkpoint(
                latest_path,
                model,
                config,
                episode_returns=episode_returns,
                optimizer_state_dict=agent.opt.state_dict(),
                training_summary=current_summary,
                training_state=training_state,
                checkpoint_kind="latest",
            )
            if (
                bool(config.get("persist_monitor_checkpoints", False))
                and episodes_seen in monitor_episodes
            ):
                monitor_path = (
                    checkpoint_root / f"monitor_ep{episodes_seen:04d}.pt"
                )
                save_checkpoint(
                    monitor_path,
                    model,
                    config,
                    episode_returns=episode_returns,
                    optimizer_state_dict=agent.opt.state_dict(),
                    training_summary=current_summary,
                    training_state=training_state,
                    checkpoint_kind="monitor",
                )

        if metrics_callback is not None:
            callback_record = copy.deepcopy(update_stats)
            callback_record.update(
                {
                    "latest_checkpoint": str(latest_path) if latest_path else None,
                    "best_safe_checkpoint": (
                        str(best_safe_path)
                        if best_safe_path is not None and best_safe_path.exists()
                        else None
                    ),
                    "best_candidate_checkpoint": (
                        str(best_candidate_path)
                        if best_candidate_path is not None
                        and best_candidate_path.exists()
                        else None
                    ),
                    "monitor_checkpoint": (
                        str(monitor_path) if monitor_path is not None else None
                    ),
                }
            )
            # 回调异常不吞掉；latest已先原子落盘，外层可返回非零并安全续训。
            metrics_callback(callback_record)

    selection_kind = "best_safe"
    if best_state is None or best_optimizer_state is None:
        if (
            bool(config["simulation_only"])
            and best_candidate_state is not None
            and best_candidate_optimizer_state is not None
        ):
            # 安全机制消融允许没有安全模型，但必须显式标成诊断候选，不能冒充best_safe。
            selection_kind = "best_candidate_unsafe"
            model.load_state_dict(best_candidate_state)
            agent.opt.load_state_dict(best_candidate_optimizer_state)
            logger.warning(
                "仿真消融%s没有通过安全门槛，返回best_candidate用于风险诊断。",
                config["experiment_variant"],
            )
        else:
            raise ConstraintViolationError(
                "训练结束后没有任何验证模型同时达到零约束故障和100%返航；"
                f"最佳诊断候选为 {best_candidate_validation!r}。"
            )
    else:
        model.load_state_dict(best_state)
        agent.opt.load_state_dict(best_optimizer_state)
    training_summary = {
        "schema_version": SCHEMA_VERSION,
        "episodes_seen": episodes_seen,
        "updates": update_index,
        "environment_interactions": environment_interactions,
        "interaction_count_complete": interaction_count_complete,
        "experiment": _experiment_metadata(
            model,
            config,
            environment_interactions,
            interaction_count_complete,
        ),
        "best_episode": best_episode,
        "best_candidate_episode": best_candidate_episode,
        "best_validation": best_validation or {},
        "best_candidate_validation": best_candidate_validation or {},
        "selection_kind": selection_kind,
        "validation_mode": str(config["validation_mode"]),
        "validation_instances_hash": str(config.get("validation_instances_hash", "")),
        "validation_instance_count": int(config["validation_scenarios"]),
        "training_mode": str(config["training_mode"]),
        "training_instances_hash": str(
            config.get("training_instances_hash", "")
        ),
        "training_instance_count": int(
            config.get("training_instance_count", 0)
        ),
        "entropy_progress": float(entropy_progress),
        "history": history,
    }
    model.training_summary = training_summary  # type: ignore[attr-defined]
    model.optimizer_state_dict = agent.opt.state_dict()  # type: ignore[attr-defined]
    model.training_rng_state = copy.deepcopy(rng.bit_generator.state)  # type: ignore[attr-defined]
    model.eval()

    if checkpoint_dir:
        selected_name = (
            "best_safe.pt" if selection_kind == "best_safe" else "best_candidate.pt"
        )
        save_checkpoint(
            Path(checkpoint_dir) / selected_name,
            model,
            config,
            episode_returns=episode_returns,
            optimizer_state_dict=agent.opt.state_dict(),
            training_summary=training_summary,
            training_state=_snapshot_training_state(),
            checkpoint_kind=(
                "best_safe" if selection_kind == "best_safe" else "best_candidate"
            ),
        )
    return model, episode_returns


def plan_with_policy_improved(
    model: PPO_PtrNet,
    start_pos: Sequence[float],
    points: np.ndarray,
    priorities: np.ndarray,
    terrain: np.ndarray,
    cfg: Mapping[str, Any],
    wind_data: Optional[Mapping[str, Any]] = None,
    *,
    return_details: bool = False,
    decode_mode: str = "deterministic",
):
    """使用学习策略生成路线，不进行top-k人工评分重排。"""

    if decode_mode not in {"deterministic", "stochastic"}:
        raise ValueError("decode_mode 只能是 'deterministic' 或 'stochastic'。")
    config = resolve_config(cfg)
    state = build_episode(
        start_pos, points, terrain, config, wind_data, randomize=False
    )
    model_device = next(model.parameters()).device
    rng_distribution_mode = decode_mode == "stochastic"
    previous_mode = model.training
    model.eval()
    try:
        for _step in range(np.asarray(points).shape[0] + 1):
            observation = _build_observation(state, priorities)
            tensors = _observation_tensors(observation, model_device)
            with torch.no_grad():
                _, logits, _ = _model_forward(model, tensors)
                if rng_distribution_mode:
                    action = torch.distributions.Categorical(logits=logits).sample()
                else:
                    # 确定性策略解码：只使用Pointer输出，不引入人工贪心分数。
                    action = torch.argmax(logits, dim=-1)
            state, _, done = step_env_improved(
                state,
                int(action.item()),
                points,
                priorities,
                terrain,
                config,
                wind_data,
            )
            if done:
                break
        if not state["done"]:
            raise ConstraintViolationError("确定性规划未在N+1步内返航。")
    finally:
        model.train(previous_mode)

    path = [np.asarray(point, dtype=np.float32).tolist() for point in state["path_history"]]
    if not return_details:
        return path
    metrics = _episode_metrics(state, np.asarray(priorities, dtype=np.float32))
    training_summary = dict(getattr(model, "training_summary", {}) or {})
    experiment = _experiment_metadata(
        model,
        config,
        int(training_summary.get("environment_interactions", 0)),
        bool(training_summary.get("interaction_count_complete", True)),
    )
    segments: List[Dict[str, Any]] = []
    for raw_segment in state["executed_segments"]:
        segment = dict(raw_segment)
        segment["mean_wind_mps"] = np.asarray(
            segment["mean_wind_mps"], dtype=np.float32
        ).tolist()
        segment["flight_path"] = np.asarray(
            segment["flight_path"], dtype=np.float32
        ).tolist()
        segments.append(segment)
    return {
        "path": path,
        "flight_path": [
            np.asarray(point, dtype=np.float32).tolist() for point in state["flight_path"]
        ],
        "visit_order": list(state["visited"]),
        "segments": segments,
        "energy_wh": metrics["energy_wh"],
        "distance_m": metrics["distance_m"],
        "time_s": metrics["time_s"],
        "min_remaining_soc": metrics["min_remaining_soc"],
        "termination_reason": metrics["termination_reason"],
        "experiment": experiment,
        "metrics": metrics,
    }


def _mat_stats(detail: Mapping[str, Any], episode_returns: Sequence[float], cfg: Mapping[str, Any]):
    metrics = detail["metrics"]
    final_window = list(episode_returns[-min(100, len(episode_returns)) :])
    return {
        "episodes": np.array([len(episode_returns)], dtype=np.float64),
        "final_avg_score": np.array(
            [float(np.mean(final_window)) if final_window else 0.0], dtype=np.float64
        ),
        "max_score": np.array(
            [float(np.max(episode_returns)) if episode_returns else 0.0], dtype=np.float64
        ),
        # 短训练不再伪造“收敛回合”，-1表示未做统计收敛判定。
        "convergence_episode": np.array([-1.0], dtype=np.float64),
        "coverage": np.array([metrics["coverage"]], dtype=np.float64),
        "weighted_coverage": np.array([metrics["weighted_coverage"]], dtype=np.float64),
        "visited_count": np.array([metrics["visited_count"]], dtype=np.float64),
        "total_energy_consumed": np.array([metrics["energy_wh"]], dtype=np.float64),
        "total_distance_m": np.array([metrics["distance_m"]], dtype=np.float64),
        "total_time_s": np.array([metrics["time_s"]], dtype=np.float64),
        "min_remaining_soc": np.array([metrics["min_remaining_soc"]], dtype=np.float64),
        "returned": np.array([1.0 if metrics["returned"] else 0.0], dtype=np.float64),
        "termination_reason": metrics["termination_reason"],
        "battery_reserve_ratio": np.array(
            [cfg["battery_reserve_ratio"]], dtype=np.float64
        ),
        "schema_version": np.array([SCHEMA_VERSION], dtype=np.float64),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python final_python_ppo_pointer.py <input_file> <output_file>")
        return 2
    if scipy is None:
        print("缺少 scipy，无法读取/写入 .mat。请在 Deeplearning 环境安装 scipy。")
        return 2

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    setup_logging(output_file.parent / "logs")
    start_pos: Sequence[float] = [0.0, 0.0, 0.0]
    try:
        mat_data = scipy.io.loadmat(input_file)
        start_pos, points, priorities, terrain, wind_data, config = extract_input(mat_data)
        model, episode_returns = train_policy_improved(
            start_pos, points, priorities, terrain, config, wind_data
        )
        detail = plan_with_policy_improved(
            model,
            start_pos,
            points,
            priorities,
            terrain,
            config,
            wind_data,
            return_details=True,
            decode_mode="deterministic",
        )
        checkpoint_path = output_file.with_suffix(".best_safe.pt")
        save_checkpoint(
            checkpoint_path,
            model,
            config,
            episode_returns=episode_returns,
            optimizer_state_dict=getattr(model, "optimizer_state_dict", None),
            training_summary=getattr(model, "training_summary", {}),
        )
        result = {
            "path": np.asarray(detail["path"], dtype=np.float64),
            "flight_path": np.asarray(detail["flight_path"], dtype=np.float64),
            "visit_order": np.asarray(detail["visit_order"], dtype=np.int64),
            "episode_returns": np.asarray(episode_returns, dtype=np.float64),
            "stats": _mat_stats(detail, episode_returns, config),
            "checkpoint_path": str(checkpoint_path),
        }
        scipy.io.savemat(output_file, result)
        logger.info(
            "PPO+Pointer v2完成：访问%d/%d点，返航=%s，能耗=%.2fWh，距离=%.1fm，时间=%.1fs",
            detail["metrics"]["visited_count"],
            len(points),
            detail["metrics"]["returned"],
            detail["metrics"]["energy_wh"],
            detail["metrics"]["distance_m"],
            detail["metrics"]["time_s"],
        )
        return 0
    except Exception as exc:
        logger.error("执行失败：%s\n%s", exc, traceback.format_exc())
        error_result = {
            "path": np.asarray([start_pos], dtype=np.float64),
            "flight_path": np.asarray([start_pos], dtype=np.float64),
            "status": "error",
            "error_message": str(exc),
            "stats": {
                "schema_version": np.array([SCHEMA_VERSION], dtype=np.float64),
                "returned": np.array([0.0], dtype=np.float64),
            },
        }
        try:
            scipy.io.savemat(output_file, error_result)
        except Exception:
            logger.error("错误结果也无法写入：%s", output_file)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
