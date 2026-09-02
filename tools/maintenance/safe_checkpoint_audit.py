#!/usr/bin/env python3
"""使用受限pickle全局表安全检查正式PyTorch checkpoint。"""

from __future__ import annotations

import _codecs
import argparse
import io
import json
import pickle
import pickletools
import types
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch

from uav_inspection.paths import WORKSPACE_ROOT


# 正式35个checkpoint经pickletools静态扫描只包含下列全局对象。
ALLOWED_GLOBALS: dict[tuple[str, str], Any] = {
    ("_codecs", "encode"): _codecs.encode,
    ("collections", "OrderedDict"): __import__("collections").OrderedDict,
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("torch", "ByteStorage"): torch.ByteStorage,
    ("torch", "FloatStorage"): torch.FloatStorage,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
}


class RestrictedUnpickler(pickle.Unpickler):
    """拒绝checkpoint中未预先登记的任意Python全局对象。"""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return ALLOWED_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"checkpoint包含未许可全局对象：{module}.{name}"
            ) from exc


def _restricted_load(file: BinaryIO, **kwargs: Any) -> Any:
    return RestrictedUnpickler(file, **kwargs).load()


def _restricted_loads(payload: bytes, **kwargs: Any) -> Any:
    return RestrictedUnpickler(io.BytesIO(payload), **kwargs).load()


RESTRICTED_PICKLE = types.ModuleType("uav_restricted_checkpoint_pickle")
RESTRICTED_PICKLE.Unpickler = RestrictedUnpickler
RESTRICTED_PICKLE.load = _restricted_load
RESTRICTED_PICKLE.loads = _restricted_loads


def _declared_globals(path: Path) -> set[tuple[str, str]]:
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        payload = archive.read(member)
    declared: set[tuple[str, str]] = set()
    for operation, argument, _ in pickletools.genops(payload):
        if operation.name == "STACK_GLOBAL":
            raise pickle.UnpicklingError("checkpoint使用不允许的STACK_GLOBAL")
        if operation.name == "GLOBAL":
            module, name = str(argument).split(" ", 1)
            declared.add((module, name))
    return declared


def _finite_tensors(value: Any) -> bool:
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tensors(item) for item in value)
    return True


def audit_checkpoints() -> dict[str, Any]:
    roots = (
        WORKSPACE_ROOT / "paper_runs/multimap_v3_1/formal_training",
        WORKSPACE_ROOT / "paper_runs/multimap_v3_2/formal_training",
    )
    checkpoints = sorted(path for root in roots for path in root.glob("formal_*/best_safe.pt"))
    errors: list[str] = []
    loaded = 0
    for path in checkpoints:
        unknown = _declared_globals(path) - set(ALLOWED_GLOBALS)
        if unknown:
            errors.append(f"{path}:unknown_globals={sorted(unknown)}")
            continue
        try:
            payload = torch.load(
                path,
                map_location="cpu",
                pickle_module=RESTRICTED_PICKLE,
                weights_only=False,
            )
        except Exception as exc:  # pragma: no cover - 失败详情必须进入审计输出
            errors.append(f"{path}:{type(exc).__name__}:{exc}")
            continue
        if not isinstance(payload, dict) or not _finite_tensors(payload):
            errors.append(f"{path}:invalid_or_nonfinite_payload")
            continue
        loaded += 1
    return {
        "schema": "restricted-checkpoint-audit-v1",
        "checkpoint_count": len(checkpoints),
        "loaded_count": loaded,
        "allowed_globals": [f"{module}.{name}" for module, name in sorted(ALLOWED_GLOBALS)],
        "errors": errors,
        "passed": len(checkpoints) == 35 and loaded == 35 and not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="安全加载并审计35个正式checkpoint")
    parser.parse_args()
    report = audit_checkpoints()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
