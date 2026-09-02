import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from uav_inspection.core import paper_protocol as pp


class ProtocolFixture:
    def __init__(self, root: Path):
        self.root = root
        self.training_root = root / "paper_runs" / "training"
        self.manifest_root = root / "paper_runs" / "manifests" / "frozen_v1"
        self.code_file = root / "core.py"
        self.code_file.write_text("VALUE = 1\n", encoding="utf-8")
        self.metadata, self.records = self._write_manifest()
        self._write_training_grid()
        # 额外的smoke/pilot/acceptance检查点必须被正式协议排除。
        for name in (
            "smoke_full_seed42_2ep",
            "pilot_full_seed42_100ep",
            "acceptance_full_seed42_4ep",
        ):
            directory = self.training_root / name
            directory.mkdir(parents=True)
            (directory / "best_safe.pt").write_bytes(b"not-formal")

    @staticmethod
    def _record(split: str, index: int) -> dict:
        denominator = max(pp.SPLIT_COUNTS[split] - 1, 1)
        fraction = index / denominator
        node_count = int(split.split("_")[-1]) if split.startswith("scale_") else 16
        return {
            "id": f"{split}_{index:03d}",
            "split": split,
            "replicate_id": index,
            "instance_seed": 10_000 + index,
            "node_count": node_count,
            "initial_soc": 0.8 + 0.2 * fraction,
            "distance_budget_scale": 0.85 + 0.15 * fraction,
            "time_budget_scale": 0.86 + 0.14 * fraction,
            "wind_scale": 0.8 + 0.4 * fraction,
            "wind_rotation_deg": -15.0 + 30.0 * fraction,
            "wind_vertical_bias_mps": -1.0 + 2.0 * fraction,
            "power_scale": 1.0,
            "priorities": [1] * node_count,
        }

    def _write_manifest(self):
        records = [
            self._record(split, index)
            for split, count in pp.SPLIT_COUNTS.items()
            for index in range(count)
        ]
        records_text = "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for record in records
        )
        records_hash = hashlib.sha256(records_text.encode("utf-8")).hexdigest()
        metadata = {
            "schema_version": 1,
            "created_by": "unit-test",
            # 回归真实Windows工作区的非ASCII路径及既有manifest_hash算法。
            "base_scenario_file": "目录/场景.npz",
            "base_scenario_hash": "a" * 64,
            "manifest_seed": 20260720,
            "split_counts": dict(pp.SPLIT_COUNTS),
            "records_file": "instances.jsonl",
            "records_sha256": records_hash,
            "training_seed_namespace": "model seeds 42-46",
            "selection_integration_status": "external_fixed_v1",
        }
        canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        metadata["manifest_hash"] = hashlib.sha256(
            (canonical + records_hash).encode("utf-8")
        ).hexdigest()
        self.manifest_root.mkdir(parents=True)
        (self.manifest_root / "instances.jsonl").write_bytes(
            records_text.encode("utf-8")
        )
        (self.manifest_root / "manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return metadata, records

    def _write_training_grid(self):
        for variant in pp.LEARNING_VARIANTS:
            for seed in pp.TRAINING_SEEDS:
                run_dir = (
                    self.training_root / f"formal_{variant}_seed{seed}_3000ep"
                )
                run_dir.mkdir(parents=True)
                checkpoint = run_dir / "best_safe.pt"
                checkpoint.write_bytes(f"{variant}:{seed}".encode("utf-8"))
                digest = pp.sha256_file(checkpoint)
                status = {
                    "schema_version": 1,
                    "state": "completed",
                    "best_safe_available": True,
                    "checkpoint_verification_passed": True,
                    "episodes_seen": 3000,
                    "target_episodes": 3000,
                    "best_checkpoint": str(checkpoint),
                }
                training_config = {
                    "experiment_variant": variant,
                    "seed": seed,
                    "experiment_stage": "formal",
                    "max_episodes": 3000,
                    "scenario_hash": self.metadata["base_scenario_hash"],
                    "paper_manifest_hash": self.metadata["manifest_hash"],
                }
                config = {
                    "schema_version": 1,
                    "kind": "learning_training",
                    "variant": variant,
                    "training_seed": seed,
                    "target_episodes": 3000,
                    "scenario_hash": self.metadata["base_scenario_hash"],
                    "manifest_hash": self.metadata["manifest_hash"],
                    "training_config": training_config,
                }
                verification = {
                    "passed": True,
                    "safe_checkpoint_available": True,
                    "simulation_only": variant == "no_return_reserve",
                    "checkpoints": [
                        {
                            "file": "best_safe.pt",
                            "checkpoint_kind": "best_safe",
                            "safe": True,
                            "deterministic_reproducible": True,
                            "sha256": digest,
                        }
                    ],
                }
                for name, payload in (
                    ("status.json", status),
                    ("run_config.json", config),
                    ("checkpoint_verification.json", verification),
                ):
                    (run_dir / name).write_text(
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        encoding="utf-8",
                    )

    def build_protocol(self) -> dict:
        return pp.build_frozen_protocol(
            self.training_root,
            self.manifest_root,
            repo_root=self.root,
            code_files=[self.code_file],
            environment={"fixture": "stable", "python": "unit-test"},
        )


def make_row(protocol: dict, metadata: dict, record: dict, algorithm: str, seed: int):
    learning = algorithm in pp.LEARNING_VARIANTS
    checkpoint_hash = ""
    if learning:
        checkpoint_hash = next(
            item["sha256"]
            for item in protocol["checkpoints"]
            if item["variant"] == algorithm and item["training_seed"] == seed
        )
    return {
        "schema_version": pp.RESULT_SCHEMA_VERSION,
        "scenario_id": record["id"],
        "split": record["split"],
        "algorithm": algorithm,
        "returned": True,
        "energy_violation": False,
        "distance_violation": False,
        "time_violation": False,
        "dynamics_violation": False,
        "termination_reason": "returned_full",
        "weighted_coverage": 1.0,
        "safe_weighted_coverage": 1.0,
        "coverage": 1.0,
        "safe_coverage": 1.0,
        "visited_count": record["node_count"],
        "low_priority_coverage": 1.0,
        "medium_priority_coverage": None,
        "high_priority_coverage": None,
        "energy_wh": 10.0,
        "distance_m": 100.0,
        "time_s": 20.0,
        "energy_budget_wh": 100.0,
        "distance_budget_m": 1000.0,
        "time_budget_s": 200.0,
        "energy_utilization": 0.1,
        "distance_utilization": 0.1,
        "time_utilization": 0.1,
        "min_remaining_soc": 0.9,
        "planning_time_s": 0.01,
        "variant": algorithm if learning else "",
        "training_seed": seed if learning else None,
        "planner_seed": None if learning else seed,
        "replicate_id": record["replicate_id"],
        "checkpoint_hash": checkpoint_hash,
        "scenario_hash": metadata["base_scenario_hash"],
        "manifest_hash": metadata["manifest_hash"],
        "evaluations": None if learning else 1,
        "optimality_gap": None,
        "solver_dual_bound": None,
        "solver_status": "",
        "optimality_certified": None,
        "node_count": record["node_count"],
        "power_scale": 1.0,
        "simulation_only": algorithm == "no_return_reserve",
        "protocol_hash": protocol["protocol_hash"],
    }


class PaperProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = ProtocolFixture(self.root)
        self.protocol = self.fixture.build_protocol()

    def tearDown(self):
        self.temporary.cleanup()

    def test_build_is_stable_complete_and_excludes_nonformal_runs(self):
        second = self.fixture.build_protocol()
        self.assertEqual(self.protocol["protocol_hash"], second["protocol_hash"])
        self.assertEqual(len(self.protocol["checkpoints"]), 35)
        self.assertEqual(
            {
                (item["variant"], item["training_seed"])
                for item in self.protocol["checkpoints"]
            },
            {
                (variant, seed)
                for variant in pp.LEARNING_VARIANTS
                for seed in pp.TRAINING_SEEDS
            },
        )
        self.assertTrue(
            all("formal_" in item["run_dir"] for item in self.protocol["checkpoints"])
        )
        self.assertEqual(
            self.protocol["statistics_families"]["main"]["members"],
            ["full", "ppo_mlp", "a2c_pointer", *pp.MAIN_BASELINES],
        )
        self.assertEqual(
            self.protocol["statistics_families"]["ablation"]["members"],
            ["full", *pp.ABLATION_VARIANTS],
        )
        self.assertEqual(
            self.protocol["statistics_families"]["supplementary"]["members"],
            ["full", *pp.SUPPLEMENTARY_BASELINES],
        )
        self.assertEqual(
            self.protocol["metrics"]["primary_metric"], "safe_weighted_coverage"
        )
        self.assertEqual(self.protocol["primary_id_test_counts"]["total"], 8000)
        self.assertEqual(self.protocol["secondary_experiments"]["stress_test"]["scope"], "full")
        self.assertEqual(
            self.protocol["secondary_experiments"]["scale_generalization"]["scope"],
            "core",
        )

    def test_representative_scenario_uses_id_tie_break(self):
        common = {field: 1.0 for field in pp.REPRESENTATIVE_FIELDS}
        records = [
            {"id": "id_test_002", "split": "id_test", **common},
            {"id": "id_test_001", "split": "id_test", **common},
            {"id": "validation_000", "split": "validation", **common},
        ]
        selected = pp.select_representative_scenario(records)
        self.assertEqual(selected["scenario_id"], "id_test_001")
        self.assertEqual(selected["fields"], list(pp.REPRESENTATIVE_FIELDS))

    def test_write_load_is_immutable_and_asset_verification_detects_drift(self):
        destination = self.root / "paper_runs" / "protocols" / pp.PROTOCOL_NAME
        path = pp.write_frozen_protocol(self.protocol, destination)
        self.assertEqual(pp.load_frozen_protocol(destination), self.protocol)
        self.assertEqual(pp.write_frozen_protocol(self.protocol, destination), path)

        changed = copy.deepcopy(self.protocol)
        changed["environment"]["fixture"] = "different"
        changed["environment_sha256"] = hashlib.sha256(
            json.dumps(
                changed["environment"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        changed["protocol_hash"] = pp.compute_protocol_hash(changed)
        with self.assertRaises(FileExistsError):
            pp.write_frozen_protocol(changed, destination)

        report = pp.verify_protocol_assets(
            path, repo_root=self.root, verify_checkpoints=True, verify_code=True
        )
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["verified_checkpoints"]), 35)

        checkpoint = self.root / self.protocol["checkpoints"][0]["path"]
        original_checkpoint = checkpoint.read_bytes()
        checkpoint.write_bytes(original_checkpoint + b"drift")
        with self.assertRaises(pp.ProtocolError):
            pp.verify_protocol_assets(path, repo_root=self.root)
        checkpoint.write_bytes(original_checkpoint)

        self.fixture.code_file.write_text("VALUE = 2\n", encoding="utf-8")
        with self.assertRaises(pp.ProtocolError):
            pp.verify_protocol_assets(path, repo_root=self.root)

    def test_checkpoint_verification_hash_mismatch_is_rejected(self):
        run_dir = self.fixture.training_root / "formal_full_seed42_3000ep"
        verification_path = run_dir / "checkpoint_verification.json"
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        verification["checkpoints"][0]["sha256"] = "0" * 64
        verification_path.write_text(json.dumps(verification), encoding="utf-8")
        with self.assertRaises(pp.ProtocolError):
            self.fixture.build_protocol()

    def _records(self, split="validation"):
        return [record for record in self.fixture.records if record["split"] == split]

    def test_jsonl_loader_does_not_duplicate_rows_and_directory_audit_checks_routes(self):
        records = self._records()
        rows = [
            make_row(
                self.protocol,
                self.fixture.metadata,
                record,
                "nearest_feasible",
                pp.DETERMINISTIC_PLANNER_SEED,
            )
            for record in records
        ]
        run_dir = self.root / "results" / "nearest_validation"
        routes = run_dir / "routes"
        routes.mkdir(parents=True)
        results_text = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
        (run_dir / "results.jsonl").write_text(results_text, encoding="utf-8")
        status = {
            "schema_version": 1,
            "state": "completed",
            "completed": len(rows),
            "total": len(rows),
        }
        immutable = {
            "protocol_hash": self.protocol["protocol_hash"],
            "manifest_hash": self.fixture.metadata["manifest_hash"],
            "scenario_hash": self.fixture.metadata["base_scenario_hash"],
            "selected_records_sha256": pp.selected_records_hash(records),
            "record_count": len(records),
            "split": "validation",
            "algorithms": ["nearest_feasible"],
            "planner_seeds": [42],
            "power_scales": [1.0],
        }
        (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
        (run_dir / "run_config.json").write_text(
            json.dumps(
                {"schema_version": 1, "kind": "traditional_baselines", "immutable": immutable}
            ),
            encoding="utf-8",
        )
        for row in rows:
            route_name = f"{row['scenario_id']}__nearest_feasible__seed42__power1.json"
            (routes / route_name).write_text("{}", encoding="utf-8")

        report = pp.audit_result_runs(
            self.protocol,
            self.fixture.manifest_root,
            run_dir,
            "nearest_feasible",
            "validation",
            (1.0,),
        )
        self.assertEqual(report["row_count"], len(records))
        self.assertEqual(report["row_count"], report["expected_row_count"])

        (routes / "validation_000__nearest_feasible__seed42__power1.json").unlink()
        with self.assertRaises(pp.ProtocolError):
            pp.audit_result_runs(
                self.protocol,
                self.fixture.manifest_root,
                run_dir,
                "nearest_feasible",
                "validation",
                (1.0,),
            )

    def test_main_family_infers_five_ten_and_one_repeats(self):
        algorithms = self.protocol["statistics_families"]["main"]["members"]
        rows = []
        for record in self._records():
            for algorithm in algorithms:
                if algorithm in pp.LEARNING_VARIANTS:
                    seeds = pp.TRAINING_SEEDS
                elif algorithm in pp.STOCHASTIC_BASELINES:
                    seeds = pp.PLANNER_SEEDS
                else:
                    seeds = (pp.DETERMINISTIC_PLANNER_SEED,)
                rows.extend(
                    make_row(
                        self.protocol,
                        self.fixture.metadata,
                        record,
                        algorithm,
                        seed,
                    )
                    for seed in seeds
                )
        report = pp.audit_result_runs(
            self.protocol,
            (self.fixture.metadata, self.fixture.records),
            rows,
            "main",
            "validation",
            [1.0],
        )
        expected_per_scenario = 3 * 5 + 3 * 10 + 3 * 1
        self.assertEqual(report["row_count"], 64 * expected_per_scenario)

        with self.assertRaises(pp.ProtocolError):
            pp.audit_result_runs(
                self.protocol,
                (self.fixture.metadata, self.fixture.records),
                rows[:-1],
                "main",
                "validation",
                [1.0],
            )

    def test_audit_rejects_duplicate_nonfinite_and_result_schema_v1(self):
        records = self._records()
        rows = [
            make_row(
                self.protocol,
                self.fixture.metadata,
                record,
                "nearest_feasible",
                42,
            )
            for record in records
        ]
        with self.assertRaises(pp.ProtocolError):
            pp.audit_result_runs(
                self.protocol,
                (self.fixture.metadata, self.fixture.records),
                [*rows, copy.deepcopy(rows[0])],
                "nearest_feasible",
                "validation",
                [1.0],
            )
        broken = copy.deepcopy(rows)
        broken[0]["energy_wh"] = math.nan
        with self.assertRaises(pp.ProtocolError):
            pp.audit_result_runs(
                self.protocol,
                (self.fixture.metadata, self.fixture.records),
                broken,
                "nearest_feasible",
                "validation",
                [1.0],
            )
        old_schema = copy.deepcopy(rows)
        old_schema[0]["schema_version"] = 1
        with self.assertRaises(pp.ProtocolError):
            pp.audit_result_runs(
                self.protocol,
                (self.fixture.metadata, self.fixture.records),
                old_schema,
                "nearest_feasible",
                "validation",
                [1.0],
            )
        missing_budget = copy.deepcopy(rows)
        missing_budget[0]["energy_budget_wh"] = None
        with self.assertRaises(pp.ProtocolError):
            pp.audit_result_runs(
                self.protocol,
                (self.fixture.metadata, self.fixture.records),
                missing_budget,
                "nearest_feasible",
                "validation",
                [1.0],
            )


if __name__ == "__main__":
    unittest.main()
