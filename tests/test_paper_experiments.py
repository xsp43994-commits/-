import json
import io
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from uav_inspection.core import final_python_ppo_pointer as core_ppo
from uav_inspection.experiments import paper_experiments as pe


class FakeScenario:
    def __init__(self):
        self.scenario_hash = "a" * 64
        self.coordinate_scale_m_per_unit = 12.5
        self.start_pos = np.array([50.0, 50.0, 100.0], dtype=np.float32)
        horizontal = np.column_stack(
            [np.linspace(0, 100, 201), np.full(201, 50.0), np.full(201, 100.0)]
        )
        vertical = np.column_stack(
            [np.full(201, 50.0), np.linspace(0, 100, 201), np.full(201, 100.0)]
        )
        self.road_1 = horizontal.astype(np.float32)
        self.road_2 = vertical.astype(np.float32)
        points = []
        arms = []
        along = []
        for arm_id, direction in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            for distance in (12.0, 24.0, 36.0, 48.0):
                points.append([50 + direction[0] * distance, 50 + direction[1] * distance, 100])
                arms.append(arm_id)
                along.append(distance * 12.5)
        self.inspection_points = np.asarray(points, dtype=np.float32)
        self.priorities = np.asarray([3] * 5 + [2] * 6 + [1] * 5, dtype=np.int32)
        self.service_times_s = np.full(16, 20.0, dtype=np.float32)
        self.point_arm_ids = np.asarray(arms, dtype=np.int16)
        self.point_along_arm_distances_m = np.asarray(along, dtype=np.float32)
        self.risk_scores = np.linspace(0.1, 0.9, 16, dtype=np.float32)
        self.terrain = np.full((101, 101), 100.0, dtype=np.float32)
        self.wind_data = {
            "positions": np.array([[50, 50, 118]], dtype=np.float32),
            "vectors": np.array([[2, 0, 0]], dtype=np.float32),
            "uniform_vector": np.array([2, 0, 0], dtype=np.float32),
        }

    def as_training_inputs(self):
        return {
            "cfg": {
                "coordinate_scale_m_per_unit": 12.5,
                "service_times_s": self.service_times_s.tolist(),
                "return_to_start": True,
            }
        }


class FakePPO:
    DEFAULT_CONFIG = {
        "max_route_distance": 8000.0,
        "max_mission_time_s": 2400.0,
        "hover_power_w": 172.8,
        "cruise_power_w": 138.24,
        "climb_power_w": 216.0,
        "descent_power_w": 138.24,
    }

    @classmethod
    def resolve_config(cls, cfg):
        return dict(cls.DEFAULT_CONFIG, **dict(cfg))

    normalize_validation_instances = staticmethod(core_ppo.normalize_validation_instances)
    transform_wind_for_domain_instance = staticmethod(
        core_ppo.transform_wind_for_domain_instance
    )
    apply_frozen_domain_instance = staticmethod(core_ppo.apply_frozen_domain_instance)

    def __init__(self):
        self.checkpoint_cfg = None
        self.checkpoint_scenario_hash = "a" * 64
        self.checkpoint_manifest_hash = ""
        self.completed_episodes = 0
        self.plan_calls = 0
        self.fail_on_plan_call = None
        self.unsafe_latest = False

    def setup_logging(self, _path):
        return None

    def train_policy_improved(self, *_args, **kwargs):
        cfg = _args[4]
        self.checkpoint_cfg = dict(cfg)
        self.completed_episodes = int(cfg["max_episodes"])
        self.last_validation_instances = list(kwargs["validation_instances"])
        run_dir = Path(cfg["checkpoint_dir"])
        (run_dir / "latest.pt").write_bytes(b"latest")
        (run_dir / "best_safe.pt").write_bytes(b"best")
        kwargs["metrics_callback"]({"update": 1, "episodes_seen": cfg["max_episodes"], "kl": 0.01})
        return object(), [1.0] * int(cfg["max_episodes"])

    def load_checkpoint(self, _path, map_location=None):
        del map_location
        name = Path(_path).name
        cfg = dict(
            self.checkpoint_cfg
            or dict(
                self.DEFAULT_CONFIG,
                experiment_variant="full",
                seed=42,
                scenario_hash=self.checkpoint_scenario_hash,
                paper_manifest_hash=self.checkpoint_manifest_hash,
            )
        )
        episodes = int(self.completed_episodes)
        model = type("LoadedModel", (), {"checkpoint_name": name})()
        return model, {
            "cfg": cfg,
            "seed": int(cfg.get("seed", 42)),
            "checkpoint_kind": (
                "best_safe" if name == "best_safe.pt" else
                "best_candidate" if name == "best_candidate.pt" else "latest"
            ),
            "optimizer_state_dict": {"state": {}},
            "returns": [1.0] * episodes,
            "training_state": {"episodes_seen": episodes},
            "training_summary": {"episodes_seen": episodes},
            "rng_state": {
                "python": (),
                "numpy_global": (),
                "torch": (),
                "training_generator": {"state": 1},
            },
        }

    def plan_with_policy_improved(self, _model, _start, points, _priorities, _terrain, _cfg, _wind, **_kwargs):
        self.plan_calls += 1
        if self.fail_on_plan_call == self.plan_calls:
            raise RuntimeError("simulated interruption")
        unsafe = self.unsafe_latest and getattr(_model, "checkpoint_name", "") == "latest.pt"
        count = len(points)
        return {
            "path": [[0, 0, 0], [0, 0, 0]],
            "flight_path": [[0, 0, 0], [0, 0, 0]],
            "metrics": {
                "returned": not unsafe,
                "energy_violation": unsafe,
                "distance_violation": False,
                "time_violation": False,
                "dynamics_violation": False,
                "termination_reason": "stranded" if unsafe else "returned_full",
                "coverage": 1.0,
                "weighted_coverage": 1.0,
                "visited_count": count,
                "energy_wh": 10.0,
                "distance_m": 1000.0,
                "time_s": 200.0,
                "min_remaining_soc": 0.8,
            },
        }


class FakePlannerResult:
    def as_dict(self):
        return {
            "scenario_hash": "derived-problem-hash",
            "runtime_s": 0.01,
            "evaluations": 7,
            "metrics": {
                "returned": True,
                "energy_violation": False,
                "distance_violation": False,
                "time_violation": False,
                "dynamics_violation": False,
                "termination_reason": "returned_partial",
                "weighted_coverage": 0.75,
                "coverage": 0.5,
                "energy_wh": 10.0,
                "distance_m": 100.0,
                "time_s": 30.0,
                "min_remaining_soc": 0.8,
            },
            "metadata": {},
        }


class FakeClassicalPackage:
    def __init__(self):
        self.calls = 0
        self.fail_on_call = None

    def run_planner(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated baseline interruption")
        return FakePlannerResult()


class FakeClassicalCommon:
    @staticmethod
    def build_context(_scenario_file, cfg=None):
        return {"cfg": cfg}


class PaperExperimentsTests(unittest.TestCase):
    def test_long_row_records_priority_and_resource_metrics(self):
        record = {
            "id": "id_test_000",
            "split": "id_test",
            "replicate_id": 0,
            "node_count": 4,
            "priorities": [1, 2, 3, 3],
        }
        row = pe._long_row(
            record=record,
            algorithm="full",
            metrics={
                "returned": True,
                "weighted_coverage": 5.0 / 9.0,
                "coverage": 0.5,
                "visited_count": 2,
                "visited_order": [0, 2],
                "energy_wh": 10.0,
                "distance_m": 100.0,
                "time_s": 30.0,
                "energy_budget_wh": 20.0,
                "distance_budget_m": 200.0,
                "time_budget_s": 60.0,
                "min_remaining_soc": 0.7,
            },
            planning_time_s=0.01,
            training_seed=42,
            variant="full",
            scenario_hash="a" * 64,
            manifest_hash="b" * 64,
            power_scale=1.0,
        )
        self.assertEqual(row["schema_version"], 2)
        self.assertEqual(row["visited_count"], 2)
        self.assertEqual(row["low_priority_coverage"], 1.0)
        self.assertEqual(row["medium_priority_coverage"], 0.0)
        self.assertEqual(row["high_priority_coverage"], 0.5)
        self.assertEqual(row["energy_utilization"], 0.5)
        self.assertEqual(row["safe_coverage"], 0.5)

    def test_resume_dry_run_reader_never_repairs_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete_tail = root / "complete.jsonl"
            complete_tail.write_bytes(b'{"scenario_id":"x"}')
            before = complete_tail.read_bytes()
            reader = pe.DurableResultJsonlWriter(
                complete_tail, resume=True, repair_trailing=False
            )
            self.assertEqual(len(reader.records()), 1)
            self.assertEqual(complete_tail.read_bytes(), before)

            broken_tail = root / "broken.jsonl"
            broken_tail.write_bytes(b'{"scenario_id":"x"')
            before = broken_tail.read_bytes()
            with self.assertRaisesRegex(ValueError, "dry-run严格只读"):
                pe.DurableResultJsonlWriter(
                    broken_tail, resume=True, repair_trailing=False
                )
            self.assertEqual(broken_tail.read_bytes(), before)
            self.assertFalse(list(root.glob("broken.jsonl.trailing_partial.*")))

    def test_aggregate_failure_does_not_occupy_formal_run_name(self):
        from uav_inspection.core import paper_evaluation

        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "paper_runs"
            args = pe.build_parser().parse_args(
                ["aggregate", "--inputs", "missing.csv", "--run-name", "atomic"]
            )
            with patch.object(pe, "PAPER_RUNS_ROOT", runs), patch.object(
                paper_evaluation,
                "run_analysis",
                side_effect=ValueError("simulated audit failure"),
            ):
                with self.assertRaisesRegex(ValueError, "audit failure"):
                    pe._aggregate(args)
            output = runs / "analysis" / "atomic"
            self.assertFalse(output.exists())
            self.assertFalse(list((runs / "analysis").glob(".atomic.*.tmp")))

    def test_aggregate_maps_auditable_result_directories_to_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "run"
            result_dir.mkdir()
            result_file = result_dir / "results.jsonl"
            result_file.write_text("{}\n", encoding="utf-8")
            standalone = root / "standalone.csv"
            standalone.write_text("scenario_id\n", encoding="utf-8")
            self.assertEqual(
                pe._analysis_inputs([result_dir, standalone]),
                [result_file, standalone],
            )
            empty = self._empty_dir(root)
            with self.assertRaisesRegex(FileNotFoundError, "results.jsonl"):
                pe._analysis_inputs([empty])

    @staticmethod
    def _empty_dir(root: Path) -> Path:
        path = root / "empty"
        path.mkdir()
        return path

    def test_manifest_is_deterministic_and_has_frozen_split_counts(self):
        scenario = FakeScenario()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "frozen"
            with patch.object(pe, "_load_scenario", return_value=scenario):
                first = pe.prepare_manifest(Path("scenario.npz"), root, manifest_seed=123)
                second = pe.prepare_manifest(Path("scenario.npz"), root, manifest_seed=123)
            self.assertEqual(first["manifest_hash"], second["manifest_hash"])
            metadata, records, _ = pe.load_manifest(root)
            self.assertEqual(metadata["split_counts"], pe.SPLIT_COUNTS)
            self.assertEqual(len(records), 364)
            self.assertEqual(len({record["id"] for record in records}), 364)
            self.assertEqual(metadata["selection_integration_status"], "external_fixed_v1")
            for node_count in (8, 12, 20, 24):
                selected = [r for r in records if r["split"] == f"scale_{node_count}"]
                self.assertEqual(len(selected), 25)
                self.assertTrue(all(len(r["inspection_points_xyz"]) == node_count for r in selected))

    def test_changed_manifest_never_overwrites_existing_directory(self):
        scenario = FakeScenario()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "frozen"
            with patch.object(pe, "_load_scenario", return_value=scenario):
                pe.prepare_manifest(Path("scenario.npz"), root, manifest_seed=123)
                with self.assertRaises(FileExistsError):
                    pe.prepare_manifest(Path("scenario.npz"), root, manifest_seed=124)

    def test_all_subcommands_accept_dry_run(self):
        parser = pe.build_parser()
        commands = {
            "doctor": ["doctor", "--dry-run"],
            "prepare": ["prepare", "--dry-run"],
            "freeze-protocol": ["freeze-protocol", "--dry-run"],
            "smoke": ["smoke", "--dry-run"],
            "train": ["train", "--episodes", "10", "--dry-run"],
            "resume": ["resume", "--run-dir", "x", "--episodes", "20", "--dry-run"],
            "evaluate": ["evaluate", "--checkpoint", "x.pt", "--manifest", "m", "--dry-run"],
            "evaluate-batch": [
                "evaluate-batch", "--protocol", "p", "--manifest", "m", "--dry-run"
            ],
            "baselines": ["baselines", "--manifest", "m", "--dry-run"],
            "aggregate": ["aggregate", "--inputs", "x.csv", "--dry-run"],
            "audit": [
                "audit", "--protocol", "p", "--manifest", "m", "--inputs", "x",
                "--family", "main", "--split", "id_test", "--dry-run",
            ],
            "status": ["status", "--run-dir", "x", "--dry-run"],
        }
        for command, argv in commands.items():
            with self.subTest(command=command):
                parsed = parser.parse_args(argv)
                self.assertEqual(parsed.command, command)
                self.assertTrue(parsed.dry_run)

    def test_registered_baseline_profiles_use_per_algorithm_frozen_budgets(self):
        import python_classical_algs as package

        args = pe.build_parser().parse_args(
            ["baselines", "--manifest", "m", "--profile", "main"]
        )
        algorithms, seeds, budgets, deterministic = pe._baseline_plan(args, package)
        self.assertEqual(
            algorithms,
            [
                "nearest_feasible",
                "priority_resource_greedy",
                "aco",
                "ga",
                "sa",
                "milp_orienteering",
            ],
        )
        self.assertEqual(seeds, list(range(42, 52)))
        self.assertEqual(
            budgets["aco"], {"max_evaluations": 50_000, "time_limit_s": None}
        )
        self.assertEqual(
            budgets["milp_orienteering"],
            {"max_evaluations": None, "time_limit_s": 60.0},
        )
        self.assertEqual(
            budgets["nearest_feasible"],
            {"max_evaluations": None, "time_limit_s": None},
        )
        self.assertIn("milp_orienteering", deterministic)

    def test_baseline_power_sensitivity_expands_dry_run_without_overwrite(self):
        scenario = FakeScenario()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = base / "manifest"
            with patch.object(pe, "_load_scenario", return_value=scenario):
                pe.prepare_manifest(Path("scenario.npz"), manifest, manifest_seed=123)
            args = pe.build_parser().parse_args(
                [
                    "baselines",
                    "--manifest",
                    str(manifest),
                    "--split",
                    "id_test",
                    "--algorithms",
                    "nearest_feasible,aco",
                    "--planner-seeds",
                    "42,43",
                    "--power-scales",
                    "0.8,1.0",
                    "--run-name",
                    "power_test",
                    "--dry-run",
                ]
            )
            output = io.StringIO()
            with patch.object(pe, "PAPER_RUNS_ROOT", base / "runs"), patch.object(
                pe, "_load_scenario", return_value=scenario
            ), patch.object(pe, "_ppo_module", return_value=FakePPO()), patch(
                "sys.stdout", output
            ):
                run_dir = pe._baselines(args)
            payload = json.loads(output.getvalue())
            # 100场景 × 2功率倍率 × (1个确定性重复 + 2个ACO重复)。
            self.assertEqual(payload["planned_runs"], 600)
            self.assertFalse(run_dir.exists())

    def test_power_scale_parser_rejects_invalid_or_duplicate_values(self):
        self.assertEqual(pe._parse_power_scales("0.8, 1.0"), [0.8, 1.0])
        for invalid in ("", "0", "nan", "1.0,1.0"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pe._parse_power_scales(invalid)

    def test_smoke_training_is_limited_and_writes_atomic_state(self):
        scenario = FakeScenario()
        fake_ppo = FakePPO()
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "paper_runs"
            manifest = Path(directory) / "manifest"
            with patch.object(pe, "_load_scenario", return_value=scenario):
                pe.prepare_manifest(Path("scenario.npz"), manifest, manifest_seed=123)
            args = pe.build_parser().parse_args(
                [
                    "smoke", "--episodes", "4", "--device", "cpu",
                    "--run-name", "smoke_test", "--manifest", str(manifest),
                ]
            )
            with patch.object(pe, "PAPER_RUNS_ROOT", runs), patch.object(pe, "_load_scenario", return_value=scenario), patch.object(pe, "_ppo_module", return_value=fake_ppo):
                run_dir = pe._run_training(args, resume=False)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["episodes_seen"], 4)
            self.assertTrue((run_dir / "latest.pt").exists())
            self.assertTrue((run_dir / "best_safe.pt").exists())
            self.assertEqual(len(fake_ppo.last_validation_instances), 64)
            self.assertTrue((run_dir / "validation_instances.jsonl").exists())
            self.assertTrue((run_dir / "checkpoint_verification.json").exists())
            self.assertTrue((run_dir / "scenario_snapshot.npz").exists())
            self.assertTrue((run_dir / "scenario_snapshot.json").exists())
            self.assertTrue((run_dir / "learning_curve.svg").exists())
            resume_args = pe.build_parser().parse_args(
                [
                    "resume", "--run-dir", str(run_dir), "--episodes", "6",
                    "--device", "cpu",
                ]
            )
            with patch.object(pe, "_load_scenario", return_value=scenario), patch.object(
                pe, "_ppo_module", return_value=fake_ppo
            ):
                resumed_dir = pe._run_training(resume_args, resume=True)
            self.assertEqual(resumed_dir, run_dir)
            resumed_status = json.loads(
                (run_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resumed_status["episodes_seen"], 6)
            with self.assertRaises(FileExistsError):
                with patch.object(pe, "PAPER_RUNS_ROOT", runs), patch.object(pe, "_load_scenario", return_value=scenario), patch.object(pe, "_ppo_module", return_value=fake_ppo):
                    pe._run_training(args, resume=False)

    def test_evaluate_emits_required_long_table_fields(self):
        scenario = FakeScenario()
        fake_ppo = FakePPO()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = base / "manifest"
            with patch.object(pe, "_load_scenario", return_value=scenario):
                metadata = pe.prepare_manifest(Path("scenario.npz"), manifest, manifest_seed=123)
            checkpoint = base / "best_safe.pt"
            checkpoint.write_bytes(b"checkpoint")
            fake_ppo.checkpoint_manifest_hash = metadata["manifest_hash"]
            args = pe.build_parser().parse_args(
                ["evaluate", "--checkpoint", str(checkpoint), "--manifest", str(manifest), "--split", "scale", "--run-name", "eval_test", "--device", "cpu"]
            )
            with patch.object(pe, "PAPER_RUNS_ROOT", base / "runs"), patch.object(pe, "_load_scenario", return_value=scenario), patch.object(pe, "_ppo_module", return_value=fake_ppo):
                run_dir = pe._evaluate(args)
            rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 100)
            self.assertTrue(set(pe.LONG_TABLE_FIELDS).issubset(rows[0]))
            self.assertEqual(rows[0]["safe_weighted_coverage"], 1.0)

    def test_evaluate_rejects_checkpoint_provenance_mismatch(self):
        payload = {
            "cfg": {
                "scenario_hash": "wrong-scenario",
                "paper_manifest_hash": "manifest-a",
            }
        }
        with self.assertRaisesRegex(ValueError, "场景哈希"):
            pe._validate_checkpoint_provenance(
                payload, scenario_hash="scenario-a", manifest_hash="manifest-a"
            )
        payload["cfg"]["scenario_hash"] = "scenario-a"
        payload["cfg"]["paper_manifest_hash"] = "wrong-manifest"
        with self.assertRaisesRegex(ValueError, "清单哈希"):
            pe._validate_checkpoint_provenance(
                payload, scenario_hash="scenario-a", manifest_hash="manifest-a"
            )

    def test_checkpoint_verification_allows_unsafe_latest_when_best_safe_is_valid(self):
        scenario = FakeScenario()
        fake_ppo = FakePPO()
        fake_ppo.completed_episodes = 2
        fake_ppo.unsafe_latest = True
        fake_ppo.checkpoint_cfg = dict(
            fake_ppo.DEFAULT_CONFIG,
            simulation_only=False,
            validation_instances_hash="validation-hash",
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "latest.pt").write_bytes(b"latest")
            (run_dir / "best_safe.pt").write_bytes(b"best")
            report = pe._verify_training_checkpoints(
                fake_ppo, scenario, run_dir, "cpu", "validation-hash"
            )
        by_name = {item["file"]: item for item in report["checkpoints"]}
        self.assertTrue(report["passed"])
        self.assertTrue(report["safe_checkpoint_available"])
        self.assertFalse(by_name["latest.pt"]["safe"])
        self.assertTrue(by_name["latest.pt"]["resumable"])
        self.assertTrue(by_name["best_safe.pt"]["safe"])

    def test_resume_updates_checkpoint_dir_after_run_directory_moves(self):
        scenario = FakeScenario()
        fake_ppo = FakePPO()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = base / "manifest"
            with patch.object(pe, "_load_scenario", return_value=scenario):
                pe.prepare_manifest(Path("scenario.npz"), manifest, manifest_seed=123)
            train_args = pe.build_parser().parse_args(
                [
                    "smoke", "--episodes", "2", "--device", "cpu",
                    "--run-name", "movable", "--manifest", str(manifest),
                ]
            )
            with patch.object(pe, "PAPER_RUNS_ROOT", base / "runs"), patch.object(
                pe, "_load_scenario", return_value=scenario
            ), patch.object(pe, "_ppo_module", return_value=fake_ppo):
                original = pe._run_training(train_args, resume=False)
            moved = base / "moved_run"
            shutil.move(str(original), moved)
            resume_args = pe.build_parser().parse_args(
                ["resume", "--run-dir", str(moved), "--episodes", "4", "--device", "cpu"]
            )
            with patch.object(pe, "_load_scenario", return_value=scenario), patch.object(
                pe, "_ppo_module", return_value=fake_ppo
            ):
                pe._run_training(resume_args, resume=True)
            stored = json.loads((moved / "run_config.json").read_text(encoding="utf-8"))
            status = json.loads((moved / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["training_config"]["checkpoint_dir"], str(moved.resolve()))
            self.assertEqual(fake_ppo.checkpoint_cfg["checkpoint_dir"], str(moved.resolve()))
            self.assertTrue((moved / "latest.pt").exists())
            self.assertEqual(status["episodes_seen"], 4)

    def test_evaluation_resume_recovers_partial_tail_and_skips_completed_keys(self):
        scenario = FakeScenario()
        fake_ppo = FakePPO()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = base / "manifest"
            with patch.object(pe, "_load_scenario", return_value=scenario):
                metadata = pe.prepare_manifest(Path("scenario.npz"), manifest, manifest_seed=123)
            checkpoint = base / "best_safe.pt"
            checkpoint.write_bytes(b"checkpoint")
            fake_ppo.checkpoint_manifest_hash = metadata["manifest_hash"]
            argv = [
                "evaluate", "--checkpoint", str(checkpoint), "--manifest", str(manifest),
                "--split", "id_test", "--run-name", "resume_eval", "--device", "cpu",
            ]
            args = pe.build_parser().parse_args(argv)
            fake_ppo.fail_on_plan_call = 4  # 1次预热、2条完成记录后中断。
            with patch.object(pe, "PAPER_RUNS_ROOT", base / "runs"), patch.object(
                pe, "_load_scenario", return_value=scenario
            ), patch.object(pe, "_ppo_module", return_value=fake_ppo):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    pe._evaluate(args)
            run_dir = base / "runs" / "evaluation" / "resume_eval"
            result_path = run_dir / "results.jsonl"
            self.assertEqual(len(result_path.read_text(encoding="utf-8").splitlines()), 2)
            with result_path.open("ab") as stream:
                stream.write(b'{"scenario_id":"truncated"')

            fake_ppo.plan_calls = 0
            fake_ppo.fail_on_plan_call = None
            resume_args = pe.build_parser().parse_args(argv + ["--resume-existing"])
            with patch.object(pe, "PAPER_RUNS_ROOT", base / "runs"), patch.object(
                pe, "_load_scenario", return_value=scenario
            ), patch.object(pe, "_ppo_module", return_value=fake_ppo):
                pe._evaluate(resume_args)
            rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
            keys = {(row["scenario_id"], row["power_scale"]) for row in rows}
            self.assertEqual(len(rows), 100)
            self.assertEqual(len(keys), 100)
            self.assertEqual(fake_ppo.plan_calls, 99)  # 1次预热 + 98个剩余任务。
            self.assertTrue(list(run_dir.glob("results.jsonl.trailing_partial.*")))
            incompatible_args = pe.build_parser().parse_args(
                argv + ["--power-scales", "0.9", "--resume-existing"]
            )
            with patch.object(pe, "PAPER_RUNS_ROOT", base / "runs"), patch.object(
                pe, "_load_scenario", return_value=scenario
            ), patch.object(pe, "_ppo_module", return_value=fake_ppo):
                with self.assertRaisesRegex(ValueError, "不可变配置"):
                    pe._evaluate(incompatible_args)

    def test_baseline_resume_uses_manifest_scenario_hash_and_skips_completed_keys(self):
        scenario = FakeScenario()
        fake_ppo = FakePPO()
        package = FakeClassicalPackage()
        common = FakeClassicalCommon()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = base / "manifest"
            with patch.object(pe, "_load_scenario", return_value=scenario):
                pe.prepare_manifest(Path("scenario.npz"), manifest, manifest_seed=123)
            argv = [
                "baselines", "--manifest", str(manifest), "--split", "validation",
                "--algorithms", "nearest_feasible", "--planner-seeds", "42",
                "--run-name", "resume_baseline",
            ]
            args = pe.build_parser().parse_args(argv)
            package.fail_on_call = 3

            def import_module(name):
                if name == "python_classical_algs":
                    return package
                if name == "python_classical_algs.common":
                    return common
                raise AssertionError(name)

            with patch.object(pe, "PAPER_RUNS_ROOT", base / "runs"), patch.object(
                pe, "_load_scenario", return_value=scenario
            ), patch.object(pe, "_ppo_module", return_value=fake_ppo), patch.object(
                pe.importlib, "import_module", side_effect=import_module
            ):
                with self.assertRaisesRegex(RuntimeError, "baseline interruption"):
                    pe._baselines(args)
            package.calls = 0
            package.fail_on_call = None
            resume_args = pe.build_parser().parse_args(argv + ["--resume-existing"])
            with patch.object(pe, "PAPER_RUNS_ROOT", base / "runs"), patch.object(
                pe, "_load_scenario", return_value=scenario
            ), patch.object(pe, "_ppo_module", return_value=fake_ppo), patch.object(
                pe.importlib, "import_module", side_effect=import_module
            ):
                run_dir = pe._baselines(resume_args)
            rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 64)
            self.assertTrue(all(row["scenario_hash"] == scenario.scenario_hash for row in rows))
            self.assertEqual(package.calls, 62)
            route = json.loads(next((run_dir / "routes").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(route["result"]["scenario_hash"], "derived-problem-hash")

    def test_dry_run_does_not_create_target_directory(self):
        scenario = FakeScenario()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "manifest"
            with patch.object(pe, "_load_scenario", return_value=scenario):
                metadata = pe.prepare_manifest(Path("scenario.npz"), root, dry_run=True)
            self.assertFalse(root.exists())
            self.assertEqual(metadata["split_counts"], pe.SPLIT_COUNTS)


if __name__ == "__main__":
    unittest.main()
