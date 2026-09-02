"""重新读取全交付清单中的每个文件并核验 SHA-256。"""

from pathlib import Path
import hashlib
import json


ROOT = Path(r"C:\Users\xsp\Desktop\DRL代码\paper_runs\multimap_v3_2_14\figures\paper_redraw_multibackend_v7_drones_style_fullsuite")


def main() -> None:
    manifest = json.loads((ROOT / "manifests" / "full_delivery_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        observed = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if observed != item["sha256"]:
            failures.append({"path": item["path"], "expected": item["sha256"], "observed": observed})
    report = {"passed": not failures, "checked_files": len(manifest["files"]), "failures": failures}
    (ROOT / "qa" / "manifest_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
