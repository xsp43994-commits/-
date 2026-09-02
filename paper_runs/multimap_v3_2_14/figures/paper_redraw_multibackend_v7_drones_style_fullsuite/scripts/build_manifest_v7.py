"""生成排除 manifests/ 自身的全交付 SHA-256 清单。"""

from pathlib import Path
import hashlib
import json


ROOT = Path(r"C:\Users\xsp\Desktop\DRL代码\paper_runs\multimap_v3_2_14\figures\paper_redraw_multibackend_v7_drones_style_fullsuite")


def main() -> None:
    files = []
    for path in sorted(item for item in ROOT.rglob("*") if item.is_file()):
        relative = path.relative_to(ROOT)
        if "manifests" in relative.parts or relative.as_posix() == "qa/manifest_verification.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": relative.as_posix(), "sha256": digest, "bytes": path.stat().st_size})
    payload = {"schema_version": 1, "self_excluded": True,
               "excluded": ["manifests/", "qa/manifest_verification.json"],
               "file_count": len(files), "files": files}
    target = ROOT / "manifests"
    target.mkdir(parents=True, exist_ok=True)
    (target / "full_delivery_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "SHA256SUMS_v7.txt").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in files), encoding="utf-8")
    print(json.dumps({"full_file_count": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
