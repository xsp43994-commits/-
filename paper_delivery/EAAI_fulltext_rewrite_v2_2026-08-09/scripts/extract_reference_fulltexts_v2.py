"""Extract searchable text from legally accessible reference PDFs for citation audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pypdf import PdfReader


# 关键路径参数：输入仅包含开放获取参考文献，输出文本用于全文命题核验。
DELIVERY_ROOT = Path(r"C:\Users\xsp\Desktop\DRL代码\paper_delivery\EAAI_fulltext_rewrite_v2_2026-08-09")
PDF_SOURCES = (
    (DELIVERY_ROOT / "literature" / "reference_fulltexts_open", True),
    (DELIVERY_ROOT / "literature" / "reference_fulltexts_accessed_link_only", False),
)
TEXT_DIR = DELIVERY_ROOT / "literature" / "reference_fulltexts_text"
MANIFEST_PATH = DELIVERY_ROOT / "literature" / "reference_fulltexts_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for pdf_dir, redistributable in PDF_SOURCES:
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            reader = PdfReader(str(pdf_path))
            pages: list[str] = []
            extraction_errors: list[str] = []
            for index, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:  # 单页异常不应阻断其余全文的核验。
                    text = ""
                    extraction_errors.append(f"page {index}: {exc}")
                pages.append(f"\n\n===== PAGE {index} =====\n\n{text}")
            output_path = TEXT_DIR / f"{pdf_path.stem}.txt"
            output_path.write_text("".join(pages), encoding="utf-8")
            records.append(
                {
                    "file": pdf_path.name,
                    "sha256": sha256(pdf_path),
                    "pages": len(reader.pages),
                    "text_chars": sum(len(page) for page in pages),
                    "text_file": str(output_path),
                    "redistributable_in_delivery": redistributable,
                    "extraction_errors": extraction_errors,
                }
            )
    MANIFEST_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"pdfs": len(records), "manifest": str(MANIFEST_PATH)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
