from __future__ import annotations

"""将全部Word导出的PDF逐页转为PNG，并生成二页联系表供人工逐页检查。"""

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
RENDER = ROOT / "qa" / "docx_render"
CONTACT = ROOT / "qa" / "contact_sheets"
PDFTOPPM = Path(r"C:\Users\xsp\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe")


def page_number(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[-1])


def render_pdf(folder: Path) -> list[Path]:
    pdfs = list(folder.glob("*.pdf"))
    if len(pdfs) != 1:
        raise RuntimeError(f"Expected one PDF in {folder}, found {len(pdfs)}")
    for old in folder.glob("page-*.png"):
        old.unlink()
    subprocess.run([str(PDFTOPPM), "-png", "-r", "130", str(pdfs[0]), str(folder / "page")], check=True)
    return sorted(folder.glob("page-*.png"), key=page_number)


def content_bbox(im: Image.Image) -> tuple[int, int, int, int] | None:
    rgb = im.convert("RGB")
    bg = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, bg).convert("L").point(lambda x: 255 if x > 10 else 0)
    return diff.getbbox()


def make_contacts(name: str, pages: list[Path]) -> list[str]:
    out = CONTACT / name; out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    outputs: list[str] = []
    for start in range(0, len(pages), 2):
        batch = pages[start:start + 2]; thumbs = []
        for p in batch:
            im = Image.open(p).convert("RGB")
            width = 1150; height = round(im.height * width / im.width)
            thumbs.append((p, im.resize((width, height))))
        cell_h = max(im.height for _, im in thumbs) + 52
        canvas = Image.new("RGB", (1200 * len(thumbs), cell_h + 20), "#D9DDE1")
        draw = ImageDraw.Draw(canvas)
        for idx, (p, im) in enumerate(thumbs):
            x = 25 + idx * 1200; canvas.paste(im, (x, 52)); draw.text((x + 8, 16), p.name, fill="#111111")
        target = out / f"{name}_{start // 2 + 1:02d}.jpg"; canvas.save(target, quality=92)
        outputs.append(str(target.relative_to(ROOT)))
    return outputs


def main() -> None:
    records = []
    for folder in sorted(p for p in RENDER.iterdir() if p.is_dir()):
        pages = render_pdf(folder)
        contacts = make_contacts(folder.name, pages)
        for page in pages:
            im = Image.open(page)
            bbox = content_bbox(im)
            records.append({
                "document": folder.name,
                "page": page_number(page),
                "pixels": list(im.size),
                "content_bbox": list(bbox) if bbox else None,
                "blank": bbox is None,
            })
        print(f"{folder.name}: {len(pages)} pages, {len(contacts)} contact sheets")
    (ROOT / "qa" / "page_render_metrics_v4.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
