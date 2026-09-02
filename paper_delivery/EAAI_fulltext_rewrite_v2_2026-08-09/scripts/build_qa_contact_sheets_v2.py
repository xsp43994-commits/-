from __future__ import annotations

"""为逐页/逐图人工审查生成可读联系表，不修改交付件。"""

from pathlib import Path
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
RENDER = ROOT / "qa" / "docx_render"
OUT = ROOT / "qa" / "contact_sheets"
FIG_OUT = ROOT / "qa" / "figure_contact_sheets"


def sheets(paths: list[Path], out_dir: Path, prefix: str, per_sheet: int = 4, thumb_w: int = 1000) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(paths), per_sheet):
        batch = paths[start:start + per_sheet]
        prepared = []
        for p in batch:
            im = Image.open(p).convert("RGB")
            h = int(im.height * thumb_w / im.width)
            prepared.append((p, im.resize((thumb_w, h))))
        cell_h = max(im.height for _, im in prepared) + 60
        cols = 2; rows = (len(prepared) + 1) // 2
        canvas = Image.new("RGB", (cols * thumb_w + 40, rows * cell_h + 40), "#D5D8DC")
        draw = ImageDraw.Draw(canvas)
        for idx, (p, im) in enumerate(prepared):
            x = 20 + (idx % cols) * thumb_w
            y = 20 + (idx // cols) * cell_h
            canvas.paste(im, (x, y + 36))
            draw.text((x + 8, y + 8), p.name, fill="#111111")
        canvas.save(out_dir / f"{prefix}_{start // per_sheet + 1:02d}.jpg", quality=90)


def main() -> None:
    for folder in sorted(p for p in RENDER.iterdir() if p.is_dir()):
        pages = sorted(folder.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
        sheets(pages, OUT / folder.name, folder.name, per_sheet=4, thumb_w=1000)

    figures = []
    for sub in ("main", "supplementary", "showcase"):
        figures.extend(sorted((ROOT / "figures" / "submission" / sub).glob("*_english.png")))
    figures.extend([
        ROOT / "figures" / "submission" / "main" / "F01_method_and_evaluation_workflow.png",
        ROOT / "figures" / "submission" / "main" / "M05_online_planning_time_ECDF_repaired.png",
        ROOT / "figures" / "submission" / "showcase" / "V02_fixed_DSM_route_repaired.png",
    ])
    figures = sorted(set(p for p in figures if p.exists()))
    sheets(figures, FIG_OUT, "english_figures", per_sheet=4, thumb_w=1000)
    print(f"document_pages={sum(len(list(p.glob('page-*.png'))) for p in RENDER.iterdir() if p.is_dir())}")
    print(f"figures={len(figures)}")


if __name__ == "__main__":
    main()
