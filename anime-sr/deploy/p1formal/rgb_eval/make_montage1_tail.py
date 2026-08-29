"""Crop-style montage for the 16 sids that the full-sheet vision pass missed
(bottom 3 rows of b-cand-montage-1: positions 49-64 of the (len, sid) order).
Writes b-cand-montage-1-tail.png (4x4 grid, same labels).
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
CAND = HERE / "rgb-eval-b-cand"


def get_font(size: int):
    for name in ("arial.ttf", "msyh.ttc", "DejaVuSans.ttf"):
        p = Path(f"C:/Windows/Fonts/{name}")
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                pass
    return ImageFont.load_default()


def main() -> None:
    sids = sorted((p.stem for p in CAND.glob("*.webp")), key=lambda s: (len(s), s))
    # montage 1 = sids[64:128].  Vision pass 1 captured rows 1-5 (global
    # 64:104) + cell 6-1 (global 104 = 5879355).  Pass 2 covered global
    # 113:128 (15 sids, b-cand-montage-1-tail.png).  Still missing: global
    # 105:113 (8 sids: 6-2..6-8 + 7-1).
    chunk = sids[105:113]
    assert len(chunk) == 8, f"expected 8, got {len(chunk)}"
    for i, s in enumerate(chunk):
        print(f"{i + 1:2d}. {s}")
    CELL = 240
    PAD = 2
    COLS = 2
    font = get_font(14)
    w = COLS * (CELL + PAD) + PAD
    h = 4 * (CELL + PAD) + PAD
    sheet = Image.new("RGB", (w, h), (24, 24, 24))
    dr = ImageDraw.Draw(sheet)
    for i, sid in enumerate(chunk):
        r, c = divmod(i, COLS)
        im = Image.open(CAND / f"{sid}.webp").convert("RGB")
        ow, oh = im.size
        s2 = CELL // 2
        im = im.resize((s2, s2))
        x = PAD + c * (CELL + PAD)
        y = PAD + r * (CELL + PAD)
        sheet.paste(im, (x + (CELL - s2) // 2, y + (CELL - s2) // 2))
        dr.text((x + 3, y + CELL - 15), f"{sid} {ow}x{oh}", fill=(255, 255, 0), font=font)
        dr.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(60, 60, 60))
    out = HERE / "b-cand-montage-1-tail2.png"
    sheet.save(out)
    print(f"{out.name}: {out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
