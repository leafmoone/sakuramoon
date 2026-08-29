"""Build labeled thumbnail montages over the 160 Set-B candidate webps.

Usage: python make_montages.py
Writes b-cand-montage-{0,1,2}.png (8x8 grids, 160px cells, sid + WxH label).
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
CAND = HERE / "rgb-eval-b-cand"
OUT = HERE
N_COLS = 8
CELL = 240
PAD = 2


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
    assert len(sids) == 160, f"expected 160 candidates, got {len(sids)}"
    font = get_font(14)
    per = N_COLS * N_COLS  # 64
    for mi in range(0, len(sids), per):
        chunk = sids[mi : mi + per]
        w = N_COLS * (CELL + PAD) + PAD
        h = N_COLS * (CELL + PAD) + PAD
        sheet = Image.new("RGB", (w, h), (24, 24, 24))
        dr = ImageDraw.Draw(sheet)
        for i, sid in enumerate(chunk):
            r, c = divmod(i, N_COLS)
            im = Image.open(CAND / f"{sid}.webp").convert("RGB")
            ow, oh = im.size
            s = CELL // 2
            im = im.resize((s, s))
            x = PAD + c * (CELL + PAD)
            y = PAD + r * (CELL + PAD)
            sheet.paste(im, (x + (CELL - s) // 2, y + (CELL - s) // 2))
            label = f"{sid} {ow}x{oh}"
            dr.text((x + 3, y + CELL - 15), label, fill=(255, 255, 0), font=font)
            dr.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(60, 60, 60))
        out = OUT / f"b-cand-montage-{mi // per}.png"
        sheet.save(out)
        print(f"{out.name}: {len(chunk)} thumbs -> {out.stat().st_size} bytes")


if __name__ == "__main__":
    main()

