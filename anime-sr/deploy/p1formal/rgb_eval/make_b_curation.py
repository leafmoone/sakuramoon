"""Assemble rgb-eval-b-curation.json from the 4 vision passes.

- true sid per montage cell = (len, sid) order of the 160 candidate files
  (same ordering make_montages.py used); vision label reads are validated
  against it (one known misread, 5-1 of montage 0, is fixed by position).
- 7 sids per category (56 total); categories with fewer than 7 take all.
- stress_p4: 8 sids chosen from the per-sheet STRESS lists, category-diverse.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
CAND = HERE / "rgb-eval-b-cand"
CATEGORIES = [
    "EYE", "LASH", "HAIR", "TEXT", "FAINT", "HALFTONE", "FLAT", "RULE",
]

# (row, col) -> (sid, category)  for the four vision passes.
# sids as printed by the vision model; validated against position order.
PASS0 = {  # montage 0 (64 cells, 8x8)
    (1, 1): "114723 TEXT", (1, 2): "343612 HAIR", (1, 3): "475355 LASH",
    (1, 4): "1089746 EYE", (1, 5): "1175723 HAIR", (1, 6): "1194746 FLAT",
    (1, 7): "1201355 FAINT", (1, 8): "1512355 HALFTONE",
    (2, 1): "1633612 EYE", (2, 2): "1939612 LASH", (2, 3): "1967612 HAIR",
    (2, 4): "2015746 HAIR", (2, 5): "2067723 HAIR", (2, 6): "2425355 HALFTONE",
    (2, 7): "2490612 HAIR", (2, 8): "2547746 TEXT",
    (3, 1): "2568612 FAINT", (3, 2): "2621612 HAIR", (3, 3): "2631723 HAIR",
    (3, 4): "2776612 HAIR", (3, 5): "2779612 HAIR", (3, 6): "2781355 HAIR",
    (3, 7): "2781723 HALFTONE", (3, 8): "2784746 HAIR",
    (4, 1): "2788746 EYE", (4, 2): "2822723 FAINT", (4, 3): "2837355 HAIR",
    (4, 4): "2951723 EYE", (4, 5): "2986612 FAINT", (4, 6): "2994612 HALFTONE",
    (4, 7): "3029612 FAINT", (4, 8): "3050355 EYE",
    # 5-1 misread as "16397355" -> position 33 of the (len,sid) order
    (5, 1): "16397355 HAIR", (5, 2): "3123746 HAIR", (5, 3): "3153612 HAIR",
    (5, 4): "3167355 HAIR", (5, 5): "3172746 HAIR", (5, 6): "3207746 HAIR",
    (5, 7): "3383612 HAIR", (5, 8): "3397355 HAIR",
    (6, 1): "3469746 HAIR", (6, 2): "3514723 HAIR", (6, 3): "3588612 HAIR",
    (6, 4): "3629612 HAIR", (6, 5): "3641355 HAIR", (6, 6): "3664355 HAIR",
    (6, 7): "3784746 HAIR", (6, 8): "3816612 HAIR",
    (7, 1): "3908612 HAIR", (7, 2): "4034612 HAIR", (7, 3): "4064746 HAIR",
    (7, 4): "4246746 HAIR", (7, 5): "4310746 HAIR", (7, 6): "4316355 HAIR",
    (7, 7): "4328723 HAIR", (7, 8): "4352746 HAIR",
    (8, 1): "4360355 HAIR", (8, 2): "4386612 TEXT", (8, 3): "4412723 HAIR",
    (8, 4): "4463723 HAIR", (8, 5): "4504612 HAIR", (8, 6): "4536612 HAIR",
    (8, 7): "4543746 HAIR", (8, 8): "4568612 HAIR",
}
PASS1_HEAD = {  # montage 1 rows 1-5 + 6-1 (41 cells)
    (1, 1): "4611746 HAIR", (1, 2): "4613355 TEXT", (1, 3): "4618723 EYE",
    (1, 4): "4621355 HAIR", (1, 5): "4654746 FLAT", (1, 6): "4671355 HALFTONE",
    (1, 7): "4731746 FAINT", (1, 8): "4745355 LASH",
    (2, 1): "4805355 EYE", (2, 2): "4842746 HAIR", (2, 3): "4894355 TEXT",
    (2, 4): "4929612 EYE", (2, 5): "4962746 FAINT", (2, 6): "5020612 LASH",
    (2, 7): "5031746 HAIR", (2, 8): "5034746 HAIR",
    (3, 1): "5046723 HAIR", (3, 2): "5068612 HAIR", (3, 3): "5143355 HAIR",
    (3, 4): "5154746 HAIR", (3, 5): "5204612 FAINT", (3, 6): "5257355 HALFTONE",
    (3, 7): "5284612 HAIR", (3, 8): "5315723 HAIR",
    (4, 1): "5325723 TEXT", (4, 2): "5341355 HAIR", (4, 3): "5346746 TEXT",
    (4, 4): "5378612 HAIR", (4, 5): "5389746 FAINT", (4, 6): "5476723 HALFTONE",
    (4, 7): "5522355 TEXT", (4, 8): "5532612 TEXT",
    (5, 1): "5594355 EYE", (5, 2): "5603612 HAIR", (5, 3): "5645355 HAIR",
    (5, 4): "5743355 HAIR", (5, 5): "5762355 HAIR", (5, 6): "5813723 HALFTONE",
    (5, 7): "5817746 HAIR", (5, 8): "5842612 HAIR",
    (6, 1): "5879355 FAINT",
}
PASS1_TAIL = {  # b-cand-montage-1-tail.png (global 113:128), 15 cells 5x3
    (1, 1): "6250355 FLAT", (1, 2): "6322723 EYE", (1, 3): "6324355 TEXT",
    (2, 1): "6325612 HAIR", (2, 2): "6349612 FAINT", (2, 3): "6432355 HALFTONE",
    (3, 1): "6489355 EYE", (3, 2): "6557612 HAIR", (3, 3): "6560723 TEXT",
    (4, 1): "6596612 HALFTONE", (4, 2): "6633746 LASH", (4, 3): "6641746 TEXT",
    (5, 1): "6688746 RULE", (5, 2): "6714612 FLAT", (5, 3): "6719612 HAIR",
}
PASS1_TAIL2 = {  # b-cand-montage-1-tail2.png (global 105:113), 8 cells 4x2
    (1, 1): "5917612 EYE", (1, 2): "5923746 HAIR",
    (2, 1): "6045723 TEXT", (2, 2): "6162355 FLAT",
    (3, 1): "6177355 FAINT", (3, 2): "6182723 HALFTONE",
    (4, 1): "6195746 TEXT", (4, 2): "6214612 HALFTONE",
}
PASS2 = {  # montage 2 (32 cells, 4x8)
    (1, 1): "6726612 EYE", (1, 2): "6727723 HAIR", (1, 3): "6761355 HALFTONE",
    (1, 4): "7416355 TEXT", (1, 5): "7436355 FLAT", (1, 6): "7436612 LASH",
    (1, 7): "7442355 FLAT", (1, 8): "7534355 FAINT",
    (2, 1): "7566746 TEXT", (2, 2): "7570723 FAINT", (2, 3): "7584612 EYE",
    (2, 4): "7674746 TEXT", (2, 5): "7724355 HAIR", (2, 6): "7737355 HAIR",
    (2, 7): "7758355 RULE", (2, 8): "7840723 HALFTONE",
    (3, 1): "7946723 HAIR", (3, 2): "7958355 HAIR", (3, 3): "7977612 FAINT",
    (3, 4): "7994612 TEXT", (3, 5): "8007612 HAIR", (3, 6): "8052355 HALFTONE",
    (3, 7): "8063355 TEXT", (3, 8): "8096612 FAINT",
    (4, 1): "8112612 RULE", (4, 2): "8147746 HAIR", (4, 3): "8189355 HAIR",
    (4, 4): "8247723 TEXT", (4, 5): "8257746 HAIR", (4, 6): "8276746 TEXT",
    (4, 7): "8322723 HALFTONE", (4, 8): "8357746 TEXT",
}

STRESS_POOL = [
    "114723", "475355", "2425355", "2781723", "2784746", "2788746", "2837355",
    "6761355", "7724355", "7994612", "8007612", "8247723", "8276746",
    "8322723", "8357746",
    "6596612", "6633746", "6688746",
    "6045723", "6195746",
]


def main() -> None:
    sids = sorted((p.stem for p in CAND.glob("*.webp")), key=lambda s: (len(s), s))
    assert len(sids) == 160

    # position -> (sid, category)
    pos = {}
    for r in range(1, 9):
        for c in range(1, 9):
            pos[(0, r, c)] = sids[(r - 1) * 8 + (c - 1)]
            pos[(1, 1 + (r - 1), c)] = None  # placeholder, filled below
    # montage 0 = sids[0:64]; montage 1 = sids[64:128]; montage 2 = sids[128:160]
    order = {}
    for r in range(1, 9):
        for c in range(1, 9):
            order[(0, r, c)] = sids[(r - 1) * 8 + (c - 1)]
            order[(1, r, c)] = sids[64 + (r - 1) * 8 + (c - 1)]
    for r in range(1, 5):
        for c in range(1, 9):
            order[(2, r, c)] = sids[128 + (r - 1) * 8 + (c - 1)]
    # tail sheets: row-major over their own cell layouts
    for i, (r, c) in enumerate(
        [(r, c) for r in range(1, 6) for c in range(1, 4)]
    ):
        order[(1, 100 + i, c)] = sids[113 + i]  # tail sheet (113:128)
    for i, (r, c) in enumerate([(r, c) for r in range(1, 5) for c in range(1, 3)]):
        order[(1, 200 + i, c)] = sids[105 + i]  # tail2 sheet (105:113)

    # map pass cells -> (true_sid, cat)
    def cells_for(passd, keymap):
        out = []
        for key, val in passd.items():
            sid_v, cat = val.split()
            (sheet, r, c) = keymap(key)
            true_sid = order[sheet, r, c]
            if true_sid != sid_v:
                print(f"  [fix] sheet={sheet} cell=({r},{c}) "
                      f"vision='{sid_v}' -> true='{true_sid}'")
            out.append((true_sid, cat))
        return out

    m0 = lambda k: (0, k[0], k[1])
    m1_head = lambda k: (1, k[0], k[1])
    m1_tail = lambda k: (1, 100 + (k[0] - 1) * 3 + (k[1] - 1), k[1])
    m1_tail2 = lambda k: (1, 200 + (k[0] - 1) * 2 + (k[1] - 1), k[1])
    m2 = lambda k: (2, k[0], k[1])

    pairs = (
        cells_for(PASS0, m0) + cells_for(PASS1_HEAD, m1_head)
        + cells_for(PASS1_TAIL, m1_tail) + cells_for(PASS1_TAIL2, m1_tail2)
        + cells_for(PASS2, m2)
    )
    assert len(pairs) == 160, f"expected 160 pairs, got {len(pairs)}"
    seen = [s for s, _ in pairs]
    assert len(set(seen)) == 160, "duplicate sids across passes"
    cat_of = dict(pairs)

    counts = {c: 0 for c in CATEGORIES}
    for c in cat_of.values():
        assert c in counts, f"unknown category {c}"
        counts[c] += 1
    print("category counts:", {c: n for c, n in counts.items()})

    by_cat = {c: [] for c in CATEGORIES}
    for sid, c in pairs:
        by_cat[c].append(sid)
    by_cat = {c: sorted(v, key=lambda s: (len(s), s)) for c, v in by_cat.items()}

    selected = {}
    for c in CATEGORIES:
        selected[c] = by_cat[c][:7]
        if len(by_cat[c]) < 7:
            print(f"  [short] {c}: only {len(by_cat[c])} available")

    sel_sids = {s for v in selected.values() for s in v}
    stress = [s for s in STRESS_POOL if s in sel_sids][:8]
    while len(stress) < 8:  # top up from non-selected pool if needed
        for c in CATEGORIES:
            for s in by_cat[c]:
                if s not in sel_sids and s not in stress:
                    sel_sids.add(s)
                    stress.append(s)
                    break
            if len(stress) == 8:
                break
        else:
            break

    out = []
    for c in CATEGORIES:
        for s in selected[c]:
            out.append({"sid": s, "category": c, "stress_p4": s in stress})
    out.sort(key=lambda r: (len(r["sid"]), r["sid"]))
    dst = HERE / "rgb-eval-b-curation.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    n_stress = sum(1 for r in out if r["stress_p4"])
    print(f"wrote {dst.name}: {len(out)} entries, {n_stress} stress_p4")
    print("stress sids:", stress)
    for c in CATEGORIES:
        print(f"  {c:9s} -> {selected[c]}")


if __name__ == "__main__":
    main()
