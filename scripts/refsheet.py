"""혼동 쌍 비교 참고표 생성 — 검수할 때 옆에 띄워놓고 보는 이미지.

라벨이 **검증된 iNat 사진만** 골라 종별로 한 줄씩 붙인다. 어종을 모르는 사람이
검수할 때 "우럭이 이렇게 생겼구나"를 눈으로 익히는 용도다.

사용 예:
    python -m scripts.refsheet                    # 혼동 쌍 6그룹 전부
    python -m scripts.refsheet --species 우럭 볼락  # 원하는 종만 한 장으로
    python -m scripts.refsheet --per-species 12   # 종당 사진 수

결과: reports/ref_<종1>_<종2>.jpg
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src import config
from scripts._common import resolve_species

TILE = 210          # 타일 한 변(px)
LABEL_W = 150       # 왼쪽 종명 칸 너비
PAD = 4

# 한글 라벨용 폰트. cv2.putText는 한글을 못 그린다 → PIL + 맑은 고딕.
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\malgun.ttf"),
    Path(r"C:\Windows\Fonts\malgunbd.ttf"),
    Path(r"C:\Windows\Fonts\gulim.ttc"),
]


def load_font(size: int) -> ImageFont.ImageFont:
    for p in FONT_CANDIDATES:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    return ImageFont.load_default()   # 한글이 깨질 수 있다(폰트 없는 환경)


def pick_images(species: str, n: int, rng: random.Random) -> list[Path]:
    """iNat 사진 우선(라벨 검증본). 부족하면 검색분으로 채운다."""
    d = config.CLEAN_DIR / species
    if not d.is_dir():
        return []
    inat = [p for p in sorted(d.iterdir())
            if p.name.startswith("inat_") and p.suffix.lower() in config.VALID_IMAGE_EXTS]
    rng.shuffle(inat)
    picked = inat[:n]
    if len(picked) < n:
        web = [p for p in sorted(d.iterdir())
               if p.name.startswith("web_") and p.suffix.lower() in config.VALID_IMAGE_EXTS]
        rng.shuffle(web)
        picked += web[:n - len(picked)]
    return picked


def square_thumb(path: Path, size: int) -> Image.Image | None:
    """가운데를 정사각으로 잘라 축소. 물고기가 보통 화면 중앙에 있다."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            return im.resize((size, size), Image.LANCZOS)
    except Exception:
        return None


def build_sheet(group: tuple[str, ...], per_species: int, rng: random.Random,
                out_dir: Path) -> Path | None:
    rows = []
    for name in group:
        paths = pick_images(name, per_species, rng)
        thumbs = [t for t in (square_thumb(p, TILE) for p in paths) if t is not None]
        if thumbs:
            rows.append((name, thumbs))
    if not rows:
        return None

    cols = max(len(t) for _, t in rows)
    sheet_w = LABEL_W + cols * (TILE + PAD) + PAD
    sheet_h = len(rows) * (TILE + PAD) + PAD + 34   # 아래 안내문 자리
    sheet = Image.new("RGB", (sheet_w, sheet_h), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    font_name = load_font(30)
    font_small = load_font(16)

    for r, (name, thumbs) in enumerate(rows):
        y = PAD + r * (TILE + PAD)
        sp = config.SPECIES[name]
        draw.text((8, y + TILE // 2 - 26), name, fill=(20, 20, 20), font=font_name)
        draw.text((8, y + TILE // 2 + 8), sp.scientific[:20], fill=(110, 110, 110),
                  font=font_small)
        for c, th in enumerate(thumbs):
            sheet.paste(th, (LABEL_W + PAD + c * (TILE + PAD), y))

    draw.text((8, sheet_h - 26),
              "iNaturalist 검증 사진 (학명 기준). 검수할 때 이 생김새와 비교할 것.",
              fill=(90, 90, 90), font=font_small)

    out = out_dir / f"ref_{'_'.join(group)}.jpg"
    sheet.save(out, quality=88)
    print(f"[OK] {out.name}  ({' vs '.join(group)}, 종당 {cols}장)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="혼동 쌍 비교 참고표 생성")
    ap.add_argument("--species", nargs="*", default=None,
                    help="종명들을 주면 그 종만 한 장으로 만든다 (기본: 혼동 쌍 6그룹)")
    ap.add_argument("--per-species", type=int, default=10, help="종당 사진 수 (기본 10)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = config.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    groups = ([tuple(resolve_species(args.species))] if args.species
              else config.CONFUSABLE_GROUPS)
    made = [p for g in groups if (p := build_sheet(g, args.per_species, rng, out_dir))]

    print(f"\n[done] {len(made)}장 생성 → {out_dir}")
    print("[next] 검수 중 이 파일을 열어두고 비교할 것")


if __name__ == "__main__":
    main()
