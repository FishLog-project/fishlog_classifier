"""data/clean → data/splits/{train,val,test} 70/15/15 stratified 분할.

**가장 조용히 망가지는 단계다.** 같은 물고기 사진이 train과 test에 동시에 있으면
test 정확도가 부풀려지고, 그걸 근거로 "다 됐다"고 판단하게 된다. 그래서 두 겹으로 막는다:

1. **관측(observation) 그룹**: `inat_<obsid>_<photoid>.jpg` 는 obsid가 같으면 한 개체를
   여러 각도로 찍은 것 → 같은 split에 몰아넣는다.
2. **phash 유사 그룹**: dedup.py가 남긴 `data/dup_groups.csv` 의 그룹을 같은 split로.
   (dedup에서 살아남은 대표들끼리도 임계값 아래로 비슷할 수 있다)

분할은 **하드링크**다(복사 아님) → 디스크 3배 낭비 없음. 시드 42 고정으로 재현 가능.

사용 예:
    python -m scripts.split                 # 전체 재분할
    python -m scripts.split --dry-run       # 분포만 확인
    python -m scripts.split --ratios 0.8 0.1 0.1
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

from src import config
from scripts._common import resolve_species

GROUPS_CSV = config.DATA_DIR / "dup_groups.csv"
INAT_RE = re.compile(r"^inat_(\d+)_")


def load_dup_groups() -> dict[tuple[str, str], str]:
    """(종, 파일명) → phash 그룹 키. dedup.py를 안 돌렸으면 빈 dict."""
    if not GROUPS_CSV.exists():
        return {}
    out: dict[tuple[str, str], str] = {}
    with GROUPS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[(row["species"], row["filename"])] = f"p{row['group']}"
    return out


def group_key(species: str, path: Path, dup: dict[tuple[str, str], str]) -> str:
    """이 파일이 어느 '분리 금지 그룹'에 속하는가."""
    m = INAT_RE.match(path.name)
    if m:
        return f"obs{m.group(1)}"                   # 같은 관측 = 같은 개체
    key = dup.get((species, path.name))
    if key:
        return key                                  # phash 유사 그룹
    return f"f{path.name}"                          # 단독


def split_species(name: str, ratios: tuple[float, float, float], rng: random.Random,
                  dup: dict[tuple[str, str], str]) -> dict[str, list[Path]]:
    """그룹 단위로 셔플해 순서대로 채운다.

    그룹 크기가 들쭉날쭉하므로 '목표 개수에 가장 못 미친 split'에 큰 그룹부터 넣는 식이
    아니라, 셔플 후 누적 비율로 배정한다 — 종별 300장 규모에서 충분히 균등하고 재현 쉽다.
    """
    src = config.CLEAN_DIR / name
    files = [p for p in sorted(src.iterdir())
             if p.is_file() and p.suffix.lower() in config.VALID_IMAGE_EXTS] if src.is_dir() else []

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        groups[group_key(name, p, dup)].append(p)

    keys = list(groups)
    rng.shuffle(keys)

    n_total = len(files)
    quota = {
        "train": round(n_total * ratios[0]),
        "val": round(n_total * ratios[1]),
    }
    quota["test"] = n_total - quota["train"] - quota["val"]

    out: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for k in keys:
        # 아직 할당량이 남은 split 중 '남은 비율'이 가장 큰 곳에 그룹 전체를 넣는다
        target = max(("train", "val", "test"),
                     key=lambda s: quota[s] - len(out[s]))
        out[target].extend(groups[k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="clean → splits 70/15/15 분할 (그룹 누수 방지)")
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42, help="고정 시드(기본 42, 재현성)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ratios = tuple(args.ratios)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit(f"[warn] ratios 합이 1이 아니다: {ratios}")

    targets = resolve_species(args.species)
    dup = load_dup_groups()
    if not dup:
        print("[warn] data/dup_groups.csv 가 없다 → phash 그룹 누수 방지가 꺼진다. "
              "먼저 `python -m scripts.dedup` 를 돌릴 것")

    print(f"[cfg] {len(targets)}클래스 | 비율 {ratios} | seed {args.seed}\n")
    print(f"{'종명':<8} {'train':>7} {'val':>6} {'test':>6} {'합계':>7} {'그룹':>6}")
    print("-" * 46)

    totals: defaultdict[str, int] = defaultdict(int)
    empty: list[str] = []

    for name in targets:
        rng = random.Random(f"{args.seed}:{name}")  # 종별 독립 시드 → 종을 추가해도 기존 배정 유지
        assign = split_species(name, ratios, rng, dup)
        n = sum(len(v) for v in assign.values())
        n_groups = len({group_key(name, p, dup) for v in assign.values() for p in v})
        print(f"{name:<8} {len(assign['train']):>7} {len(assign['val']):>6} "
              f"{len(assign['test']):>6} {n:>7} {n_groups:>6}")
        if n == 0:
            empty.append(name)
        for s, paths in assign.items():
            totals[s] += len(paths)
            if args.dry_run:
                continue
            dst_dir = config.SPLIT_DIRS[s] / name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for p in dst_dir.iterdir():   # 재분할 시 이전 결과 제거(하드링크만 삭제)
                if p.is_file():
                    p.unlink()
            for src in paths:
                dst = dst_dir / src.name
                try:
                    dst.hardlink_to(src)
                except (OSError, NotImplementedError):
                    import shutil
                    shutil.copy2(src, dst)

    print("-" * 46)
    print(f"{'합계':<8} {totals['train']:>7} {totals['val']:>6} {totals['test']:>6} "
          f"{sum(totals.values()):>7}")
    if empty:
        print(f"\n[warn] clean이 비어 있는 클래스 {len(empty)}개: {', '.join(empty)}")
    if args.dry_run:
        print("\n[dry-run] 파일은 만들지 않았다")
    else:
        print("\n[OK] 분할 완료")
        print("[next] python -m src.dataset --check   → 분포 확인 후 Phase 3(학습)")


if __name__ == "__main__":
    main()
