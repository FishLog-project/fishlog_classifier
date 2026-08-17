"""사람 검수 보조 — **볼 것만** 골라 폴더로 만들고, 지운 결과를 되돌릴 수 없게 확정한다.

왜 필요한가: `data/clean/<종>` 에는 iNat 사진(학명 검증본, 검수 불필요)과 검색 수집분
(라벨 오염 있음)이 섞여 있다. 전부 훑으면 7천 장이지만 실제로 볼 것은 검색분뿐이다.
이 스크립트가 검색분만 `data/review/<종>/` 에 하드링크로 모아준다(디스크 추가 0).

작업 흐름:
    python -m scripts.review --make --species confusable   # 검수 폴더 생성
    (탐색기에서 data/review/<종>/ 를 열고 '아주 큰 아이콘'으로 훑으며 나쁜 사진 삭제)
    python -m scripts.review --record                      # 삭제 결과를 거부 목록에 확정
    python -m scripts.review --clear                       # 검수 폴더 정리(선택)

`--record` 를 돌리면 지운 파일이 `data/rejected.txt` 에 기록되어 dedup을 다시 돌려도
되살아나지 않는다. **이걸 안 돌리면 검수가 헛수고가 된다.**
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from src import config
from scripts._common import append_rejected, resolve_species

REVIEW_DIR = config.DATA_DIR / "review"
REVIEW_MANIFEST = config.DATA_DIR / "review_manifest.csv"


def images_in(directory: Path, prefix: str = "") -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.name.startswith(prefix)
                  and p.suffix.lower() in config.VALID_IMAGE_EXTS)


def make(targets: list[str], prefix: str) -> None:
    """clean의 해당 파일들을 review 폴더에 하드링크한다."""
    rows: list[tuple[str, str]] = []
    print(f"{'종명':<8}{'검수 대상':>9}")
    print("-" * 18)

    for name in targets:
        src_files = images_in(config.CLEAN_DIR / name, prefix)
        dst_dir = REVIEW_DIR / name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in images_in(dst_dir):     # 이전 검수 폴더 정리 (하드링크만 삭제)
            p.unlink()
        for src in src_files:
            dst = dst_dir / src.name
            try:
                dst.hardlink_to(src)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copy2(src, dst)
            rows.append((name, src.name))
        print(f"{name:<8}{len(src_files):>9}")

    REVIEW_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_MANIFEST.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "filename"])
        w.writerows(rows)

    print("-" * 18)
    print(f"{'합계':<8}{len(rows):>9}\n")
    print(f"[OK] 검수 폴더: {REVIEW_DIR}")
    print("     탐색기에서 열고 '보기 → 아주 큰 아이콘'으로 훑으며 나쁜 사진을 삭제할 것")
    print("     (여기서 지워도 raw 원본은 남는다 — 되살릴 수 있다)")
    print("[next] 다 끝나면: python -m scripts.review --record")


def record(targets: list[str] | None) -> None:
    """검수 폴더에서 사라진 파일 = 사람이 버린 것 → 거부 목록에 확정."""
    if not REVIEW_MANIFEST.exists():
        raise SystemExit("[warn] review_manifest.csv 가 없다. --make 를 먼저 돌려야 한다")

    with REVIEW_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        manifest = [(r["species"], r["filename"]) for r in csv.DictReader(f)
                    if r.get("species") and r.get("filename")]

    scope = set(targets) if targets else {sp for sp, _ in manifest}
    gone: list[tuple[str, str]] = []
    for name in sorted(scope):
        present = {p.name for p in images_in(REVIEW_DIR / name)}
        gone += [(sp, fn) for sp, fn in manifest if sp == name and fn not in present]

    if not gone:
        print("[OK] 지운 파일이 없다 (검수 전이거나 전부 유지)")
        return

    added = append_rejected(gone)
    by_species: dict[str, int] = defaultdict(int)
    for sp, _ in gone:
        by_species[sp] += 1

    print(f"[OK] 삭제 {len(gone)}건 확인 → 거부 목록에 {added}건 추가")
    for sp, n in sorted(by_species.items(), key=lambda kv: -kv[1]):
        total = sum(1 for s, _ in manifest if s == sp)
        print(f"  {sp:<8} {n:>4} / {total} 삭제 ({n / total:.0%})")
    print("\n[next] python -m scripts.dedup   (clean 재구성 — 지운 파일은 제외된다)")
    print("[next] python -m scripts.split")


def clear() -> None:
    if not REVIEW_DIR.exists():
        print("[OK] 검수 폴더가 없다")
        return
    n = 0
    for d in REVIEW_DIR.iterdir():
        if d.is_dir():
            for p in images_in(d):
                p.unlink()
                n += 1
            d.rmdir()
    REVIEW_DIR.rmdir()
    print(f"[OK] 검수 폴더 정리 완료 (하드링크 {n}개 삭제, raw 원본은 그대로)")


def main() -> None:
    ap = argparse.ArgumentParser(description="검수 대상만 모아 보여주고, 삭제 결과를 확정한다")
    ap.add_argument("--make", action="store_true", help="검수 폴더 생성")
    ap.add_argument("--record", action="store_true", help="삭제 결과를 거부 목록에 확정")
    ap.add_argument("--clear", action="store_true", help="검수 폴더 삭제")
    ap.add_argument("--species", nargs="*", default=None,
                    help="대상 클래스. 'confusable' 이면 혼동 쌍 전체 (기본: 전체)")
    ap.add_argument("--prefix", default="web_",
                    help="검수할 파일 접두사. 기본 web_(검색 수집분만). "
                         "iNat 사진까지 보려면 '' 로 지정")
    args = ap.parse_args()

    if not (args.make or args.record or args.clear):
        raise SystemExit("--make / --record / --clear 중 하나를 지정할 것")

    if args.clear:
        clear()
        return
    targets = resolve_species(args.species)
    if args.make:
        make(targets, args.prefix)
    if args.record:
        record(targets if args.species else None)


if __name__ == "__main__":
    main()
