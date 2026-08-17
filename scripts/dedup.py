"""중복 제거 + 손상 파일 걸러내기: data/raw → data/clean

왜 raw를 그대로 두는가: 수집은 비싸다(iNat 레이트리밋, 크롤링 시간). raw는 **불변**으로 두고
clean에 **하드링크**를 만든다 → 디스크 추가 사용 0, 정제 기준을 바꿔도 재수집 불필요.
사람 검수는 clean에서 하고, 잘못 지웠으면 이 스크립트를 다시 돌리면 복구된다.

무엇을 거르는가:
1. 열리지 않는/너무 작은/극단 비율 파일
2. **정확히 같은 파일** (내용 md5)
3. **거의 같은 사진** (phash 해밍거리 <= --threshold): 리사이즈·워터마크·크롭 변형본
   → 같은 사진이 train과 test에 나뉘어 들어가면 test 정확도가 부풀려진다.

`--report-cross`: 클래스 간 중복도 검사한다. 같은 사진이 붕어와 잉어에 동시에 있으면
라벨이 하나는 틀렸다는 뜻이므로 **반드시 사람이 봐야 한다**(자동 삭제하지 않는다).

사용 예:
    python -m scripts.dedup                      # 전체 클래스, clean 재구성
    python -m scripts.dedup --species 붕어 잉어 --report-cross
    python -m scripts.dedup --threshold 6        # 더 공격적으로 제거
    python -m scripts.dedup --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

from src import config
from scripts._common import (
    MIN_SIDE,
    append_rejected,
    load_clean_manifest,
    load_rejected,
    resolve_species,
    write_clean_manifest,
)

try:
    import imagehash
except ImportError:  # pragma: no cover
    raise SystemExit("imagehash가 없다 → pip install -r requirements.txt")

# phash 그룹 정보를 split.py에 넘긴다. 유사본이 train/test로 쪼개지는 것을 막는 근거 파일.
GROUPS_CSV = config.DATA_DIR / "dup_groups.csv"

# `inat_<obsid>_<photoid>.jpg` 의 photoid 추출 (같은 사진이 여러 관측에 붙는 경우 탐지용)
INAT_PHOTO_RE = re.compile(r"^inat_\d+_(\d+)\.")


def iter_images(d: Path):
    if not d.is_dir():
        return
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in config.VALID_IMAGE_EXTS:
            yield p


def file_md5(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def image_fingerprint(p: Path) -> tuple[imagehash.ImageHash, tuple[int, int]] | None:
    """(phash, 크기). 열 수 없거나 학습에 못 쓸 이미지면 None."""
    try:
        with Image.open(p) as img:
            img.load()                 # 잘린 파일은 여기서 터진다
            size = img.size
            if min(size) < MIN_SIDE:
                return None
            if max(size) / max(1, min(size)) > 4.0:
                return None
            return imagehash.phash(img.convert("RGB")), size
    except Exception:
        return None


def link_or_copy(src: Path, dst: Path) -> None:
    """하드링크 우선(디스크 절약). 실패 시 복사로 대체."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.hardlink_to(src)
    except (OSError, NotImplementedError):
        import shutil
        shutil.copy2(src, dst)


def pick_best(paths: list[Path], sizes: dict[Path, tuple[int, int]]) -> Path:
    """유사 그룹의 대표 1장: iNat 사진 우선(라벨 신뢰도) → 해상도 큰 것 → 이름 순."""
    def key(p: Path):
        is_inat = p.name.startswith("inat_")
        w, h = sizes.get(p, (0, 0))
        return (0 if is_inat else 1, -(w * h), p.name)
    return sorted(paths, key=key)[0]


def dedup_species(name: str, *, threshold: int, dry_run: bool, rebuild: bool,
                  rejected: set[tuple[str, str]]) -> dict[str, int]:
    """한 클래스의 raw를 정제해 clean에 하드링크한다."""
    raw_dir = config.RAW_DIR / name
    clean_dir = config.CLEAN_DIR / name
    stats = {"raw": 0, "broken": 0, "exact": 0, "near": 0, "kept": 0, "rejected": 0}

    fingerprints: dict[Path, imagehash.ImageHash] = {}
    sizes: dict[Path, tuple[int, int]] = {}
    seen_md5: dict[str, Path] = {}

    for p in iter_images(raw_dir):
        stats["raw"] += 1
        if (name, p.name) in rejected:   # 검수/프리필터에서 이미 버린 파일
            stats["rejected"] += 1
            continue
        fp = image_fingerprint(p)
        if fp is None:
            stats["broken"] += 1
            continue
        md5 = file_md5(p)
        if md5 in seen_md5:
            stats["exact"] += 1
            continue
        seen_md5[md5] = p
        fingerprints[p], sizes[p] = fp[0], fp[1]

    # 근접 중복 그룹핑 (union-find 없이: 대표 해시와 비교하는 그리디 클러스터링).
    # 종당 수백~수천 장 규모라 O(n·k)로 충분하다.
    groups: list[list[Path]] = []
    reps: list[imagehash.ImageHash] = []
    for p, h in sorted(fingerprints.items(), key=lambda kv: kv[0].name):
        for i, rep in enumerate(reps):
            if (h - rep) <= threshold:
                groups[i].append(p)
                break
        else:
            reps.append(h)
            groups.append([p])

    keep: list[tuple[Path, int]] = []  # (대표 경로, 그룹 인덱스)
    for gi, g in enumerate(groups):
        best = pick_best(g, sizes)
        keep.append((best, gi))
        stats["near"] += len(g) - 1

    stats["kept"] = len(keep)

    if not dry_run:
        if rebuild and clean_dir.is_dir():
            # 하드링크만 지운다(raw는 안전). 사람이 검수하며 지운 파일도 되살아난다.
            for p in iter_images(clean_dir):
                p.unlink()
        clean_dir.mkdir(parents=True, exist_ok=True)
        for src, _ in keep:
            link_or_copy(src, clean_dir / src.name)

    flag = ""
    if stats["kept"] < config.MVP_TARGET_IMAGES:
        flag = f"  ← MVP({config.MVP_TARGET_IMAGES}) 미달"
    print(f"{name:<8} raw {stats['raw']:>5} → clean {stats['kept']:>5}  "
          f"(손상 {stats['broken']:>3}, 완전중복 {stats['exact']:>4}, "
          f"유사 {stats['near']:>4}, 기거부 {stats['rejected']:>4}){flag}")
    return stats | {"groups": groups, "kept_names": [p.name for p, _ in keep]}


def find_label_conflicts(kept: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    """**같은 사진이 두 클래스에 들어간 경우**를 찾는다 → 최소 하나는 라벨이 틀렸다.

    두 가지 키로 본다. 둘 다 정확 비교라서 오탐이 없다:
      1. 파일명 — 검색 수집분은 URL 해시라서 같은 사진이면 파일명도 같다.
      2. iNat photo_id — 같은 사진이 서로 다른 관측(다른 종으로 동정)에 붙어 있는 경우.
         `inat_<obsid>_<photoid>.jpg` 에서 photoid가 같으면 동일 사진이다.

    (phash 근접 비교는 쓰지 않는다. '흰 배경에 물고기 한 마리' 같은 단순 구도가
     서로 다른 종끼리 거리 2~6으로 붙어 오탐이 대량 발생한다 — 2026-08-14 확인)
    """
    by_key: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for species, fname in kept:
        m = INAT_PHOTO_RE.match(fname)
        key = f"photo:{m.group(1)}" if m else f"file:{fname}"
        by_key[key].append((species, fname))

    return {k: v for k, v in by_key.items()
            if len({sp for sp, _ in v}) > 1}


def hashes_of(directory: Path) -> dict[Path, imagehash.ImageHash]:
    out: dict[Path, imagehash.ImageHash] = {}
    for p in iter_images(directory):
        fp = image_fingerprint(p)
        if fp is not None:
            out[p] = fp[0]
    return out


def report_cross_class(all_hashes: dict[str, dict[Path, imagehash.ImageHash]],
                       threshold: int) -> None:
    """클래스 간 중복 = 라벨 충돌. 자동 삭제하지 않고 사람에게 보고만 한다."""
    print("\n[data] 클래스 간 중복 검사 (라벨 충돌 후보)")
    items = [(sp, p, h) for sp, d in all_hashes.items() for p, h in d.items()]
    hits = 0
    for i in range(len(items)):
        sp_i, p_i, h_i = items[i]
        for j in range(i + 1, len(items)):
            sp_j, p_j, h_j = items[j]
            if sp_i != sp_j and (h_i - h_j) <= threshold:
                print(f"  ⚠ {sp_i}/{p_i.name}  ↔  {sp_j}/{p_j.name}")
                hits += 1
    if hits == 0:
        print("  충돌 없음")
    else:
        print(f"  {hits}건 — 둘 중 하나는 라벨이 틀렸다. 직접 보고 지울 것")


def record_deleted(targets: list[str]) -> None:
    """검수로 clean에서 지운 파일을 거부 목록에 확정한다.

    직전 dedup이 남긴 `clean_manifest.csv`(넣었던 목록)와 현재 clean 폴더를 비교한다.
    차이 = 사람이 지운 것. 이걸 기록해두면 다음 dedup에서 되살아나지 않는다.
    **검수를 끝냈으면 반드시 이 명령을 한 번 돌릴 것.**
    """
    manifest = load_clean_manifest()
    if not manifest:
        raise SystemExit("[warn] data/clean_manifest.csv 가 없다. dedup을 먼저 돌려야 한다")

    gone: list[tuple[str, str]] = []
    for name in targets:
        present = {p.name for p in iter_images(config.CLEAN_DIR / name)}
        gone += [(sp, fn) for (sp, fn) in manifest if sp == name and fn not in present]

    added = append_rejected(gone)
    print(f"[OK] 검수로 지운 파일 {len(gone)}건 확인 → 거부 목록에 {added}건 추가")
    if gone:
        by_species: dict[str, int] = defaultdict(int)
        for sp, _ in gone:
            by_species[sp] += 1
        for sp, n in sorted(by_species.items(), key=lambda kv: -kv[1]):
            print(f"  {sp}: {n}건")
    print("[next] python -m scripts.split")


def main() -> None:
    ap = argparse.ArgumentParser(description="phash 중복 제거: data/raw → data/clean")
    ap.add_argument("--species", nargs="*", default=None, help="대상 클래스(기본 전체)")
    ap.add_argument("--record-deleted", action="store_true",
                    help="사람 검수로 clean에서 지운 파일을 거부 목록에 확정(검수 후 필수)")
    ap.add_argument("--threshold", type=int, default=8,
                    help="phash 해밍거리 임계값. 낮으면 보수적(기본 8, 0~64)")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="clean을 비우지 않고 추가만 한다(검수로 지운 파일을 되살리지 않음)")
    ap.add_argument("--reject-conflicts", action="store_true",
                    help="클래스 간 라벨 충돌(같은 사진이 여러 클래스에 존재)을 전 사본 제거")
    ap.add_argument("--report-cross", action="store_true",
                    help="phash 근접 비교까지 한다 — 느리고 오탐이 많다(단순 구도끼리 붙는다)")
    ap.add_argument("--dry-run", action="store_true", help="집계만 하고 clean을 건드리지 않는다")
    args = ap.parse_args()

    targets = resolve_species(args.species)

    if args.record_deleted:
        record_deleted(targets)
        return

    rejected = load_rejected()
    print(f"[cfg] 대상 {len(targets)}클래스 | phash 임계 {args.threshold} | "
          f"거부목록 {len(rejected)}건 | "
          f"{'DRY-RUN' if args.dry_run else 'clean 재구성' if not args.no_rebuild else 'clean 추가만'}\n")

    totals: defaultdict[str, int] = defaultdict(int)
    group_rows: list[tuple[str, int, str]] = []
    manifest_rows: list[tuple[str, str]] = []
    cross_hashes: dict[str, dict[Path, imagehash.ImageHash]] = {}

    for name in targets:
        st = dedup_species(name, threshold=args.threshold, dry_run=args.dry_run,
                           rebuild=not args.no_rebuild, rejected=rejected)
        manifest_rows += [(name, fn) for fn in st["kept_names"]]
        for k in ("raw", "broken", "exact", "near", "kept"):
            totals[k] += st[k]
        for gi, paths in enumerate(st["groups"]):
            for p in paths:
                group_rows.append((name, gi, p.name))
        if args.report_cross:
            cross_hashes[name] = hashes_of(config.CLEAN_DIR / name)

    print("-" * 72)
    print(f"{'합계':<8} raw {totals['raw']:>5} → clean {totals['kept']:>5}  "
          f"(손상 {totals['broken']}, 완전중복 {totals['exact']}, 유사 {totals['near']})")

    # 클래스 간 라벨 충돌 — 같은 사진이 두 클래스에 있으면 최소 하나는 오라벨이다
    conflicts = find_label_conflicts(manifest_rows)
    if conflicts:
        n_files = sum(len(v) for v in conflicts.values())
        print(f"\n[warn] 라벨 충돌 {len(conflicts)}건 ({n_files}장) — 같은 사진이 여러 클래스에 있다")
        for key, items in sorted(conflicts.items()):
            where = ", ".join(f"{sp}/{fn}" for sp, fn in sorted(items))
            print(f"  {where}")
        if args.reject_conflicts and not args.dry_run:
            pairs = [(sp, fn) for v in conflicts.values() for sp, fn in v]
            for sp, fn in pairs:
                (config.CLEAN_DIR / sp / fn).unlink(missing_ok=True)
            added = append_rejected(pairs)
            print(f"  → 전 사본 {len(pairs)}장 제거 (거부 목록 +{added}). "
                  f"어느 쪽이 맞는지 알 수 없으므로 모두 버린다")
            totals["kept"] -= len(pairs)
            manifest_rows = [r for r in manifest_rows if r not in set(pairs)]
        else:
            print("  → 그대로 두려면 무시. 자동 제거는 --reject-conflicts")

    if not args.dry_run:
        # split.py가 유사 그룹을 같은 split에 몰아넣을 때 쓴다
        GROUPS_CSV.parent.mkdir(parents=True, exist_ok=True)
        with GROUPS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["species", "group", "filename"])
            w.writerows(group_rows)
        # 검수 후 --record-deleted 가 '무엇이 지워졌는지' 역산하는 기준
        write_clean_manifest(manifest_rows)
        print(f"[OK] 유사 그룹 기록: {GROUPS_CSV}")

    if args.report_cross:
        report_cross_class(cross_hashes, args.threshold)

    print("\n[next] python -m scripts.prefilter   (비물고기 후보 걸러내기)")


if __name__ == "__main__":
    main()
