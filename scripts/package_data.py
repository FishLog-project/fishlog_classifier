"""data/splits → data/splits.zip (Colab 업로드용 단일 파일 패키징).

Colab에 소파일 7천 개를 그대로 올리거나 Drive에서 직접 읽으면 학습보다 I/O가
오래 걸린다. zip 1개로 올려 Colab **로컬 디스크에** 푸는 것이 전제다.
([setup.md](../docs/setup.md) "Colab에서 학습하기")

두 가지를 신경 쓴다:

1. **압축 방식은 STORED(무압축)**. 내용물이 전부 JPEG라 DEFLATE를 걸어도 크기는
   1~2%밖에 안 줄고 패키징 시간만 몇 배로 늘어난다.
2. **한글 폴더명**. `zipfile` 은 비ASCII 이름에 UTF-8 플래그(0x800)를 자동으로
   세우므로 Colab의 `unzip` 에서 그대로 복원된다. 마지막에 검증한다.

사용 예:
    python -m scripts.package_data              # → data/splits.zip
    python -m scripts.package_data --out /path/to/other.zip
    python -m scripts.package_data --verify-only    # 기존 zip만 검사
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from src import config

DEFAULT_OUT = config.DATA_DIR / "splits.zip"


def build(src: Path, out: Path) -> int:
    """src 폴더를 통째로 zip에 담는다. 아카이브 경로는 `splits/<split>/<종>/…`."""
    if not src.is_dir():
        raise FileNotFoundError(
            f"분할 폴더가 없다: {src}\n  → `python -m scripts.split` 를 먼저 돌릴 것."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file() and p.suffix.lower() in config.VALID_IMAGE_EXTS:
                z.write(p, arcname=p.relative_to(src.parent).as_posix())
                n += 1
    return n


def verify(out: Path) -> None:
    """CRC·UTF-8 플래그·split별 장수를 확인한다. 업로드 전에 반드시 통과시킬 것."""
    with zipfile.ZipFile(out) as z:
        infos = z.infolist()
        broken = z.testzip()
        if broken is not None:
            raise SystemExit(f"[FAIL] 손상된 항목: {broken}")

        non_ascii = [i for i in infos if not i.filename.isascii()]
        no_utf8 = [i for i in non_ascii if not i.flag_bits & 0x800]
        if no_utf8:
            raise SystemExit(
                f"[FAIL] UTF-8 플래그 없는 한글 경로 {len(no_utf8)}건 "
                f"(예: {no_utf8[0].filename}) — Colab에서 폴더명이 깨진다."
            )

        per_split: dict[str, int] = {}
        for i in infos:
            parts = i.filename.split("/")
            if len(parts) >= 2:
                per_split[parts[1]] = per_split.get(parts[1], 0) + 1

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"[OK] {out}  ({size_mb:,.1f} MB, {len(infos):,}장)")
    for split in ("train", "val", "test"):
        print(f"     {split:<6} {per_split.get(split, 0):>6,}")
    print(f"     한글 경로 {len(non_ascii):,}건 전부 UTF-8 플래그 있음")


def main() -> None:
    ap = argparse.ArgumentParser(description="Colab 업로드용 splits.zip 패키징")
    ap.add_argument("--src", type=Path, default=config.SPLITS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify-only", action="store_true",
                    help="새로 만들지 않고 기존 zip만 검사")
    args = ap.parse_args()

    if not args.verify_only:
        if args.out.exists():
            args.out.unlink()  # 이어쓰기 방지 (구 버전 잔재가 섞이면 조용히 틀어진다)
        n = build(args.src, args.out)
        print(f"[..] {n:,}장 담음 → 검증")
    verify(args.out)
    print("\n다음: 이 파일을 Drive `MyDrive/fishlog/splits.zip` 로 업로드 → docs/setup.md")


if __name__ == "__main__":
    main()
