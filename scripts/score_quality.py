"""data/splits 전 이미지에 품질 점수(fish_prob)를 매겨 캐시한다.

`quality_audit.py` 가 "품질 점수가 낮은 사진이 정말 더 많이 틀린다"를 test셋에서
확인했다(2026-08-18: 하위 20% Top-3 77% vs 상위 20% 93~96%). 이제 그 점수를
학습에 실제로 쓰기 위해 **전 split을 한 번 채점해 CSV로 남긴다.**

점수를 매기기만 하고 **파일은 건드리지 않는다.** 실제 제외는 학습 시점에
`--min-fish-prob` 로 정하므로, 임계값을 바꿔가며 실험해도 재분할이 필요 없다.
(분할을 다시 하면 train/test 경계가 바뀌어 이전 실험과 비교가 불가능해진다.)

사용 예:
    python -m scripts.score_quality                  # → data/quality_scores.csv
    python -m scripts.score_quality --device cuda --batch-size 64
    python -m scripts.score_quality --summary        # 기존 CSV 요약만 출력
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from src import config
from scripts.prefilter import FISH_IDX, load_model

SCORES_CSV = config.DATA_DIR / "quality_scores.csv"


def collect_paths() -> list[tuple[str, str, Path]]:
    """(split, 종명, 경로) 목록. config.CLASSES 순서를 따른다."""
    out = []
    for split, root in config.SPLIT_DIRS.items():
        if not root.is_dir():
            continue
        for name in config.CLASSES:
            d = root / name
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in config.VALID_IMAGE_EXTS:
                    out.append((split, name, p))
    return out


@torch.no_grad()
def score_all(items: list[tuple[str, str, Path]], model, tf, device,
              batch_size: int) -> list[dict]:
    fish = list(FISH_IDX)
    rows: list[dict] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        tensors, ok = [], []
        for split, name, p in chunk:
            try:
                with Image.open(p) as img:
                    tensors.append(tf(img.convert("RGB")))
                ok.append((split, name, p))
            except Exception:
                # 열리지 않는 파일은 0점 — 어차피 학습에서도 건너뛴다
                rows.append({"split": split, "species": name,
                             "filename": p.name, "fish_prob": 0.0})
        if tensors:
            probs = model(torch.stack(tensors).to(device)).softmax(dim=-1)
            for (split, name, p), row in zip(ok, probs):
                rows.append({"split": split, "species": name, "filename": p.name,
                             "fish_prob": round(float(row[fish].sum()), 4)})
        print(f"\r  채점 {min(i + batch_size, len(items)):,}/{len(items):,}",
              end="", flush=True)
    print()
    return rows


def summarize(rows: list[dict]) -> None:
    """임계값별로 split마다 몇 장이 빠지는지. 학습셋이 얼마나 얇아지는지가 관건이다."""
    by_split: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(float(r["fish_prob"]))

    print("\n" + "=" * 66)
    print("  임계값별 제외 장수")
    print("=" * 66)
    header = f"  {'임계값':>8}" + "".join(f"{s:>16}" for s in ("train", "val", "test"))
    print(header)
    print("  " + "-" * 62)
    for th in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        line = f"  {th:>8.2f}"
        for s in ("train", "val", "test"):
            v = by_split.get(s, [])
            if not v:
                line += f"{'-':>16}"
                continue
            drop = sum(1 for x in v if x < th)
            line += f"{f'-{drop} ({drop / len(v) * 100:.1f}%)':>16}"
        print(line)

    # 종별로 몰리면 그 종만 학습이 무너진다 — 반드시 확인할 것
    print("\n  [임계값 0.30 기준, train에서 가장 많이 빠지는 종]")
    per: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            per[r["species"]].append(float(r["fish_prob"]))
    ranked = sorted(per.items(),
                    key=lambda kv: -sum(1 for x in kv[1] if x < 0.30) / max(len(kv[1]), 1))
    for name, v in ranked[:8]:
        drop = sum(1 for x in v if x < 0.30)
        print(f"    {name:<6} {len(v):>4}장 중 {drop:>3}장 제외 "
              f"({drop / len(v) * 100:4.1f}%) → 남는 건 {len(v) - drop}장")


def main() -> None:
    ap = argparse.ArgumentParser(description="전 split 품질 점수 채점")
    ap.add_argument("--model", default="tf_efficientnet_b0.ns_jft_in1k")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=SCORES_CSV)
    ap.add_argument("--summary", action="store_true", help="채점 없이 기존 CSV 요약만")
    args = ap.parse_args()

    if args.summary:
        if not args.out.exists():
            raise SystemExit(f"[FAIL] {args.out} 가 없다 → --summary 없이 먼저 채점할 것.")
        with args.out.open("r", encoding="utf-8-sig", newline="") as f:
            summarize(list(csv.DictReader(f)))
        return

    items = collect_paths()
    if not items:
        raise SystemExit("[FAIL] data/splits 에 이미지가 없다.")
    print(f"[data] {len(items):,}장 채점 대상")

    device = torch.device(args.device)
    print(f"[env] device={device} | {args.model}")
    model, tf, _ = load_model(args.model, device)

    rows = score_all(items, model, tf, device, args.batch_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "species", "filename", "fish_prob"])
        w.writeheader()
        w.writerows(rows)

    summarize(rows)
    print(f"\n[out] {args.out}")
    print("[next] python -m src.train --min-fish-prob 0.30   "
          "(학습셋에만 적용된다 — val/test는 건드리지 않는다)")


if __name__ == "__main__":
    main()
