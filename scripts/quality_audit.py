"""사진 품질 점수와 모델 오답률의 상관을 검증한다 (Phase 4 후속).

**목적은 필터를 거는 게 아니라, 필터를 걸어도 되는지 확인하는 것이다.**

Phase 4 진단에서 오답의 절반이 "사람도 종을 못 맞추는 사진"이었다
(물고기가 너무 작음 / 물속이라 흐림 / 여러 마리). 이걸 걸러내면 성능이 오를
것으로 보이지만, **눈짐작으로 임계값을 정하면 멀쩡한 사진까지 대량으로 날아간다.**

그래서 순서를 지킨다:

1. test셋 전 이미지에 `fish_prob`(ImageNet 어류 클래스 확률 합)를 매긴다
2. `reports/predictions.csv`(evaluate.py 산출물)와 대조한다
3. **점수가 낮은 구간이 정말 더 많이 틀리는지** 확인한다
4. 그때서야 임계값을 고른다

3번에서 상관이 없으면 이 가설은 기각이고, 필터를 걸면 안 된다.

`prefilter.py`가 iNat 사진을 제외했던 이유는 "수중 사진에서 fish_prob가 낮게 나와
오탐이 많다"였다. 당시 목적은 '라벨이 맞나'였으니 맞는 말이었지만, 지금 목적은
'사람이 이 사진으로 종을 구분할 수 있나'라 **같은 현상이 원하는 신호가 된다.**

사용 예:
    python -m src.evaluate                    # 먼저 predictions.csv 를 만들고
    python -m scripts.quality_audit           # 그 다음 이걸 돌린다
    python -m scripts.quality_audit --split val --device cuda
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from src import config
from scripts.prefilter import FISH_IDX, load_model

PREDICTIONS_CSV = config.REPORTS_DIR / "predictions.csv"
OUT_CSV = config.REPORTS_DIR / "quality_audit.csv"


@torch.no_grad()
def score_images(paths: list[Path], model, tf, device, batch_size: int) -> dict[str, float]:
    """경로 → fish_prob. 읽기 실패한 파일은 0.0으로 둔다(어차피 학습에도 못 쓴다)."""
    scores: dict[str, float] = {}
    fish = list(FISH_IDX)
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        tensors, ok = [], []
        for p in chunk:
            try:
                with Image.open(p) as img:
                    tensors.append(tf(img.convert("RGB")))
                ok.append(p)
            except Exception:
                scores[str(p)] = 0.0
        if not tensors:
            continue
        x = torch.stack(tensors).to(device)
        probs = model(x).softmax(dim=-1)
        for p, row in zip(ok, probs):
            scores[str(p)] = float(row[fish].sum())
        done = min(i + batch_size, len(paths))
        print(f"\r  채점 {done:,}/{len(paths):,}", end="", flush=True)
    print()
    return scores


def load_predictions(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"[FAIL] {path} 가 없다.\n"
            "  → `python -m src.evaluate` 를 먼저 돌려 predictions.csv 를 만들 것."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def report(rows: list[dict], n_buckets: int = 10) -> None:
    """fish_prob 오름차순 10분위별 정확도. 아래 구간이 낮아야 가설이 맞다."""
    rows = sorted(rows, key=lambda r: r["fish_prob"])
    n = len(rows)
    size = max(n // n_buckets, 1)

    print("\n" + "=" * 74)
    print(f"  품질 점수(fish_prob) 10분위별 정확도  — {n:,}장")
    print("=" * 74)
    print(f"  {'구간':<6} {'점수 범위':<20} {'장수':>6} {'Top-1':>8} {'Top-3':>8}")
    print("  " + "-" * 70)

    for b in range(n_buckets):
        lo = b * size
        hi = n if b == n_buckets - 1 else (b + 1) * size
        chunk = rows[lo:hi]
        if not chunk:
            continue
        t1 = sum(r["correct_top1"] for r in chunk) / len(chunk)
        t3 = sum(r["correct_top3"] for r in chunk) / len(chunk)
        rng = f"{chunk[0]['fish_prob']:.3f} ~ {chunk[-1]['fish_prob']:.3f}"
        print(f"  {b + 1:<6} {rng:<20} {len(chunk):>6,} {t1 * 100:>7.1f}% {t3 * 100:>7.1f}%")

    # 출처별 — iNat 사진이 정말 더 나쁜지 확인한다(진단의 근거였다)
    print("\n  [출처별]")
    for src in ("inat", "web", "other"):
        chunk = [r for r in rows if r["source"] == src]
        if not chunk:
            continue
        t3 = sum(r["correct_top3"] for r in chunk) / len(chunk)
        avg = sum(r["fish_prob"] for r in chunk) / len(chunk)
        print(f"    {src:<6} {len(chunk):>6,}장  Top-3 {t3 * 100:5.1f}%  "
              f"평균 fish_prob {avg:.3f}")

    # 임계값 후보 — 버리는 양 대비 남는 것의 정확도
    print("\n  [임계값 후보]  '버림'은 학습셋에서 제외될 비율")
    print(f"  {'임계값':>8} {'버림':>8} {'남은 것 Top-3':>14} {'버린 것 Top-3':>14}")
    print("  " + "-" * 70)
    for th in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        keep = [r for r in rows if r["fish_prob"] >= th]
        drop = [r for r in rows if r["fish_prob"] < th]
        if not keep:
            continue
        kt3 = sum(r["correct_top3"] for r in keep) / len(keep)
        dt3 = (sum(r["correct_top3"] for r in drop) / len(drop)) if drop else float("nan")
        print(f"  {th:>8.2f} {len(drop) / len(rows) * 100:>7.1f}% "
              f"{kt3 * 100:>13.1f}% {dt3 * 100:>13.1f}%")

    print("\n  읽는 법:")
    print("    - 아래 구간(1~2분위)의 Top-3가 위 구간보다 뚜렷이 낮아야 가설이 맞다.")
    print("    - '버린 것 Top-3'가 '남은 것 Top-3'보다 한참 낮아야 그 임계값이 쓸 만하다.")
    print("    - 차이가 없으면 fish_prob는 쓸 신호가 아니다 → **필터를 걸지 말 것**.")


def main() -> None:
    ap = argparse.ArgumentParser(description="사진 품질 점수 vs 오답률 상관 검증")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--predictions", type=Path, default=PREDICTIONS_CSV)
    ap.add_argument("--model", default="tf_efficientnet_b0.ns_jft_in1k")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    preds = load_predictions(args.predictions)
    print(f"[data] predictions.csv {len(preds):,}행")

    device = torch.device(args.device)
    print(f"[env] device={device} | 채점 모델 {args.model}")
    model, tf, _ = load_model(args.model, device)

    paths = [Path(r["path"]) for r in preds]
    scores = score_images(paths, model, tf, device, args.batch_size)

    rows = []
    for r in preds:
        rows.append({
            "path": r["path"],
            "filename": r["filename"],
            "source": r["source"],
            "true": r["true"],
            "pred": r["pred"],
            "fish_prob": round(scores.get(r["path"], 0.0), 4),
            "correct_top1": int(r["correct_top1"]),
            "correct_top3": int(r["correct_top3"]),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    report(rows)
    print(f"\n[out] {args.out}")
    print("      fish_prob 오름차순으로 정렬해 실제 사진을 눈으로 확인할 것.")


if __name__ == "__main__":
    main()
