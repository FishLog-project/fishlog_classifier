"""`uncertain` 임계값을 **실제 서빙 경로**(ONNX + server 전처리)로 튜닝한다.

`src/evaluate.py --split val` 도 임계값 곡선을 그리지만 그건 torch 경로이고, 서버의
판정 규칙(둘 중 하나라도 걸리면 uncertain)을 반영하지 않는다:

    1) top1 확률 < 임계값        — 모델이 자신 없다
    2) top1이 `기타`             — 물고기가 아니거나 24종 밖이다

두 규칙이 함께 걸리면 거부율은 임계값만 보고 예측한 값보다 항상 높다. 배포할 값은
배포할 경로에서 재야 한다.

**val에서만 돌린다.** test로 임계값을 고르면 최종 수치가 더 이상 "처음 보는 데이터"가
아니게 된다(evaluation.md "하지 말 것").

기준(evaluation.md): **어종 사진 거부율 10% 이하를 유지하면서 통과분 Top-3가 최대**인 지점.

사용 예:
    python -m scripts.tune_threshold                    # val 전체
    python -m scripts.tune_threshold --limit 300        # 빠르게 훑기
    python -m scripts.tune_threshold --cached           # 저장된 확률로 재계산만
    python -m scripts.tune_threshold --write            # 고른 값을 labels.json에 반영
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src import config

PROBS_NPZ = config.REPORTS_DIR / "val_probs_server.npz"
OUT_JSON = config.REPORTS_DIR / "threshold_tuning.json"


def collect_files(split: str, limit: int | None) -> list[tuple[Path, int]]:
    root = config.SPLIT_DIRS[split]
    if not root.is_dir():
        raise SystemExit(f"[FAIL] {root} 가 없다")
    items: list[tuple[Path, int]] = []
    for idx, name in enumerate(config.CLASSES):
        d = root / name
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in config.VALID_IMAGE_EXTS:
                items.append((p, idx))
    if limit:
        # 종별로 고르게 자른다 — 앞에서 자르면 클래스가 통째로 빠진다
        rng = np.random.default_rng(0)
        sel = rng.permutation(len(items))[:limit]
        items = [items[i] for i in sorted(sel)]
    return items


def run_inference(items: list[tuple[Path, int]], tta: bool) -> tuple[np.ndarray, np.ndarray]:
    from server.inference import Predictor, decode_image

    p = Predictor(tta=tta)
    print(f"[cfg] {p.model_path.name} | {p.pp.img_size}px | TTA {'on' if p.tta else 'off'} "
          f"| 워밍업 {p.warmup_ms:.0f}ms")

    probs, y, t0 = [], [], time.perf_counter()
    for i, (path, label) in enumerate(items, 1):
        try:
            probs.append(p.infer_probs(decode_image(path.read_bytes())))
            y.append(label)
        except Exception as exc:  # noqa: BLE001 — 서버가 4xx로 돌려보낼 사진은 튜닝 대상이 아니다
            print(f"[warn] 건너뜀 {path.name}: {exc}")
        if i % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  {i}/{len(items)} | {el / i * 1000:.0f}ms/장 | 남은 시간 "
                  f"{el / i * (len(items) - i):.0f}s")
    el = time.perf_counter() - t0
    print(f"[OK] {len(probs)}장 추론 | 장당 평균 {el / max(1, len(probs)) * 1000:.0f}ms")
    return np.asarray(probs, np.float32), np.asarray(y, np.int64)


def sweep(probs: np.ndarray, y: np.ndarray, other_idx: int,
          ths: np.ndarray) -> list[dict]:
    """임계값별로 서버 판정 규칙을 그대로 적용해 본다."""
    top1 = probs.argmax(1)
    conf = probs.max(1)
    is_other_label = y == other_idx
    fish = ~is_other_label

    # 서버는 후보에서 `기타`를 뺀 뒤 Top-3를 준다 → 정확도도 그 기준으로 재야 한다
    fish_probs = probs.copy()
    fish_probs[:, other_idx] = -1
    top3_fish = np.argsort(-fish_probs, 1)[:, :3]
    hit3 = (top3_fish == y[:, None]).any(1)
    hit1 = top3_fish[:, 0] == y

    rows = []
    for t in ths:
        uncertain = (conf < t) | (top1 == other_idx)
        kept_fish = fish & ~uncertain
        rows.append({
            "threshold": round(float(t), 2),
            # UX 비용: 진짜 물고기인데 "다시 찍어주세요"를 본 비율
            "fish_reject_rate": round(float(uncertain[fish].mean()), 4),
            # 통과한 어종 사진의 정확도 — 사용자가 실제로 보는 후보의 품질
            "top3_on_kept": round(float(hit3[kept_fish].mean()) if kept_fish.any() else 0.0, 4),
            "top1_on_kept": round(float(hit1[kept_fish].mean()) if kept_fish.any() else 0.0, 4),
            # 이득: 24종 밖/비물고기를 제대로 걸러낸 비율
            "other_caught": round(float(uncertain[is_other_label].mean()), 4)
            if is_other_label.any() else None,
            "kept_n": int(kept_fish.sum()),
        })
    return rows


def choose(rows: list[dict], cap: float) -> dict:
    """거부율 상한 안에서 통과분 Top-3가 최대인 지점. 동률이면 거부율이 낮은 쪽."""
    ok = [r for r in rows if r["fish_reject_rate"] <= cap]
    if not ok:
        return min(rows, key=lambda r: r["fish_reject_rate"])
    return max(ok, key=lambda r: (r["top3_on_kept"], -r["fish_reject_rate"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="uncertain 임계값 튜닝 (서빙 경로 기준)")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--cap", type=float, default=0.10, help="어종 사진 거부율 상한")
    ap.add_argument("--cached", action="store_true", help="저장된 확률로 스윕만 다시")
    ap.add_argument("--write", action="store_true",
                    help="고른 값을 server/labels.json 의 confidence_threshold 에 반영")
    args = ap.parse_args()

    if args.split == "test":
        print("[warn] test로 임계값을 고르면 최종 수치의 의미가 사라진다 "
              "(evaluation.md '하지 말 것'). 확인용으로만 볼 것.")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.cached:
        if not PROBS_NPZ.exists():
            raise SystemExit(f"[FAIL] {PROBS_NPZ} 가 없다 → --cached 없이 먼저 돌릴 것")
        d = np.load(PROBS_NPZ)
        probs, y = d["probs"], d["y"]
        print(f"[cfg] 저장된 확률 사용: {PROBS_NPZ.name} ({len(y)}장)")
    else:
        items = collect_files(args.split, args.limit)
        print(f"[cfg] {args.split}셋 {len(items)}장")
        probs, y = run_inference(items, args.tta)
        np.savez_compressed(PROBS_NPZ, probs=probs, y=y)
        print(f"[OK] 확률 저장: {PROBS_NPZ}")

    labels = json.loads(config.LABELS_JSON.read_text(encoding="utf-8"))
    other_idx = int(labels["other_index"])
    rows = sweep(probs, y, other_idx, np.arange(0.20, 0.85, 0.05))
    best = choose(rows, args.cap)

    print(f"\n{'임계값':>6} {'어종거부율':>9} {'통과Top-3':>9} {'통과Top-1':>9} "
          f"{'기타검출':>8} {'통과장수':>7}")
    for r in rows:
        mark = " ←" if r["threshold"] == best["threshold"] else ""
        oc = f"{r['other_caught']:.3f}" if r["other_caught"] is not None else "   -  "
        print(f"{r['threshold']:>6.2f} {r['fish_reject_rate']:>9.3f} "
              f"{r['top3_on_kept']:>9.4f} {r['top1_on_kept']:>9.4f} {oc:>8} "
              f"{r['kept_n']:>7}{mark}")

    current = float(labels.get("confidence_threshold", 0.45))
    caught = (f" | `기타` {best['other_caught']:.1%} 검출"
              if best["other_caught"] is not None else "")
    print(f"\n[선택] 임계값 {best['threshold']:.2f} — 어종 거부율 "
          f"{best['fish_reject_rate']:.1%} (상한 {args.cap:.0%}) | 통과 Top-3 "
          f"{best['top3_on_kept']:.4f}{caught}")
    print(f"[현재] labels.json = {current:.2f}")

    OUT_JSON.write_text(json.dumps({
        "split": args.split, "n": int(len(y)), "tta": bool(args.tta),
        "cap": args.cap, "rows": rows, "chosen": best, "current": current,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {OUT_JSON}")

    if args.write and best["threshold"] != current:
        labels["confidence_threshold"] = best["threshold"]
        config.LABELS_JSON.write_text(
            json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] labels.json confidence_threshold {current:.2f} → {best['threshold']:.2f}")
        print("[next] src/config.py 의 CONFIDENCE_THRESHOLD 도 같이 맞출 것 "
              "(labels.json 재생성 시 되돌아간다)")
    elif not args.write:
        print("[next] 값을 반영하려면 --write")


if __name__ == "__main__":
    main()
