"""서빙 전처리가 학습 전처리와 같은 결과를 내는지 실이미지로 대조한다.

**이 검사가 없으면 전처리 불일치는 발견되지 않는다.** 서버는 에러 없이 잘 돌고,
정확도만 조용히 몇 %씩 낮아진다 — 배포 후에는 "원래 이 정도인가 보다" 하고 넘어가게 된다.

비교 대상:
  기준(학습) `src.dataset.build_transforms("val", cfg)` — albumentations + cv2
  대상(서빙) `server.inference.preprocess_image` — cv2 (torch/albumentations 없이)

두 경로가 **비트 단위로 같아야** 통과다(허용 오차를 두지 않는다 — 같은 cv2 연산을 같은
순서로 태우므로 다를 이유가 없고, 오차를 허용하기 시작하면 진짜 불일치를 못 잡는다).
크기별로 나눠 보고한다: 짧은 변이 목표보다 큰 사진은 **축소**를, 작은 사진은 **확대**를
겪으므로 깨지는 방식이 다르다.

사용 예:
    python -m scripts.check_preprocess                  # val 200장
    python -m scripts.check_preprocess --split test -n 500
    python -m scripts.check_preprocess --img-size 384   # labels.json 대신 강제 지정
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from src import config
from src.config import TrainConfig
from server.inference import Preprocess, decode_image, preprocess_image

TOLERANCE = 0.0  # 비트 단위 일치를 요구한다


def collect(split: str, n: int, seed: int) -> list[Path]:
    root = config.SPLIT_DIRS[split]
    if not root.is_dir():
        raise SystemExit(f"[FAIL] {root} 가 없다 — Phase 2(분할)를 먼저 끝낼 것")
    files: list[Path] = []
    for name in config.CLASSES:  # 종별로 고르게 뽑아야 크기 분포가 한쪽에 쏠리지 않는다
        d = root / name
        if d.is_dir():
            files += [p for p in sorted(d.iterdir())
                      if p.suffix.lower() in config.VALID_IMAGE_EXTS]
    random.Random(seed).shuffle(files)
    return files[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="학습 전처리 vs 서빙 전처리 대조")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("-n", "--num", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=None,
                    help="지정하면 labels.json 대신 이 크기로 비교")
    ap.add_argument("--labels", type=Path, default=config.LABELS_JSON)
    args = ap.parse_args()

    import json

    import albumentations as A

    from src.dataset import build_transforms, imread_unicode

    if args.img_size:
        pp = Preprocess(img_size=args.img_size,
                        resize_shorter_to=int(args.img_size * 1.14),
                        mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    else:
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        pp = Preprocess.from_labels(labels, args.labels)

    cfg = TrainConfig(img_size=pp.img_size, mean=tuple(pp.mean), std=tuple(pp.std))
    train_tf: A.Compose = build_transforms("val", cfg)

    files = collect(args.split, args.num, args.seed)
    print(f"[cfg] {args.split}셋 {len(files)}장 | img_size={pp.img_size} "
          f"| 짧은변 → {pp.resize_shorter_to}px")

    upscaled, downscaled, failed = [], [], []
    for path in files:
        ref_img = imread_unicode(path)
        if ref_img is None:
            failed.append((path, "학습 로더가 못 읽음"))
            continue
        try:
            got_img = decode_image(path.read_bytes())
        except Exception as exc:  # noqa: BLE001
            failed.append((path, f"서버 디코더가 못 읽음: {exc}"))
            continue

        # 학습 경로는 EXIF 회전을 반영하지 않는다(cv2.imdecode). 서버는 반영한다.
        # 회전이 붙은 사진은 애초에 shape부터 달라지므로 여기서 갈라 보고한다.
        if ref_img.shape != got_img.shape:
            failed.append((path, f"shape 불일치 {ref_img.shape} vs {got_img.shape} "
                                 "(EXIF 회전 — 서버 쪽이 옳다)"))
            continue

        # build_transforms는 ToTensorV2까지 포함하므로 (3,H,W) 텐서가 나온다
        ref = train_tf(image=ref_img)["image"].numpy()
        got = preprocess_image(got_img, pp)[0]                     # (1,3,H,W) → (3,H,W)
        diff = float(np.abs(ref - got).max())

        shrunk = min(ref_img.shape[:2]) > pp.resize_shorter_to
        (downscaled if shrunk else upscaled).append((path, diff))

    def report(rows: list[tuple[Path, float]], title: str) -> float:
        if not rows:
            print(f"  {title}: 해당 없음")
            return 0.0
        d = np.array([r[1] for r in rows])
        print(f"  {title}: {len(rows)}장 | max {d.max():.2e} | mean {d.mean():.2e} | "
              f"완전일치 {(d == 0).sum()}/{len(rows)}")
        return float(d.max())

    print("\n[결과]")
    worst = max(report(upscaled, "확대 구간(짧은 변 ≤ 목표)"),
                report(downscaled, "축소 구간(짧은 변 > 목표 — 폰 사진이 여기 해당)"))

    if failed:
        print(f"\n[warn] 비교 불가 {len(failed)}장:")
        for path, why in failed[:10]:
            print(f"  - {path.name}: {why}")

    ok = worst <= TOLERANCE
    print(f"\n[{'OK' if ok else 'FAIL'}] 서빙 전처리 = 학습 전처리 "
          f"(max diff {worst:.2e}, 허용 {TOLERANCE:.0e})")
    if not ok:
        raise SystemExit(
            "[FAIL] 전처리가 어긋난다 — 이대로 배포하면 정확도가 조용히 떨어진다.\n"
            "  → server/inference.preprocess_image 와 dataset.build_transforms('val') 대조할 것"
        )


if __name__ == "__main__":
    main()
