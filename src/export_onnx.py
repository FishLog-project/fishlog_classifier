"""Phase 5 — 학습된 `.pt` 를 서빙용 `.onnx` 로 변환한다.

**이 단계의 위험은 성능이 아니라 조용한 불일치다.** 변환 자체는 거의 실패하지 않는데,
라벨 순서가 어긋나거나 전처리 파라미터가 달라지면 **에러 없이 엉뚱한 답**이 나온다.
그래서 변환보다 검증에 코드가 더 많다:

1. 체크포인트의 `classes` == `config.CLASSES` == `server/labels.json` 의 `classes`
2. 같은 입력에 대해 `.pt` 와 `.onnx` 출력의 max abs diff < 1e-4
3. 실제 이미지로도 한 번 더 — 랜덤 텐서만으로는 전처리 실수를 못 잡는다
4. 전처리 파라미터(img_size/mean/std)를 `labels.json` 에 박아 서버가 읽게 한다

**softmax는 넣지 않는다.** 모델은 logits까지만 내고 서버에서 softmax를 적용한다.
그래야 나중에 temperature scaling을 붙일 수 있다 (decisions.md C-2).

사용 예:
    python -m src.export_onnx                                   # models/best.pt → server/model.onnx
    python -m src.export_onnx --ckpt models/exp_384.pt
    python -m src.export_onnx --verify-only                     # 기존 onnx만 검증
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import config
from src.config import CLASSES, NUM_CLASSES
from src.evaluate import load_checkpoint
from src.train import get_device

INPUT_NAME = "input"
OUTPUT_NAME = "logits"
# torch 2.13의 dynamo 익스포터는 opset 18로 뽑는다. 13으로 낮추려 하면
# onnx version_converter가 실패하고 **에러를 찍은 뒤 18로 그냥 진행**한다
# (요청과 결과가 다른데 성공한 것처럼 보인다). onnxruntime>=1.19가 18을 지원하고
# serving.md 요구사항도 "12 이상"이므로 18을 그대로 쓰고, 아래에서 실제 값을 검증한다.
OPSET = 18
TOLERANCE = 1e-4


def check_labels(ckpt_classes: list[str] | None) -> None:
    """체크포인트 · config · labels.json 세 곳의 클래스 순서를 대조한다.

    하나라도 어긋나면 서버가 내놓는 어종명이 통째로 밀린다. 지표는 멀쩡해 보이므로
    배포 후에야 발견되는 종류의 사고다 → 여기서 막는다.
    """
    if ckpt_classes is not None and list(ckpt_classes) != CLASSES:
        raise SystemExit(
            "[FAIL] 체크포인트의 클래스 순서가 config.SPECIES 와 다르다.\n"
            f"  ckpt[0:3]={list(ckpt_classes)[:3]} / config[0:3]={CLASSES[:3]}\n"
            "  → config를 되돌리거나 재학습할 것 (CLAUDE.md 불변 규칙)"
        )

    if not config.LABELS_JSON.exists():
        print(f"[warn] {config.LABELS_JSON} 이 없다 → 새로 만든다")
        config.write_labels_json()

    labels = json.loads(config.LABELS_JSON.read_text(encoding="utf-8"))
    if labels.get("classes") != CLASSES:
        raise SystemExit(
            "[FAIL] server/labels.json 의 클래스 순서가 config 와 다르다.\n"
            "  → `python -m src.config --init` 으로 재생성할 것"
        )
    if labels.get("num_classes") != NUM_CLASSES:
        raise SystemExit(f"[FAIL] labels.json num_classes={labels.get('num_classes')} "
                         f"≠ {NUM_CLASSES}")
    print(f"[OK] 라벨 순서 일치 (체크포인트 · config · labels.json, {NUM_CLASSES}클래스)")


def export(model: torch.nn.Module, img_size: int, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, img_size, img_size)
    model.eval().cpu()

    torch.onnx.export(
        model, dummy, str(out),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        # 배치 가변 — 서버가 여러 장을 한 번에 처리할 여지를 남긴다.
        # 해상도는 고정이다(전처리가 항상 img_size로 맞추므로 가변일 이유가 없고,
        # 고정이어야 런타임이 최적화하기 좋다).
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
        # ⚠️ 기본값 True면 가중치가 `model.onnx.data` 로 따로 빠진다.
        #    Docker 이미지에 .onnx만 복사하면 **로드 시점에 터진다** — 파일 하나로 묶는다.
        external_data=False,
    )

    # 부산물 정리: 이전 실행이 남긴 외부 데이터 파일이 있으면 혼동을 부른다
    stale = out.with_suffix(out.suffix + ".data")
    if stale.exists():
        stale.unlink()
        print(f"[..] 이전 외부 데이터 파일 제거: {stale.name}")

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"[OK] export 완료: {out}  ({size_mb:.1f} MB)")
    if size_mb < 5:
        raise SystemExit(
            f"[FAIL] {size_mb:.1f} MB는 너무 작다 — 가중치가 안 담겼을 수 있다.\n"
            "  → external_data 설정을 확인할 것 (efficientnet_b0는 약 16MB)"
        )

    # 요청한 opset이 실제로 반영됐는지 확인한다(위 주석 참조: 조용히 달라질 수 있다)
    import onnx
    got = {(o.domain or "ai.onnx"): o.version for o in onnx.load(str(out)).opset_import}
    print(f"[OK] opset = {got}")
    if got.get("ai.onnx", 0) < 12:
        raise SystemExit(f"[FAIL] opset {got} 은 요구사항(12 이상) 미달")


def verify_numeric(model: torch.nn.Module, onnx_path: Path, img_size: int,
                   n: int = 4) -> float:
    """랜덤 입력으로 .pt 와 .onnx 출력을 대조한다."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(n, 3, img_size, img_size)

    with torch.no_grad():
        want = model.eval().cpu()(x).numpy()
    got = sess.run([OUTPUT_NAME], {INPUT_NAME: x.numpy()})[0]

    diff = float(np.abs(want - got).max())
    mark = "OK" if diff < TOLERANCE else "FAIL"
    print(f"[{mark}] 랜덤 입력 {n}장 max abs diff = {diff:.2e} (허용 {TOLERANCE:.0e})")
    if diff >= TOLERANCE:
        raise SystemExit("[FAIL] 수치 불일치 — opset을 올리거나 연산 호환성을 확인할 것")
    return diff


def verify_real_images(model: torch.nn.Module, onnx_path: Path, cfg, n: int = 8) -> None:
    """실제 이미지로 **Top-1 예측이 같은지**까지 본다.

    랜덤 텐서는 전처리를 거치지 않으므로 정규화 실수 같은 것을 못 잡는다.
    val셋 앞쪽 몇 장을 실제 파이프라인에 태워 확인한다.
    """
    import onnxruntime as ort

    from src.dataset import FishDataset, build_transforms

    root = config.SPLIT_DIRS["val"]
    if not root.is_dir():
        print("[skip] val셋이 없어 실이미지 검증을 건너뛴다")
        return

    ds = FishDataset(root, transform=build_transforms("val", cfg))
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    x = torch.stack([ds[i][0] for i in idx])

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        want = model.eval().cpu()(x).numpy()
    got = sess.run([OUTPUT_NAME], {INPUT_NAME: x.numpy()})[0]

    diff = float(np.abs(want - got).max())
    same = int((want.argmax(1) == got.argmax(1)).sum())
    print(f"[{'OK' if diff < TOLERANCE and same == len(idx) else 'FAIL'}] "
          f"실이미지 {len(idx)}장 max abs diff = {diff:.2e} | Top-1 일치 {same}/{len(idx)}")
    if diff >= TOLERANCE or same != len(idx):
        raise SystemExit("[FAIL] 실이미지에서 불일치 — 전처리/변환을 다시 볼 것")

    print(f"       예측 예시: {[CLASSES[i] for i in got.argmax(1)[:5]]}")


def stamp_preprocessing(cfg, ckpt_path: Path) -> None:
    """전처리 파라미터를 labels.json 에 박는다. 서버는 이걸 읽어 쓴다(하드코딩 금지).

    학습과 서빙의 전처리가 조금이라도 다르면 정확도가 조용히 몇 %씩 떨어진다.
    값을 한 곳에 두고 양쪽이 같은 파일을 읽게 만드는 것이 유일한 방어책이다.
    """
    labels = json.loads(config.LABELS_JSON.read_text(encoding="utf-8"))
    labels["preprocess"] = {
        "img_size": cfg.img_size,
        "resize_shorter_to": int(cfg.img_size * 1.14),  # build_transforms와 같은 식
        "mean": list(cfg.mean),
        "std": list(cfg.std),
        "interpolation": "bilinear",
        "note": "짧은 변을 resize_shorter_to 로 맞춘 뒤 중앙 img_size 크롭 → /255 → (x-mean)/std",
    }
    labels["model"] = {
        "backbone": cfg.backbone,
        "onnx": config.ONNX_PATH.name,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "opset": OPSET,
        "applies_softmax": False,   # 서버에서 적용한다 (temperature scaling 여지)
        "source_checkpoint": ckpt_path.name,
    }
    config.LABELS_JSON.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 전처리 파라미터 기록: {config.LABELS_JSON}")
    print(f"       img_size={cfg.img_size} | 짧은변 리사이즈 {int(cfg.img_size * 1.14)} "
          f"→ 중앙크롭 {cfg.img_size}")


def main() -> None:
    ap = argparse.ArgumentParser(description="학습된 .pt → 서빙용 .onnx 변환·검증")
    ap.add_argument("--ckpt", type=Path, default=config.BEST_CKPT)
    ap.add_argument("--out", type=Path, default=config.ONNX_PATH)
    ap.add_argument("--verify-only", action="store_true", help="변환 없이 기존 onnx만 검증")
    ap.add_argument("--skip-real", action="store_true", help="실이미지 검증 생략")
    args = ap.parse_args()

    device = torch.device("cpu")   # 변환은 항상 CPU에서 (서빙 환경과 맞춘다)
    model, cfg = load_checkpoint(args.ckpt, device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    check_labels(ckpt.get("classes"))

    if not args.verify_only:
        export(model, cfg.img_size, args.out)
    elif not args.out.exists():
        raise SystemExit(f"[FAIL] {args.out} 가 없다 → --verify-only 없이 먼저 변환할 것")

    verify_numeric(model, args.out, cfg.img_size)
    if not args.skip_real:
        verify_real_images(model, args.out, cfg)

    if not args.verify_only:
        stamp_preprocessing(cfg, args.ckpt)

    print(f"\n[done] {args.out}")
    print("[next] server/inference.py 작성 → labels.json 의 preprocess 값을 읽어 쓸 것")


if __name__ == "__main__":
    main()
