"""Phase 6 — ONNX 추론 코어. FastAPI(`main.py`)는 이 파일을 얇게 감싸기만 한다.

**torch/timm을 쓰지 않는다** (conventions.md 파일 규칙). onnxruntime + OpenCV + Pillow만
쓰므로 Docker 이미지가 2GB가 아니라 400MB 안쪽에 들어온다.

이 파일이 막으려는 사고는 "에러 없이 조용히 틀리는 것" 셋이다:

1. **전처리 불일치** — 학습과 1px, 1개 보간 플래그만 달라도 정확도가 몇 %씩 조용히 샌다.
   그래서 값(img_size/mean/std)은 `labels.json` 의 `preprocess` 블록에서 읽고,
   리사이즈·크롭은 albumentations가 쓰는 **cv2 연산을 그대로** 재현한다.
2. **라벨 어긋남** — 모델 출력 25개와 `classes` 25개의 순서가 다르면 종명이 통째로 밀린다.
   ONNX 출력 차원과 클래스 수를 로드 시점에 대조한다.
3. **해상도 어긋남** — 384로 학습한 모델에 224 입력을 넣어도 ONNX는 (동적 축이면) 그냥 돈다.
   `labels.json` 의 img_size와 ONNX 입력 shape를 대조해 시작 자체를 막는다.

환경변수(전부 선택):
    MODEL_PATH / LABELS_PATH        기본 server/model.onnx, server/labels.json
    CONFIDENCE_THRESHOLD            기본 labels.json 값(0.45)
    TOP_K                           기본 labels.json 값(3)
    ORT_THREADS                     기본 1 (컨테이너 1코어 가정)
    TTA                             1이면 좌우반전 평균 (추론 2배 비용)
    MODEL_VERSION                   기본 labels.json 의 model.version

단독 실행하면 이미지 파일 하나를 추론해 본다:
    python -m server.inference reports/ref_붕어_잉어.jpg
"""

from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SERVER_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SERVER_DIR / "model.onnx"
DEFAULT_LABELS = SERVER_DIR / "labels.json"

# 폰 사진은 MPO(연사/HDR JPEG 변형)로 올라오는 일이 잦다 — 거부하면 정상 사진이 막힌다.
# GIF/BMP는 serving.md 명세 밖이지만 받아준다: 갤러리에서 고른 사진이 그럴 수 있고,
# 415로 막아서 얻는 게 없다(PIL이 첫 프레임을 RGB로 디코딩한다).
ALLOWED_FORMATS = {"JPEG", "JPG", "MPO", "PNG", "WEBP", "GIF", "BMP"}

# 디컴프레션 폭탄 방어. 이 이상은 디코딩 자체가 초 단위로 걸리고 메모리를 삼킨다.
# (5000만 화소 = 요즘 폰 최대치보다 넉넉하다)
MAX_PIXELS = 50_000_000

cv2.setNumThreads(1)  # ORT 스레드와 경합하지 않게 (컨테이너 CPU 제한 환경)


class InferenceError(Exception):
    """서버가 4xx로 돌려줄 수 있는, 입력 때문에 생긴 오류."""

    code = "INFERENCE_FAILED"
    http_status = 400


class EmptyFileError(InferenceError):
    code = "EMPTY_FILE"


class UnsupportedFormatError(InferenceError):
    code = "UNSUPPORTED_FORMAT"
    http_status = 415


class ImageDecodeError(InferenceError):
    code = "IMAGE_DECODE_FAILED"


class ImageTooLargeError(InferenceError):
    code = "IMAGE_TOO_LARGE"
    http_status = 413


# ---------------------------------------------------------------------------
# 전처리
# ---------------------------------------------------------------------------
@dataclass
class Preprocess:
    """`labels.json` 의 preprocess 블록. 하드코딩 금지(conventions.md 절대규칙 6)."""

    img_size: int
    resize_shorter_to: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __post_init__(self) -> None:
        # albumentations `Normalize` 와 **같은 식·같은 순서**로 계산해야 비트 단위로 일치한다:
        #   (img - mean*255) * (1 / (std*255))
        # `img/255` 를 먼저 하는 흔한 변형은 결과가 1e-7 수준으로 어긋난다.
        self.mean_px = (np.asarray(self.mean, np.float32) * 255.0).reshape(1, 1, 3)
        self.denominator = np.reciprocal(
            np.asarray(self.std, np.float32) * 255.0).reshape(1, 1, 3)

    @classmethod
    def from_labels(cls, labels: dict, labels_path: Path) -> "Preprocess":
        pp = labels.get("preprocess")
        if not pp:
            raise RuntimeError(
                f"{labels_path} 에 preprocess 블록이 없다 — 전처리 값을 추측할 수 없다.\n"
                "  → `python -m src.export_onnx --ckpt <최종.pt>` 로 다시 뽑고\n"
                "     생성된 labels.json 을 함께 커밋할 것 (serving.md 참조)."
            )
        missing = [k for k in ("img_size", "resize_shorter_to", "mean", "std") if k not in pp]
        if missing:
            raise RuntimeError(f"{labels_path} preprocess 블록에 {missing} 가 없다")
        return cls(
            img_size=int(pp["img_size"]),
            resize_shorter_to=int(pp["resize_shorter_to"]),
            mean=tuple(float(v) for v in pp["mean"]),
            std=tuple(float(v) for v in pp["std"]),
        )


def decode_image(raw: bytes) -> np.ndarray:
    """업로드 바이트 → RGB ndarray. EXIF 회전을 반영한다.

    폰 사진은 세로로 찍혀도 픽셀은 가로이고 **EXIF Orientation 태그로만** 회전돼 있다.
    이걸 반영하지 않으면 90도 누운 물고기를 추론하게 된다.
    """
    if not raw:
        raise EmptyFileError("빈 파일")
    try:
        im = Image.open(io.BytesIO(raw))
        fmt = (im.format or "").upper()
    except Exception as exc:  # noqa: BLE001 — 손상 파일은 종류를 가리지 않고 400
        raise ImageDecodeError(f"이미지로 열 수 없음: {exc}") from exc

    if fmt not in ALLOWED_FORMATS:
        raise UnsupportedFormatError(f"지원하지 않는 형식: {fmt or 'unknown'}")

    # 픽셀 수는 파일 크기와 별개다 — 10MB 안에도 1억 화소 PNG가 들어온다.
    # 헤더만 읽은 시점(디코딩 전)에 막아야 비용을 물지 않는다.
    px = im.size[0] * im.size[1]
    if px > MAX_PIXELS:
        raise ImageTooLargeError(f"{im.size[0]}x{im.size[1]} = {px / 1e6:.0f}MP "
                                 f"(최대 {MAX_PIXELS / 1e6:.0f}MP)")

    try:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")  # 실제 픽셀 디코딩은 여기서 일어난다 (잘린 파일이면 여기서 터짐)
        arr = np.asarray(im, dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        raise ImageDecodeError(f"디코딩 실패: {exc}") from exc

    if arr.ndim != 3 or arr.shape[2] != 3 or min(arr.shape[:2]) < 2:
        raise ImageDecodeError(f"이상한 이미지 shape: {arr.shape}")
    return arr


def _resize_shorter(img: np.ndarray, target: int, interpolation: int) -> np.ndarray:
    """albumentations `SmallestMaxSize` 와 동일한 스케일/반올림 규칙."""
    h, w = img.shape[:2]
    scale = target / min(h, w)
    if scale == 1.0:
        return img
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=interpolation)


def preprocess_image(img: np.ndarray, pp: Preprocess) -> np.ndarray:
    """RGB ndarray → (1, 3, S, S) float32. 학습 val 변환과 **비트 단위로 같은** 결과를 낸다.

    학습: `SmallestMaxSize(img_size*1.14)` → `CenterCrop(img_size)` → 정규화
    (`src/dataset.build_transforms("val", cfg)`). 보간 플래그·반올림·정규화 식까지 맞췄다
    → `python -m scripts.check_preprocess` 로 실이미지 대조.

    **큰 사진을 미리 줄이지 않는다.** 폰 사진(3000px)을 INTER_LINEAR로 한 번에 437px까지
    줄이면 안티에일리어싱이 없어 계단이 지는데, 학습 데이터의 **27.8%가 바로 그 처리를
    거쳤다**(짧은 변 437px 초과, 최대 8.4배 축소). 그 "거친 축소본"이 모델이 실제로 학습한
    입력이므로, 보기 좋으라고 INTER_AREA를 끼워 넣으면 학습과 다른 그림을 넣는 셈이 된다.
    """
    img = _resize_shorter(img, pp.resize_shorter_to, cv2.INTER_LINEAR)

    s = pp.img_size
    h, w = img.shape[:2]
    top, left = (h - s) // 2, (w - s) // 2          # albumentations CenterCrop과 동일
    img = img[top:top + s, left:left + s]

    x = (img.astype(np.float32) - pp.mean_px) * pp.denominator
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v not in (None, "") else default


class Predictor:
    """ONNX 세션 + 라벨 + 전처리를 한 덩어리로 들고 있는 추론기.

    **프로세스당 1회만 만든다.** 요청마다 세션을 만들면 매번 수백 ms가 날아간다.
    """

    def __init__(self, model_path: Path | None = None, labels_path: Path | None = None,
                 tta: bool | None = None) -> None:
        import onnxruntime as ort  # 지연 import — 이 모듈을 도구에서 부분적으로 쓸 때를 위해

        self.model_path = Path(model_path or os.getenv("MODEL_PATH") or DEFAULT_MODEL)
        self.labels_path = Path(labels_path or os.getenv("LABELS_PATH") or DEFAULT_LABELS)

        if not self.labels_path.exists():
            raise RuntimeError(f"labels.json 이 없다: {self.labels_path}")
        if not self.model_path.exists():
            raise RuntimeError(
                f"model.onnx 가 없다: {self.model_path}\n"
                "  → `python -m src.export_onnx` 산출물을 server/ 에 둘 것 "
                "(git 제외 파일이라 클론만으로는 생기지 않는다)."
            )

        labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
        self.classes: list[str] = labels["classes"]
        self.other_index: int = int(labels["other_index"])
        self.other_class: str = labels["other_class"]
        self.species: dict = labels.get("species", {})
        self.pp = Preprocess.from_labels(labels, self.labels_path)

        self.threshold = _env_float("CONFIDENCE_THRESHOLD",
                                    float(labels.get("confidence_threshold", 0.45)))
        self.top_k = _env_int("TOP_K", int(labels.get("top_k", 3)))
        model_meta = labels.get("model", {})
        self.model_version = (os.getenv("MODEL_VERSION")
                              or model_meta.get("version") or "unknown")
        self.input_name = model_meta.get("input_name", "input")
        self.output_name = model_meta.get("output_name", "logits")
        self.tta = tta if tta is not None else os.getenv("TTA", "0") not in ("0", "", "false")

        so = ort.SessionOptions()
        # 컨테이너 CPU 제한 환경에서 스레드를 많이 쓰면 컨텍스트 스위칭으로 오히려 느려진다
        so.intra_op_num_threads = _env_int("ORT_THREADS", 1)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), so,
                                            providers=["CPUExecutionProvider"])
        self._check_graph()

        # 워밍업: 첫 요청이 최적화 비용을 뒤집어쓰지 않도록 여기서 한 번 돌린다
        t0 = time.perf_counter()
        self._run(np.zeros((1, 3, self.pp.img_size, self.pp.img_size), np.float32))
        self.warmup_ms = (time.perf_counter() - t0) * 1000

    # -- 로드 시점 검증 ------------------------------------------------------
    def _check_graph(self) -> None:
        """모델 그래프가 labels.json 의 주장과 맞는지 대조한다. 안 맞으면 아예 안 뜬다."""
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"입출력이 1개씩이 아니다: in={len(inputs)} out={len(outputs)}")

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        shape = inputs[0].shape          # 보통 ['batch', 3, S, S]
        if len(shape) != 4:
            raise RuntimeError(f"입력 rank가 4가 아니다: {shape}")
        chan, h, w = shape[1], shape[2], shape[3]
        if isinstance(chan, int) and chan != 3:
            raise RuntimeError(f"입력 채널이 3이 아니다: {shape}")
        for dim in (h, w):
            # 해상도가 고정이면 labels.json 의 img_size와 반드시 같아야 한다.
            # (384로 학습한 모델에 224를 넣어도 동적 축이면 조용히 돌아간다 — 그게 무섭다)
            if isinstance(dim, int) and dim != self.pp.img_size:
                raise RuntimeError(
                    f"ONNX 입력 해상도({h}x{w})와 labels.json img_size"
                    f"({self.pp.img_size})가 다르다 — 전처리가 어긋난다.\n"
                    "  → export에 쓴 체크포인트와 labels.json 이 같은 학습 결과인지 확인할 것."
                )

        out_dim = outputs[0].shape[-1]
        if isinstance(out_dim, int) and out_dim != len(self.classes):
            raise RuntimeError(
                f"모델 출력 {out_dim}개 ≠ labels.json classes {len(self.classes)}개 — "
                "라벨이 통째로 어긋난다."
            )
        if self.classes[self.other_index] != self.other_class:
            raise RuntimeError(
                f"other_index({self.other_index})가 가리키는 것이 "
                f"'{self.classes[self.other_index]}' 다 ('{self.other_class}' 여야 함)"
            )

    # -- 추론 ---------------------------------------------------------------
    def _run(self, x: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: x})[0]

    def infer_probs(self, img: np.ndarray) -> np.ndarray:
        """RGB ndarray → 클래스 확률 (25,)."""
        x = preprocess_image(img, self.pp)
        probs = softmax(self._run(x)[0].astype(np.float32))
        if self.tta:
            # 물고기는 머리가 어느 쪽을 향해도 같은 종이므로 좌우반전은 안전한 변형이다.
            # 비용은 추론 2배 — 켤지 말지는 evaluate.py --tta 결과와 **반드시 일치**시킬 것.
            flipped = np.ascontiguousarray(x[:, :, :, ::-1])
            probs = (probs + softmax(self._run(flipped)[0].astype(np.float32))) / 2
        return probs

    def predict(self, raw: bytes) -> dict:
        """업로드 바이트 → API 응답 dict."""
        t0 = time.perf_counter()
        probs = self.infer_probs(decode_image(raw))
        return self.format_result(probs, latency_ms=(time.perf_counter() - t0) * 1000)

    def format_result(self, probs: np.ndarray, latency_ms: float | None = None) -> dict:
        """확률 → 앱이 받는 응답 형태.

        `기타`는 **후보 목록에서 뺀다.** 사용자에게 "기타"를 선택지로 보여줄 이유가 없다
        (serving.md). 대신 판단 근거로 `other_confidence` 를 남긴다.
        확률은 재정규화하지 않는다 — 25클래스 softmax 값 그대로여야 임계값·로그가 일관된다.
        """
        top1_idx = int(probs.argmax())
        top1_conf = float(probs[top1_idx])

        order = [i for i in np.argsort(-probs) if i != self.other_index][:self.top_k]
        predictions = [
            {"rank": r, "species": self.classes[i], "confidence": round(float(probs[i]), 4)}
            for r, i in enumerate(order, start=1)
        ]

        result = {
            "success": True,
            # 둘 중 하나라도 걸리면 앱은 "다시 찍어주세요"를 띄운다(후보는 그대로 보여준다).
            #   1) 최고 확률이 임계값 미만 — 모델이 자신 없다
            #   2) Top-1이 `기타` — 물고기가 아니거나 24종 밖이다
            "uncertain": bool(top1_conf < self.threshold or top1_idx == self.other_index),
            "model_version": self.model_version,
            "predictions": predictions,
            "other_confidence": round(float(probs[self.other_index]), 4),
            "top1_confidence": round(top1_conf, 4),
        }
        if latency_ms is not None:
            result["latency_ms"] = round(latency_ms, 1)
        return result


# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="ONNX 추론 단독 실행(디버깅용)")
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    p = Predictor(args.model, args.labels, tta=args.tta or None)
    print(f"[OK] {p.model_path.name} | {len(p.classes)}클래스 | {p.pp.img_size}px | "
          f"버전 {p.model_version} | TTA {'on' if p.tta else 'off'} | "
          f"워밍업 {p.warmup_ms:.0f}ms")
    for path in args.images:
        r = p.predict(path.read_bytes())
        cands = " ".join(f"{c['species']} {c['confidence']:.3f}" for c in r["predictions"])
        print(f"  {path.name}: {cands} | 기타 {r['other_confidence']:.3f} | "
              f"uncertain={r['uncertain']} | {r['latency_ms']:.0f}ms")


if __name__ == "__main__":
    main()
