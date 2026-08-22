"""serving.md "배포 전 체크리스트"를 코드로 옮긴 것. 서버를 띄우지 않고 검사한다.

여기서 보는 것은 정확도가 아니라 **계약**이다: 손상 파일에 500이 아니라 4xx가 나오는가,
모델이 없을 때 `/health` 가 503인가, 회전된 폰 사진이 제대로 서는가.
정확도는 `src/evaluate.py`, 전처리 일치는 `scripts/check_preprocess.py` 담당.

가중치는 상관없는 검사들이므로 **`server/model.onnx` 가 없으면 무작위 가중치 더미 모델**을
임시로 만들어 쓴다(`--dummy`, 기본 자동). 계약 검증은 그걸로 충분하고, 실제 모델이
없다고 이 검사를 못 도는 편이 더 나쁘다.

사용 예:
    python -m scripts.check_server                 # 있으면 실모델, 없으면 더미
    python -m scripts.check_server --dummy         # 더미 강제(실모델이 있어도)
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from src import config

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 테스트 입력 만들기
# ---------------------------------------------------------------------------
def fish_like_image(size: tuple[int, int] = (800, 600), seed: int = 0) -> Image.Image:
    """물고기는 아니지만 구조가 있는 이미지. 단색은 전처리 버그를 못 잡는다."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size[1], 0:size[0]]
    base = (np.sin(x / 23.0) * 60 + np.cos(y / 17.0) * 60 + 128)
    arr = np.stack([base, base * 0.8 + 20, base * 0.6 + 40], -1)
    arr = np.clip(arr + rng.normal(0, 8, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def to_bytes(im: Image.Image, fmt: str = "JPEG", **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


def exif_rotated(im: Image.Image) -> bytes:
    """Orientation=6(시계방향 90도 회전 필요) 태그를 붙인 JPEG.

    폰 세로 사진의 실제 저장 방식이다 — 픽셀은 눕힌 채로, 회전은 메타데이터로만.
    """
    exif = Image.Exif()
    exif[274] = 6  # 274 = Orientation
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def make_dummy_onnx(out: Path, img_size: int, num_classes: int) -> Path:
    """무작위 가중치 efficientnet_b0 → onnx. 계약 검사용(정확도와 무관)."""
    import timm
    import torch

    from src.export_onnx import INPUT_NAME, OUTPUT_NAME, OPSET

    model = timm.create_model("efficientnet_b0", pretrained=False,
                              num_classes=num_classes).eval()
    torch.onnx.export(
        model, torch.randn(1, 3, img_size, img_size), str(out),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True, external_data=False,
    )
    return out


# ---------------------------------------------------------------------------
def load_app(model_path: Path | None):
    """환경변수를 바꾼 뒤 서버 모듈을 새로 로드한다(모델 로드는 프로세스 시작 시 1회라서)."""
    if model_path is None:
        os.environ["MODEL_PATH"] = str(config.SERVER_DIR / "__없는파일__.onnx")
    else:
        os.environ["MODEL_PATH"] = str(model_path)
    import server.inference
    import server.main
    importlib.reload(server.inference)
    return importlib.reload(server.main)


def run_contract_tests(app_module, labels: dict) -> None:
    from fastapi.testclient import TestClient

    img = fish_like_image()
    ok_jpeg = to_bytes(img)

    with TestClient(app_module.app) as c:
        # --- /health ---------------------------------------------------------
        r = c.get("/health")
        check("/health 200", r.status_code == 200, str(r.status_code))
        body = r.json()
        check("/health 에 model_version·num_classes",
              body.get("num_classes") == labels["num_classes"] and "model_version" in body,
              json.dumps(body, ensure_ascii=False)[:90])

        # --- 정상 추론 --------------------------------------------------------
        r = c.post("/predict", files={"file": ("a.jpg", ok_jpeg, "image/jpeg")})
        check("정상 JPEG 200", r.status_code == 200, r.text[:120])
        res = r.json()
        preds = res.get("predictions", [])
        check("Top-3 후보 3개", len(preds) == labels["top_k"], str(len(preds)))
        check("`기타`는 후보에서 제외",
              all(p["species"] != labels["other_class"] for p in preds))
        check("후보가 24종 안의 이름", all(p["species"] in labels["classes"] for p in preds))
        check("confidence 내림차순",
              all(preds[i]["confidence"] >= preds[i + 1]["confidence"]
                  for i in range(len(preds) - 1)))
        check("rank 1..k", [p["rank"] for p in preds] == list(range(1, len(preds) + 1)))
        check("other_confidence·uncertain 존재",
              "other_confidence" in res and isinstance(res.get("uncertain"), bool))

        # --- 포맷 --------------------------------------------------------------
        for fmt, mime in (("PNG", "image/png"), ("WEBP", "image/webp")):
            r = c.post("/predict", files={"file": (f"a.{fmt.lower()}", to_bytes(img, fmt), mime)})
            check(f"{fmt} 200", r.status_code == 200, r.text[:80])

        r = c.post("/predict", files={"file": ("a.tiff", to_bytes(img, "TIFF"), "image/tiff")})
        check("TIFF는 415", r.status_code == 415, str(r.status_code))

        # --- EXIF 회전 ---------------------------------------------------------
        landscape = fish_like_image((600, 400), seed=3)
        r_rot = c.post("/predict", files={"file": ("p.jpg", exif_rotated(landscape), "image/jpeg")})
        pre_rotated = landscape.rotate(-90, expand=True)   # Orientation=6 이 뜻하는 것
        r_ref = c.post("/predict", files={"file": ("p.jpg", to_bytes(pre_rotated), "image/jpeg")})
        check("EXIF 회전 사진 200", r_rot.status_code == 200)
        check("EXIF 회전 = 미리 돌려놓은 사진과 같은 예측",
              r_rot.json()["predictions"][0]["species"]
              == r_ref.json()["predictions"][0]["species"],
              f'{r_rot.json()["predictions"][0]["species"]} vs '
              f'{r_ref.json()["predictions"][0]["species"]}')

        # --- 잘못된 입력 (전부 4xx여야 한다. 500이면 앱이 재시도해서 부하만 는다) ---
        cases = {
            "빈 파일": (b"", 400),
            "이미지가 아닌 바이트": (b"hello, not an image" * 100, 400),
            "잘린 JPEG": (ok_jpeg[:len(ok_jpeg) // 3], 400),
        }
        for name, (payload, want) in cases.items():
            r = c.post("/predict", files={"file": ("x.jpg", payload, "image/jpeg")})
            check(f"{name} → {want}", r.status_code == want,
                  f"got {r.status_code} {r.text[:80]}")
            check(f"{name} 응답에 success:false·error",
                  r.json().get("success") is False and "error" in r.json())

        # 상한 초과 업로드 (기본 10MB)
        big = b"\xff\xd8\xff\xe0" + os.urandom(11 * 1024 * 1024)
        r = c.post("/predict", files={"file": ("big.jpg", big, "image/jpeg")})
        check("11MB 업로드 → 413", r.status_code == 413, str(r.status_code))

        # 파일은 작은데 화소는 거대한 PNG (디컴프레션 폭탄)
        bomb = to_bytes(Image.new("RGB", (9000, 9000), (7, 7, 7)), "PNG")
        r = c.post("/predict", files={"file": ("bomb.png", bomb, "image/png")})
        check(f"거대 화소 PNG({len(bomb)/1e6:.1f}MB) → 413", r.status_code == 413,
              str(r.status_code))

        # --- 동시 요청 10개 ----------------------------------------------------
        def one(_: int):
            return c.post("/predict", files={"file": ("a.jpg", ok_jpeg, "image/jpeg")}).status_code

        with ThreadPoolExecutor(max_workers=10) as ex:
            codes = list(ex.map(one, range(10)))
        check("동시 요청 10개 전부 200", all(c_ == 200 for c_ in codes), str(codes))

        # --- /labels -----------------------------------------------------------
        r = c.get("/labels")
        check("/labels 가 classes 25개 반환",
              r.status_code == 200 and r.json()["classes"] == labels["classes"])


def run_no_model_tests() -> None:
    from fastapi.testclient import TestClient

    mod = load_app(None)
    with TestClient(mod.app) as c:
        r = c.get("/health")
        check("모델 미로드 시 /health 503", r.status_code == 503, str(r.status_code))
        check("/health 에 원인 노출", r.json().get("error") == "MODEL_NOT_LOADED")
        r = c.post("/predict", files={"file": ("a.jpg", to_bytes(fish_like_image()), "image/jpeg")})
        check("모델 미로드 시 /predict 503", r.status_code == 503, str(r.status_code))


def main() -> None:
    ap = argparse.ArgumentParser(description="서버 계약 검사 (serving.md 배포 전 체크리스트)")
    ap.add_argument("--dummy", action="store_true", help="실모델이 있어도 더미로 검사")
    args = ap.parse_args()

    labels = json.loads(config.LABELS_JSON.read_text(encoding="utf-8"))
    img_size = labels["preprocess"]["img_size"]

    with tempfile.TemporaryDirectory() as tmp:
        real = config.ONNX_PATH
        if real.exists() and not args.dummy:
            model_path = real
            print(f"[cfg] 실모델 {real.name} 로 검사")
        else:
            model_path = make_dummy_onnx(Path(tmp) / "dummy.onnx", img_size,
                                         labels["num_classes"])
            why = "강제" if args.dummy else f"{real} 없음"
            print(f"[cfg] 더미 모델로 검사 ({why}) — 정확도가 아니라 계약만 본다")

        print("\n[1] 모델 로드 상태")
        run_contract_tests(load_app(model_path), labels)

        print("\n[2] 모델 미로드 상태")
        run_no_model_tests()

    print(f"\n[{'done' if not FAIL else 'FAIL'}] 통과 {len(PASS)} / 실패 {len(FAIL)}")
    if FAIL:
        for name in FAIL:
            print(f"  - {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
