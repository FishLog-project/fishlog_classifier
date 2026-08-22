"""Phase 6 — FastAPI 추론 서버. 상태를 갖지 않는다(integration.md).

이 파일은 얇다. 추론·전처리·검증은 전부 `inference.py` 에 있고, 여기서는
HTTP 규약만 지킨다:

- 모델은 **프로세스 시작 시 1회** 로드한다. 실패해도 프로세스는 살려두고
  `/health` 가 503을 반환하게 한다 — 죽어버리면 배포 헬스체크가 "왜" 죽었는지 못 알려준다.
- 잘못된 업로드는 **500이 아니라 4xx**. 앱은 4xx를 재시도하지 않는다(integration.md 계약).
- 업로드는 상한(기본 10MB)까지만 읽는다. 상한 없이 read()하면 메모리가 업로더 손에 있다.

실행:
    uvicorn server.main:app --host 0.0.0.0 --port 8000
환경변수는 `inference.py` 상단 참조 + 아래:
    MAX_UPLOAD_MB   기본 10
    CORS_ORIGINS    쉼표구분. 비우면 CORS 미들웨어를 아예 안 붙인다(서버간 호출은 무관)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from server.inference import InferenceError, Predictor

log = logging.getLogger("fishlog")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")

MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)

# 모델 로드 결과를 담아둔다. predictor 가 None이면 서버는 살아 있지만 추론 불가 상태다.
state: dict[str, object] = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        p = Predictor()
        state["predictor"] = p
        log.info("모델 로드 완료: %s | %d클래스 | %dpx | 버전 %s | TTA %s | 워밍업 %.0fms",
                 p.model_path.name, len(p.classes), p.pp.img_size, p.model_version,
                 "on" if p.tta else "off", p.warmup_ms)
    except Exception as exc:  # noqa: BLE001 — 어떤 이유든 서버는 뜨고 /health가 503을 알린다
        state["error"] = str(exc)
        log.error("모델 로드 실패 — /predict 는 503을 반환한다:\n%s", exc)
    yield


app = FastAPI(
    title="fishilog-ai 어류 분류 서버",
    version="1.0",
    description="사진 → 어종 Top-3 후보. 확정은 앱에서 사용자가 한다.",
    lifespan=lifespan,
)

_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    # 앱 백엔드 ↔ 모델 서버는 서버간 호출이라 CORS가 필요 없다. 브라우저에서 직접
    # 부르는 경우에만 도메인을 명시적으로 넣는다(와일드카드 기본값을 두지 않는 이유).
    app.add_middleware(CORSMiddleware, allow_origins=_origins,
                       allow_methods=["POST", "GET"], allow_headers=["*"])


def _fail(status: int, code: str, detail: str | None = None) -> JSONResponse:
    body = {"success": False, "error": code}
    if detail:
        body["detail"] = detail
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(500)
async def _internal_error(request: Request, exc: Exception) -> JSONResponse:
    return _fail(500, "INTERNAL_ERROR")


@app.get("/health")
async def health() -> JSONResponse:
    """배포 헬스체크용. 모델이 없으면 503이어야 롤아웃이 멈춘다."""
    p = state["predictor"]
    if p is None:
        return JSONResponse(status_code=503,
                            content={"status": "unavailable", "error": "MODEL_NOT_LOADED",
                                     "detail": state["error"]})
    return JSONResponse({
        "status": "ok",
        "model_version": p.model_version,
        "num_classes": len(p.classes),
        "img_size": p.pp.img_size,
        "tta": p.tta,
        "confidence_threshold": p.threshold,
    })


@app.get("/labels")
async def labels() -> JSONResponse:
    """앱 백엔드가 종명 문자열을 대조할 수 있게 노출한다(integration.md: 종명이 조인 키다)."""
    p = state["predictor"]
    if p is None:
        return _fail(503, "MODEL_NOT_LOADED")
    return JSONResponse({
        "classes": p.classes,
        "other_class": p.other_class,
        "species": p.species,     # 학명·서식지 — 앱 표시용
    })


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    p = state["predictor"]
    if p is None:
        return _fail(503, "MODEL_NOT_LOADED", state["error"])

    # 상한 + 1바이트만 읽어 초과 여부를 판정한다 (전체를 읽고 나서 재는 건 이미 늦다)
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return _fail(413, "FILE_TOO_LARGE", f"최대 {MAX_UPLOAD_BYTES // 1024 // 1024}MB")

    try:
        # ONNX 추론은 CPU를 오래 잡는 동기 작업이다 — 이벤트 루프를 막지 않게 스레드로 뺀다
        result = await run_in_threadpool(p.predict, raw)
    except InferenceError as exc:
        return _fail(exc.http_status, exc.code, str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("추론 실패: %s", exc)
        return _fail(500, "INTERNAL_ERROR")

    return JSONResponse(result)
