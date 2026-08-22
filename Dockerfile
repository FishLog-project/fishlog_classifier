# fishilog-ai 추론 서버. ONNX Runtime CPU 전용 — torch/timm은 들어가지 않는다.
# (루트 requirements.txt 를 쓰면 torch 때문에 이미지가 2GB를 넘는다 → server/requirements.txt)
#
# 2단계 빌드: 설치 도구(pip·setuptools·wheel, 약 25MB)를 최종 이미지에 남기지 않는다.
# ---- 1단계: 의존성 설치 -----------------------------------------------------
FROM python:3.11-slim AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip uninstall -y pip setuptools wheel 2>/dev/null || true

# ---- 2단계: 실행 이미지 -----------------------------------------------------
FROM python:3.11-slim

# opencv-python-headless 는 GUI 없이도 libglib 계열을 링크한다. 없으면 `import cv2` 에서
# ImportError(libgthread-2.0.so.0)로 죽는다 — 빌드가 아니라 런타임에 터지므로 미리 넣는다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY server/ ./server/

# model.onnx 는 .gitignore 대상이라 클론만으로는 생기지 않는다. 없는 채로 빌드하면
# 컨테이너는 뜨지만 /health 가 계속 503이다 → 빌드 시점에 막는다.
RUN test -f server/model.onnx || ( \
      echo "[FAIL] server/model.onnx 가 빌드 컨텍스트에 없다." && \
      echo "  → python -m src.export_onnx 산출물을 server/ 에 두고 다시 빌드할 것" && \
      exit 1 )

# 컨테이너 CPU 제한 환경에서 스레드를 많이 쓰면 컨텍스트 스위칭으로 오히려 느려진다.
# ORT 안쪽(intra_op)과 OpenMP 양쪽을 다 눌러야 실제로 1스레드가 된다.
# 추론 1건이 코어 하나를 80ms 동안 통째로 쓴다 → 처리량은 스레드가 아니라 워커 수로 늘린다.
# EC2 vCPU에 맞춰 `docker run -e WEB_CONCURRENCY=2` 로 덮어쓴다(워커당 메모리 약 130MB).
ENV ORT_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WEB_CONCURRENCY=1

# 추론 서버는 파일을 쓸 일이 없다 — root로 돌릴 이유가 없다
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000

# curl이 slim 이미지에 없으므로 파이썬으로 확인한다. 모델 미로드 시 /health가 503이라
# 이 헬스체크가 곧 "모델까지 정상"의 판정이 된다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health').read()"

# --workers 를 명시하지 않는다 — 명시하면 WEB_CONCURRENCY 를 덮어써서 조절이 막힌다.
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
