# Phase 5·6 — ONNX Export · FastAPI 서버 · Docker

구현 예정: `src/export_onnx.py`, `server/inference.py`, `server/main.py`, `Dockerfile` (모두 미작성)

## Phase 5 — ONNX 변환

```bash
python -m src.export_onnx --ckpt models/best.pt --out server/model.onnx
```

요구사항:
- opset 12 이상, `dynamic_axes={"input": {0: "batch"}}` (배치 가변)
- 입력 이름 `input`, 출력 이름 `logits` 로 고정
- **검증 필수**: 동일 입력에 대해 `.pt`와 `.onnx` 출력의 `max abs diff < 1e-4`
- 체크포인트의 `classes` 와 `server/labels.json` 의 `classes` 가 **완전히 동일한지** 확인 (다르면 라벨이 통째로 어긋난다)
- softmax는 모델 밖(서버)에서 적용한다 — 그래야 temperature scaling을 나중에 붙일 수 있다

## Phase 6 — 추론 서버

### 전처리 (학습과 100% 동일해야 함)

```
RGB 변환 → 짧은 변을 img_size*1.14 로 리사이즈 → 중앙 크롭 img_size
→ /255 → ImageNet mean/std 정규화 → CHW → float32 배치 차원 추가
```

⚠️ 전처리가 학습과 조금이라도 다르면 정확도가 조용히 몇 %씩 떨어진다.
값은 `server/labels.json`과 체크포인트에 저장된 `img_size`/`mean`/`std`를 읽어 쓴다. 하드코딩 금지.

### API 명세

**`POST /predict`** — `multipart/form-data`, field `file`

```json
{
  "success": true,
  "uncertain": false,
  "model_version": "b0-20260813",
  "predictions": [
    {"rank": 1, "species": "붕어", "confidence": 0.82},
    {"rank": 2, "species": "잉어", "confidence": 0.11},
    {"rank": 3, "species": "향어", "confidence": 0.03}
  ]
}
```

- `uncertain: true` 조건 (둘 중 하나라도 해당):
  1. `top1 confidence < CONFIDENCE_THRESHOLD` (초기 0.45)
  2. **Top-1이 `기타`(인덱스 24)** — 물고기가 아니거나 24종 밖으로 판단
- **`기타`는 `predictions` 배열에서 제외하고 어종만 Top-3로 반환한다.** 사용자에게 "기타"를 후보로 보여줄 이유가 없다. 대신 `"other_confidence": 0.71` 같은 필드로 근거를 남긴다.
- 실패 시: `{"success": false, "error": "IMAGE_DECODE_FAILED"}` + 4xx

**`GET /health`** → `{"status": "ok", "model_version": "...", "num_classes": 25}`
(모델 로드 실패 시 503을 반환해야 배포 헬스체크가 작동한다)

### 서버 요구사항

- 모델 세션은 **프로세스 시작 시 1회 로드** (요청마다 로드 금지)
- 업로드 크기 제한 (기본 10MB), 지원 포맷 JPEG/PNG/WebP
- 긴 변 1024px 초과 시 서버에서 축소 후 전처리
- EXIF 회전 반영 (`PIL.ImageOps.exif_transpose`) — 폰 사진은 세로로 찍혀도 메타데이터로만 회전돼 있다
- ONNX Runtime 스레드 수 고정: `sess_options.intra_op_num_threads = 1~2` (컨테이너 CPU 제한 환경에서 과도한 스레드는 오히려 느리다)
- CORS: 앱 백엔드 도메인만 허용

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server/ ./server/
EXPOSE 8000
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- **`server/requirements.txt`(슬림)를 쓴다.** 루트 `requirements.txt`에는 torch가 있어 이미지가 2GB를 넘는다.
- `server/` 안에 `model.onnx`와 `labels.json`이 함께 들어가야 한다.
- 예상 이미지 크기 ~300MB, 메모리 ~500MB, CPU 1코어에서 b0 224px 추론 30~60ms.

### 배포 전 체크리스트

- [ ] `.pt` vs `.onnx` 수치 일치 검증 통과
- [ ] 실사용 사진 20장으로 수동 테스트 (손에 든/바닥/역광/흐릿)
- [ ] 물고기가 아닌 사진(사람·풍경·요리) 입력 시 `uncertain: true` 반환 확인 (`기타` 클래스, A-10)
- [ ] 회전된 폰 사진 정상 처리
- [ ] 큰 파일(10MB+)·손상 파일·빈 파일 업로드 시 500이 아니라 4xx
- [ ] 동시 요청 10개 부하 확인
- [ ] `/health`가 모델 미로드 시 503
