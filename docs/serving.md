# Phase 5·6 — ONNX Export · FastAPI 서버 · Docker

`src/export_onnx.py` (2026-08-18) / `server/inference.py`·`server/main.py`·`Dockerfile` (2026-08-22)

```bash
python -m src.export_onnx                    # models/best.pt → server/model.onnx
python -m src.export_onnx --verify-only      # 기존 onnx만 검증
```

Colab에서는 `onnxscript`·`onnxruntime` 이 기본 설치돼 있지 않다:
```python
!pip install -q onnxscript onnxruntime
```

### torch 2.13 dynamo 익스포터에서 걸린 함정 2개 (2026-08-18 실측)

**1. `external_data=True` 가 기본값이다.**
가중치가 `model.onnx.data` 로 따로 빠져 `model.onnx` 는 0.5MB만 남는다.
Docker 이미지에 `.onnx` 만 복사하면 **로드 시점에 터진다.**
→ `external_data=False` 로 단일 파일(15.8MB) 고정. 크기가 5MB 미만이면 실패 처리한다.

**2. `opset_version=13` 을 요청해도 18로 나온다.**
dynamo 익스포터는 18로 뽑은 뒤 13으로 낮추려 하는데, onnx version_converter가
`axes_input_to_attribute` 에서 실패한다. 그런데 **에러를 찍고도 18로 그냥 진행**한다 —
요청과 결과가 다른데 성공한 것처럼 보인다.
→ opset 18을 그대로 쓰고(onnxruntime>=1.19가 지원), export 후 **실제 opset을 검증**한다.

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

### 전처리 — 학습과 **비트 단위로** 같다 (2026-08-22 실측)

```
EXIF 회전 반영 → RGB → 짧은 변을 437px 로 리사이즈(cv2 INTER_LINEAR)
→ 중앙 384 크롭 → (x - mean*255) * 1/(std*255) → CHW → 배치 차원
```

값은 `server/labels.json` 의 `preprocess` 블록에서 읽는다(하드코딩 금지).
검증: `python -m scripts.check_preprocess` → **val 300장 전부 diff 0.00e+00**.

맞추는 데 걸린 함정 셋:

| 항목 | 틀리기 쉬운 방식 | 실제 |
|---|---|---|
| 리사이즈 | PIL `Image.BILINEAR` | cv2 `INTER_LINEAR` — PIL은 축소 시 안티에일리어싱이 들어가 결과가 다르다(실측 정규화값 max 0.14 차이) |
| 정규화 | `img/255` 후 `(x-mean)/std` | `(img - mean*255) * 1/(std*255)` — albumentations와 같은 순서. 아니면 1e-7 어긋난다 |
| 큰 사진 | 긴 변 1024px로 미리 축소 | **미리 축소하지 않는다** (아래) |

**"큰 사진은 미리 줄인다"는 원래 계획을 폐기했다.** 폰 사진(3000px)을 INTER_LINEAR로
한 번에 437px까지 줄이면 계단이 지므로 INTER_AREA를 먼저 걸까 검토했는데,
**학습 데이터의 27.8%가 바로 그 거친 축소를 거친 것**으로 측정됐다
(짧은 변 437px 초과, 최대 8.4배 축소 — 1,200장 표본). 그 거친 축소본이 모델이 실제로
학습한 입력이므로, 서빙에서 "더 좋은" 축소를 하면 오히려 학습과 다른 그림을 넣게 된다.
대신 디컴프레션 폭탄만 화소 수(5,000만)로 막는다.

### API 명세

**`POST /predict`** — `multipart/form-data`, field `file`

```json
{
  "success": true,
  "uncertain": false,
  "model_version": "b0-384-20260818",
  "predictions": [
    {"rank": 1, "species": "붕어", "confidence": 0.82},
    {"rank": 2, "species": "잉어", "confidence": 0.11},
    {"rank": 3, "species": "가물치", "confidence": 0.03}
  ],
  "other_confidence": 0.0071,
  "top1_confidence": 0.82,
  "latency_ms": 112.4
}
```

`confidence` 는 **25클래스 softmax 값 그대로**다(`기타`를 빼고 재정규화하지 않는다).
재정규화하면 임계값·로그·평가 수치가 서로 다른 척도가 되어 비교가 불가능해진다.
그래서 `predictions` 의 합은 1이 아니며, `top1_confidence` 는 `기타`까지 포함한 최댓값이라
`predictions[0].confidence` 와 다를 수 있다(그 경우가 곧 `기타`가 1순위인 상황이다).

- `uncertain: true` 조건 (둘 중 하나라도 해당):
  1. `top1 confidence < CONFIDENCE_THRESHOLD` (초기 0.45)
  2. **Top-1이 `기타`(인덱스 24)** — 물고기가 아니거나 24종 밖으로 판단
- **`기타`는 `predictions` 배열에서 제외하고 어종만 Top-3로 반환한다.** 사용자에게 "기타"를 후보로 보여줄 이유가 없다. 대신 `"other_confidence": 0.71` 같은 필드로 근거를 남긴다.
- 실패 시: `{"success": false, "error": "<코드>", "detail": "..."}` + 4xx/5xx

| 상황 | HTTP | error |
|---|---|---|
| 빈 파일 | 400 | `EMPTY_FILE` |
| 손상·비이미지 | 400 | `IMAGE_DECODE_FAILED` |
| TIFF 등 미지원 | 415 | `UNSUPPORTED_FORMAT` |
| 업로드 10MB 초과 | 413 | `FILE_TOO_LARGE` |
| 화소 5,000만 초과 | 413 | `IMAGE_TOO_LARGE` |
| 모델 미로드 | 503 | `MODEL_NOT_LOADED` |
| 그 밖 | 500 | `INTERNAL_ERROR` |

**4xx는 앱이 재시도하지 않는다**(integration.md 계약). 잘못된 입력에 500을 주면
앱이 재시도해서 부하만 늘어나므로, 입력 문제는 전부 4xx로 분류한다.

**`GET /health`** → `{"status": "ok", "model_version": "...", "num_classes": 25, "img_size": 384, "tta": false, "confidence_threshold": 0.45}`
모델 로드 실패 시 **503** + `{"status": "unavailable", "error": "MODEL_NOT_LOADED", "detail": "<원인>"}`.
모델을 못 읽어도 프로세스는 죽이지 않는다 — 죽으면 배포 로그에 원인이 안 남는다.

**`GET /labels`** → `classes`(25) · `other_class` · `species`(학명·서식지).
앱 백엔드가 종명 문자열을 대조하는 용도다(integration.md: 종명이 두 시스템의 조인 키).

### 서버 동작

- 모델 세션은 **프로세스 시작 시 1회 로드** + 워밍업 1회(첫 요청이 최적화 비용을 물지 않게)
- 로드 시점에 그래프를 대조한다 — ONNX 입력 해상도 vs `labels.json` 의 `img_size`,
  출력 차원 vs 클래스 수. **어긋나면 서버가 아예 안 뜬다.**
  (384로 학습한 모델에 224를 넣어도 조용히 돌아가는 게 이 단계의 진짜 위험이다)
- 업로드 상한 10MB(`MAX_UPLOAD_MB`), 상한+1바이트만 읽어 판정 — 상한 없이 read하면
  메모리 사용량이 업로더 손에 있다
- 지원 포맷 JPEG/MPO/PNG/WebP **+ GIF/BMP**. MPO는 폰 연사/HDR 사진의 실제 포맷이고,
  GIF/BMP는 갤러리에서 고를 수 있는데 415로 막아서 얻는 게 없다
- EXIF 회전 반영(`ImageOps.exif_transpose`)
- 추론은 스레드풀에서 — ONNX는 CPU를 오래 잡는 동기 작업이라 이벤트 루프를 막는다
- 스레드 고정: `intra_op_num_threads=1`(`ORT_THREADS`) + `OMP_NUM_THREADS=1` + `cv2.setNumThreads(1)`.
  **셋 다 눌러야** 실제로 1스레드가 된다
- CORS: `CORS_ORIGINS` 가 비어 있으면 미들웨어를 붙이지 않는다(서버간 호출은 CORS와 무관).
  브라우저에서 직접 부를 때만 도메인을 명시한다 — 와일드카드 기본값을 두지 않는다

#### 환경변수

| 변수 | 기본 | 용도 |
|---|---|---|
| `MODEL_PATH` / `LABELS_PATH` | `server/model.onnx`, `server/labels.json` | 경로 |
| `CONFIDENCE_THRESHOLD` | labels.json (0.45) | `uncertain` 임계값 |
| `TOP_K` | labels.json (3) | 후보 개수 |
| `ORT_THREADS` | 1 | ONNX Runtime intra-op |
| `TTA` | 0 | 1이면 좌우반전 평균(추론 2배) |
| `MAX_UPLOAD_MB` | 10 | 업로드 상한 |
| `CORS_ORIGINS` | (없음) | 쉼표구분 도메인 |
| `MODEL_VERSION` | labels.json | `/health` 노출 버전 |

### 임계값 튜닝

```bash
python -m scripts.tune_threshold              # val 전체, 서빙 경로(ONNX+서버 전처리)로 측정
python -m scripts.tune_threshold --cached     # 저장된 확률로 스윕만 다시
python -m scripts.tune_threshold --write      # 고른 값을 labels.json 에 반영
```

`src/evaluate.py` 의 `threshold_curve.png` 와 **다른 것을 잰다.** 평가 쪽은 torch 경로에
"top1 < 임계값" 규칙만 적용하는데, 서버는 여기에 "top1이 `기타`"까지 OR로 묶는다.
두 규칙이 겹치므로 실제 거부율은 임계값만 보고 예측한 값보다 항상 높다 —
**배포할 값은 배포할 경로에서 재야 한다.**

기준(evaluation.md): 어종 사진 거부율 10% 이하를 유지하면서 통과분 Top-3 최대.
`기타` 사진이 거부되는 것은 비용이 아니라 이득이므로 거부율 계산에서 분리한다.

#### 결과 (2026-08-22, val 1,117장 / 실모델 / TTA off)

| 임계값 | 어종 거부율 | 통과 Top-3 | 통과 Top-1 | `기타` 검출 |
|---|---|---|---|---|
| 0.20 | 5.8% | 92.75% | 85.19% | 88.7% |
| **0.25** | **8.5%** | **93.73%** | **86.92%** | **89.6%** |
| 0.30 | 12.8% | 95.24% | 89.12% | 92.4% |
| 0.45 (기존) | **22.3%** | 97.33% | 94.40% | 95.3% |
| 0.60 | 31.8% | 98.26% | 96.38% | 97.2% |

**0.45 → 0.25 로 낮췄다.** 초기값 0.45는 근거 없이 잡은 값이었고, 실측해 보니
**어종 사진의 22.3%** 가 "다시 찍어주세요"를 보게 된다 — 기준(10%)의 두 배가 넘는다.
다섯 장에 한 장씩 재촬영을 요구하는 앱은 쓰이지 않는다.

읽을 때 주의: 통과 Top-3가 임계값과 함께 오르는 것은 **모델이 좋아져서가 아니라
어려운 사진을 빼고 채점하기 때문**이다. 이 표에서 고를 것은 최고점이 아니라
거부율 상한을 지키는 지점이다.

거부율의 바닥은 5.8%(임계값 0.20)인데, 이는 확률 규칙이 아니라 **"Top-1이 `기타`"**
규칙에서 나온다. 즉 임계값을 아무리 낮춰도 어종 사진의 약 6%는 `기타`로 오인돼
거부된다 — 이건 임계값이 아니라 `기타` 클래스 품질의 문제다(로드맵 "알려진 한계" 1).

반영 위치가 **두 곳**이다: `server/labels.json` 의 `confidence_threshold` 와
`src/config.py` 의 `CONFIDENCE_THRESHOLD`. 후자를 안 고치면 `python -m src.config --init`
한 번에 조용히 0.45로 되돌아간다.

### Docker

> EC2에 올리는 절차는 [deploy.md](deploy.md) 참조.

```bash
docker build -t fishilog-ai .
docker run --rm -p 8000:8000 fishilog-ai
curl localhost:8000/health
curl -F "file=@sample.jpg" localhost:8000/predict
```

`Dockerfile` 에 넣어둔 것과 이유:

- **`server/requirements.txt`(슬림)를 쓴다.** 루트 `requirements.txt`에는 torch가 있어 2GB를 넘는다
- `libglib2.0-0` apt 설치 — `opencv-python-headless` 는 GUI 없이도 libglib을 링크한다.
  없으면 빌드가 아니라 **런타임에** `import cv2` 에서 죽는다
- `RUN test -f server/model.onnx` — 모델 없이 빌드하면 컨테이너는 뜨는데 `/health` 가
  계속 503이다. 빌드 시점에 막는 편이 낫다
- `ORT_THREADS=1`, `OMP_NUM_THREADS=1`
- 비root(uid 10001) 실행 — 추론 서버는 파일을 쓸 일이 없다
- `HEALTHCHECK` 는 curl 대신 파이썬(slim에 curl이 없다)
- `.dockerignore` 로 `data/`(수 GB)를 제외 — 없으면 빌드 컨텍스트 전송에만 몇 분이 든다

#### 실측 (2026-08-22, 이 랩탑 / Docker Desktop / 384px b0 / TTA off)

| 항목 | 값 |
|---|---|
| 이미지 크기 | **748MB** (압축 전송 **195MB**) |
| 컨테이너 메모리 | **약 132MB** (추론 중에도 200MB 미만) |
| 추론 지연 | 중앙값 **62~80ms** (HTTP 왕복 포함, 첫 요청만 244ms) |
| 처리량 | 2워커 **18건/초** (워커당 약 9건/초) → [deploy.md](deploy.md) 2절 |
| 컨테이너 vs 로컬 | 실사진 25장 Top-3 **완전 일치**, confidence 최대차 **0.0000** |

이미지 748MB의 내역은 `cv2` 153MB + `numpy` 73MB + `onnxruntime` 66MB + python:slim
약 150MB다. 2단계 빌드로 pip·setuptools를, `uvicorn[standard]` → 필요한 것만으로
watchfiles·websockets를 걷어내 772→748MB까지만 줄었다 — **남은 것은 전부 실제로 쓰는
라이브러리라 더 줄이려면 기능을 빼야 한다.** 배포 시 실제 전송량은 압축 기준 **195MB**다.

메모리는 예상(500MB)의 1/4인 **132MB**다. 계획서가 torch 기준으로 잡은 값이었다.
1GB RAM 인스턴스에서도 충분히 돈다.

계획서의 "30~60ms"는 224px 기준이었다 — 384px로 올린 대가로 80ms가 됐고,
앱 타임아웃 5초(integration.md) 기준으로는 여유가 크다.

### 배포 전 체크리스트

자동 검사 2종이 이 목록의 대부분을 대신한다:

```bash
python -m scripts.check_preprocess     # 전처리 = 학습 전처리 (실이미지 대조)
python -m scripts.check_server         # 계약 검사 27종 (모델 없으면 더미로 진행)
```

`check_server` 는 `server/model.onnx` 가 없으면 **무작위 가중치 더미 모델**을 만들어
검사한다. 계약(4xx/503/EXIF/동시요청)은 가중치와 무관하고, 모델이 아직 없다고
검사를 못 도는 편이 더 나쁘기 때문이다.

- [x] `.pt` vs `.onnx` 수치 일치 검증 통과 (Phase 5)
- [x] 회전된 폰 사진 정상 처리 — `check_server` (EXIF Orientation=6)
- [x] 큰 파일(10MB+)·손상 파일·빈 파일 → 500이 아니라 4xx — `check_server`
- [x] 동시 요청 10개 — `check_server`
- [x] `/health` 가 모델 미로드 시 503 — `check_server`
- [x] 전처리 학습 일치 — `check_preprocess` (val 300장 diff 0)
- [x] **실모델로 위 두 검사 재실행** — 2026-08-22, 27/27 · 전처리 diff 0
- [x] `docker build` / `docker run` 실행 확인 — `/health` 200, 실사진 25장 응답 정상
- [x] 컨테이너 응답 = 로컬 추론 (confidence 최대차 0.0000)
- [x] `scripts.tune_threshold` 로 임계값 확정 — 0.45 → **0.25**
- [ ] **실사용 사진 20장 수동 테스트** (손에 든/바닥/역광/흐릿) — 실제 폰 사진 필요
- [ ] 물고기 아닌 사진(사람·풍경·요리) → `uncertain: true` 확인 (A-10)

## TTA (Test-Time Augmentation)

`src/evaluate.py --tta` 는 원본과 **좌우반전본**의 softmax를 평균낸다.
물고기는 머리가 어느 쪽을 향하든 같은 종이므로 좌우반전은 안전한 변형이고,
두 번 보고 평균내면 한쪽 방향에서만 우연히 흔들린 예측이 완화된다.

- **재학습이 필요 없다.** 이미 있는 체크포인트에 그대로 적용된다.
- 비용은 **추론 2배**. 384px efficientnet_b0 기준 이미지 1장에 수십 ms 수준이라
  서빙에서 감당 가능하다.
- 상하반전·회전은 쓰지 않는다. 학습 증강이 ±15도까지만 다뤘으므로 그 밖의 변형은
  분포를 벗어나 오히려 해롭다.

**서빙에 반영할 때 주의**: 평가에서 TTA를 켜고 성능을 보고했다면 `server/inference.py`
에서도 반드시 켜야 한다. 안 그러면 실제 서비스 정확도가 보고값보다 낮다.

→ **현재 서버 기본값은 TTA off**(`TTA=1` 로 켠다). 로드맵에 적힌 최종 수치
(test Top-3 90.66%)가 TTA 없이 측정된 값이라 그 기준에 맞춘 것이다.
켤 거면 `evaluate.py --tta` 로 val 이득을 먼저 재고, 로드맵 수치도 함께 갱신할 것 —
**한쪽만 바꾸면 보고 수치와 실제 서비스가 어긋난다.**
