# 24종 어류 분류 CNN + FastAPI 배포 — 개발 계획서 (원본 아카이브)

> ⚠️ **이 문서는 2026-08-12 착수 시점의 원본 계획서다. 아카이브 목적으로 보존한다.**
> 이후 결정으로 달라진 부분이 있으므로, **현재 기준은 [roadmap.md](roadmap.md)(진행)와
> [decisions.md](decisions.md)(결정)가 우선한다.**
>
> 원본 대비 주요 변경:
> - 24종 → **25클래스** (어종 24 + `기타` OOD 클래스, 인덱스 24 고정) — decisions A-10
> - 수집: 전량 13,150장 → **MVP 종당 250장 우선**, 평가 후 부족한 종만 보강 — A-11
> - 학습 환경: **Colab 무료 T4**, 로컬은 코드 검증용 — A-12
> - 레포 루트는 `fish-classifier/`가 아니라 `C:\fishilog_ai`, `scripts/`·`reports/`·`docs/` 추가
> - 배포 이미지용 슬림 의존성 `server/requirements.txt` 분리 (torch 미탑재)

---

> 이 문서는 새 Claude 세션에 그대로 붙여넣고 착수할 수 있도록 정리된 실행 계획서다.
> 개념 설명은 최소화하고 "무엇을 어떤 순서로 하는지"에 집중한다.

---

## 0. 프로젝트 개요

- **목표**: 사용자가 찍은 물고기 사진을 CNN으로 인식해 24종 중 어떤 종인지 후보를 제시. 사용자가 직접 확정·매핑.
- **서비스 흐름**: 사진 촬영 → AI 모델 서버가 **Top-3 후보 + confidence** 반환 → 사용자가 확정 인증 → 스팟에 매핑.
- **핵심 결정사항 (이미 확정)**:
  - 밑바닥 학습 ❌ → **전이학습(transfer learning)** 사용.
  - Top-1 정답 강요 ❌ → **Top-3 후보 제시** (사용자가 확정하므로 Top-3 안에만 정답 들어오면 됨).
  - AI 추론은 **별도 FastAPI 모델 서버(마이크로서비스)**로 분리 배포.
- **연동 데이터 (완료됨)**: `fishing_spots_all.json` (담수 50 + 바다 49 스팟, 종별 매핑). 모델과 별개로 앱 백엔드에서 사용.

---

## 1. 기술 스택

| 구분 | 선택 | 비고 |
|---|---|---|
| 언어 | Python 3.10+ | |
| 학습 | PyTorch + `timm` | EfficientNet/ConvNeXt 등 백본 파인튜닝 |
| 증강 | `albumentations` | |
| 서빙 | FastAPI + Uvicorn | 별도 추론 서버 |
| 추론 런타임 | ONNX Runtime | CPU 단일추론 빠름, GPU 선택 |
| 배포 | Docker | |
| (대안) | Ultralytics **YOLOv8-cls** | 더 빠른 경로. 데이터만 폴더로 넣으면 학습 1줄. MVP 급하면 이걸로 시작 가능 |

---

## 2. 레포 구조

```
fish-classifier/
├── data/
│   ├── raw/                # 수집 원본 (종별 폴더)
│   │   ├── 붕어/  잉어/  배스/ ... (24개 폴더)
│   ├── clean/              # 정제 후
│   └── splits/             # train/val/test 분할 결과
│       ├── train/  val/  test/   (각 폴더 아래 24개 클래스 폴더)
├── src/
│   ├── config.py           # 클래스 목록, 하이퍼파라미터
│   ├── dataset.py          # Dataset/DataLoader + 증강
│   ├── train.py            # 학습 루프
│   ├── evaluate.py         # 정확도/혼동행렬/Top-3
│   └── export_onnx.py      # 학습된 모델 → ONNX
├── server/
│   ├── main.py             # FastAPI 앱
│   ├── inference.py        # 모델 로드 + 전처리 + 추론
│   ├── labels.json         # 인덱스→종명 매핑
│   └── model.onnx          # 배포용 모델
├── models/                 # 체크포인트(.pt) 저장
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 3. 24종 목록 · 난이도 · 데이터 수집 우선순위

> 난이도 = 시각적 구분 난이도. **어려운 종일수록 이미지를 더 많이 모은다.**

| 종 | 난이도 | 권장 이미지 수 | 헷갈리는 상대 |
|---|---|---|---|
| 돌돔 | 쉬움 | 300~400 | (세로 줄무늬로 독보적) |
| 참돔 | 쉬움 | 300~400 | (붉은색) |
| 광어 | 쉬움 | 300~400 | (납작) |
| 쏘가리 | 쉬움 | 300~400 | (표범무늬) |
| 배스 | 쉬움 | 300~400 | |
| 블루길 | 쉬움 | 300~400 | |
| 가물치 | 쉬움 | 300~400 | |
| 메기 | 쉬움 | 300~400 | 동자개 |
| 동자개 | 쉬움 | 300~400 | 메기 |
| 갈치 | 쉬움 | 300~400 | |
| 방어 | 쉬움 | 300~400 | |
| 피라미 | 쉬움 | 300~400 | |
| 송어 | 쉬움 | 300~400 | |
| 고등어 | 보통 | 500~700 | 전갱이·삼치 |
| 삼치 | 보통 | 500~700 | 고등어 |
| 전갱이 | 보통 | 500~700 | 고등어 |
| 숭어 | 보통 | 500~700 | 농어 |
| 농어 | 보통 | 500~700 | 숭어 |
| 벵에돔 | 보통 | 500~700 | 감성돔 |
| 붕어 | **어려움** | 800~1200 | 잉어 |
| 잉어 | **어려움** | 800~1200 | 붕어 |
| 감성돔 | **어려움** | 800~1200 | 벵에돔 |
| 우럭 | **어려움** | 800~1200 | 볼락 |
| 볼락 | **어려움** | 800~1200 | 우럭 |

- **총 목표량**: 약 **14,000~15,000장** (수집 시작 후 부족한 종은 추가).
- **데이터 출처**: iNaturalist(라이선스 확인), FishBase, 국립수산과학원 어류도감, 네이버/구글 이미지 크롤링, 낚시 커뮤니티. 실사용 유사 사진(손에 든/바닥/젖은)을 반드시 섞을 것.
- **라벨링**: 분류 문제라 **폴더명 = 라벨**. 바운딩박스 불필요. 종별 폴더에 넣기만 하면 됨.

---

## 4. 단계별 실행 계획

### Phase 1 — 환경 세팅 (0.5일)
- [ ] 레포 생성, 위 폴더 구조 생성.
- [ ] `requirements.txt` 작성 후 설치:
  ```
  torch torchvision timm albumentations opencv-python pillow
  scikit-learn matplotlib pandas tqdm
  fastapi uvicorn[standard] onnx onnxruntime python-multipart
  ```
- [ ] `src/config.py`에 24종 클래스 리스트(고정 순서) 정의 → 이 순서가 모델 출력 인덱스가 됨.

### Phase 2 — 데이터 수집·정제·분할 (3~4일, 가장 오래 걸림)
- [ ] 종별 이미지 수집 → `data/raw/<종명>/`.
- [ ] 정제: 중복 제거(해시), 물고기 아닌 사진/여러 마리/워터마크 제거, 손상 파일 제거.
- [ ] 클래스 불균형 확인. 부족한 종 추가 수집.
- [ ] **train/val/test = 70/15/15** 로 분할 → `data/splits/`. (분할은 종별로 stratified)
- [ ] `labels.json` 생성 (인덱스→종명).

### Phase 3 — 학습 (2~3일)
- [ ] 백본: `timm`의 `efficientnet_b0`(가벼움) 또는 `convnext_tiny`(정확도) 로 시작.
- [ ] 입력 224×224, ImageNet 정규화.
- [ ] 증강: RandomResizedCrop, HFlip, 밝기/대비/색조 지터, 약한 회전/블러 (albumentations).
- [ ] 손실: CrossEntropy (+ 클래스 불균형 시 class weights 또는 WeightedRandomSampler).
- [ ] 옵티마이저 AdamW, LR 스케줄러(cosine), epochs 20~40, early stopping(val 기준).
- [ ] 2단계 파인튜닝: (1) 백본 freeze + 헤드만 학습 → (2) 전체 unfreeze 낮은 LR.
- [ ] 베스트 체크포인트 `models/best.pt` 저장.
- **급하면 대안**: `yolo classify train model=yolov8n-cls.pt data=data/splits epochs=30 imgsz=224` 한 줄.

### Phase 4 — 평가 (1일)
- [ ] test셋 **Top-1 / Top-3 정확도** 측정 (서비스 기준은 Top-3).
- [ ] **혼동행렬** 출력 → 어떤 종이 어디로 새는지 확인 (붕어↔잉어, 우럭↔볼락, 감성돔↔벵에돔 집중 점검).
- [ ] per-class precision/recall. 바닥인 종은 데이터 추가 or 재수집.
- [ ] 목표선: 흔한 종 Top-1 90%+, 어려운 종은 **Top-3 90%+** 확보하면 MVP 합격.

### Phase 5 — 모델 Export (0.5일)
- [ ] `src/export_onnx.py`로 `best.pt` → `server/model.onnx` 변환 (opset 12+).
- [ ] ONNX Runtime으로 로드해 .pt와 출력 일치하는지 검증(수치 오차 확인).

### Phase 6 — FastAPI 서빙 + Docker (2일)
- [ ] `server/inference.py`: onnxruntime 세션 로드, 전처리(리사이즈·정규화), softmax, **Top-3 추출**.
- [ ] `server/main.py`: `/predict`(이미지 업로드), `/health`.
- [ ] confidence 임계값 로직: 최고 확률 < 임계값이면 `"uncertain": true` 반환 → 앱은 "다시 촬영" 유도.
- [ ] `Dockerfile` 작성 후 컨테이너 빌드·실행.
- [ ] 앱 백엔드와 연동 테스트 (실사용 사진으로 엣지케이스 점검).

---

## 5. FastAPI 엔드포인트 명세

**`POST /predict`** — multipart/form-data, field `file` (이미지)

응답 예시:
```json
{
  "success": true,
  "uncertain": false,
  "predictions": [
    {"rank": 1, "species": "붕어", "confidence": 0.82},
    {"rank": 2, "species": "잉어", "confidence": 0.11},
    {"rank": 3, "species": "향어", "confidence": 0.03}
  ]
}
```

**`GET /health`** → `{"status": "ok"}`

`server/main.py` 스켈레톤:
```python
from fastapi import FastAPI, UploadFile, File
from inference import predict_topk

app = FastAPI(title="Fish Classifier")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    preds = predict_topk(image_bytes, k=3)   # [{"species","confidence"}, ...]
    top1 = preds[0]["confidence"]
    return {
        "success": True,
        "uncertain": top1 < 0.45,            # 임계값 튜닝
        "predictions": [
            {"rank": i + 1, **p} for i, p in enumerate(preds)
        ],
    }
```

실행: `uvicorn server.main:app --host 0.0.0.0 --port 8000`

`Dockerfile` 스켈레톤:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server/ ./server/
EXPOSE 8000
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 6. 2주 타임라인

| 일차 | 작업 |
|---|---|
| D1 | Phase 1 환경 세팅 + 데이터 수집 착수 |
| D2~D4 | Phase 2 데이터 수집·정제·분할 |
| D5~D7 | Phase 3 베이스라인 학습 + 증강 |
| D8~D9 | Phase 4 평가 + 헷갈리는 종 데이터 보강·재학습 |
| D10 | Phase 5 ONNX export + 검증 |
| D11~D12 | Phase 6 FastAPI 서버 + Docker |
| D13~D14 | 앱 연동 테스트, 엣지케이스, confidence 임계값 튜닝, 버퍼 |

---

## 7. 새 Claude 세션 시작용 프롬프트 (복사해서 사용)

```
낚시 앱의 어류 인식 모델을 개발한다. 첨부한 fish_model_plan.md 계획서를 따라 Phase 1부터 시작하자.
- 24종 어류 분류 CNN을 전이학습(PyTorch + timm)으로 만들고 FastAPI 별도 서버로 배포한다.
- 서비스는 사용자가 사진을 찍으면 Top-3 후보를 반환하고 사용자가 확정하는 구조다.
먼저 레포 폴더 구조와 requirements.txt, src/config.py(24종 클래스 리스트)를 만들어줘.
그다음 dataset.py, train.py 순서로 진행하자.
```
> 새 세션에는 이 `fish_model_plan.md`와 `fishing_spots_all.json`을 함께 첨부할 것.

*(→ Phase 1이 완료되었으므로 이 프롬프트는 더 이상 필요 없다. 새 세션에서는 `CLAUDE.md` →
`docs/roadmap.md`의 "현재 위치"를 읽고 바로 이어서 작업하면 된다.)*

---

## 8. 체크리스트

- [ ] Phase 1 환경·레포·config
- [ ] Phase 2 데이터 14k+ 수집·정제·분할
- [ ] Phase 3 학습 (best.pt)
- [ ] Phase 4 평가 (Top-3 90%+ / 혼동행렬 점검)
- [ ] Phase 5 ONNX export·검증
- [ ] Phase 6 FastAPI + Docker 배포
- [ ] 앱 연동 + confidence 임계값 튜닝

## 리스크 & 대응 (요약)
- **데이터가 병목**: 모델보다 수집이 오래 걸림 → D1부터 수집 병행 시작.
- **헷갈리는 5쌍**(붕어↔잉어, 우럭↔볼락, 감성돔↔벵에돔, 고등어↔전갱이↔삼치, 숭어↔농어): 데이터 넉넉히 + Top-3 서빙으로 흡수.
- **도메인 갭**(웹 사진 vs 실사용 사진): 실사용 유사 사진을 학습셋에 반드시 포함.
- **정확도 안 나오는 종**: 삭제는 최후 수단. 데이터 수집 다 해보고 결정.

---

## 부록 A — 데이터 수집 파이프라인 (Phase 2 상세)

> 직접 손으로 수천 장 다운로드 ❌. **크롤링 파이프라인을 실행**해서 모은다.
> 용도: **공모전(비상업)**. 모델 학습 목적이라 소스를 넓게 쓸 수 있으나, **출처·라이선스는 기록**하고 데이터셋 자체를 재배포하지 않는다.

### A.1 소스 우선순위

1. **iNaturalist / GBIF API (1순위, 가장 깨끗)** — 종 라벨·라이선스·지오태그가 붙은 연구용 사진. 라벨 오염 거의 없음. 종당 수백 장 확보 가능.
2. **검색엔진 이미지 크롤링 (2순위, 빈 곳 채우기)** — "○○ 낚시" 한글 키워드로 실사용 유사(손에 든/바닥) 사진 확보. **수집 후 사람 검수 필수**(라벨 오염).
3. **기존 공개 데이터셋 (보조)** — Kaggle 등 물고기 데이터셋 중 겹치는 종만 활용.

### A.2 도구/라이브러리

```
requests            # iNaturalist/GBIF API 호출
icrawler            # 구글/빙 이미지 크롤러
imagehash pillow    # 퍼셉추얼 해시 중복 제거
torch timm          # 사전학습 모델로 '비물고기' 1차 필터
```

### A.3 파이프라인 순서

```
(1) API 다운로드      crawl_inat.py   → data/raw/<종>/inat_*.jpg
(2) 키워드 크롤링     crawl_search.py → data/raw/<종>/web_*.jpg
(3) 중복 제거         dedup.py        (해시로 동일/유사 이미지 삭제)
(4) 비물고기 1차 필터  prefilter.py    (사전학습 모델로 물고기 아닌 것 걸러 후보 축소)
(5) 사람 검수         (썸네일 훑으며 오라벨/요리사진/여러마리 제거)
(6) 분할             split.py        → data/splits/{train,val,test}/<종>/
```

### A.4 스크립트 스켈레톤

**`crawl_inat.py`** — iNaturalist 관측 사진 다운로드 (학명으로 조회, 라이선스 기록)
```python
import requests, os, time
API = "https://api.inaturalist.org/v1/observations"

def download_species(sci_name, out_dir, target=400):
    os.makedirs(out_dir, exist_ok=True)
    page, got = 1, 0
    while got < target:
        r = requests.get(API, params={
            "taxon_name": sci_name, "photos": "true",
            "quality_grade": "research", "per_page": 200, "page": page,
        }).json()
        results = r.get("results", [])
        if not results: break
        for obs in results:
            for p in obs.get("photos", []):
                url = p["url"].replace("square", "large")   # 원본 크기
                lic = p.get("license_code")                 # 라이선스 기록
                img = requests.get(url).content
                open(f"{out_dir}/inat_{obs['id']}_{p['id']}.jpg", "wb").write(img)
                # (lic를 csv 로그로 남길 것)
                got += 1
        page += 1
        time.sleep(1)   # API 예의
```

**`crawl_search.py`** — 키워드 크롤러 (icrawler)
```python
from icrawler.builtin import BingImageCrawler

def crawl(keyword, out_dir, max_num=300):
    c = BingImageCrawler(storage={"root_dir": out_dir})
    c.crawl(keyword=keyword, max_num=max_num)
# 예: crawl("배스 낚시", "data/raw/배스", 300)
```

**`dedup.py`** — 퍼셉추얼 해시 중복 제거
```python
import imagehash, os
from PIL import Image

def dedup(folder):
    seen = set()
    for f in os.listdir(folder):
        try:
            h = str(imagehash.phash(Image.open(os.path.join(folder, f))))
        except Exception:
            os.remove(os.path.join(folder, f)); continue
        if h in seen:
            os.remove(os.path.join(folder, f))
        else:
            seen.add(h)
```

### A.5 종별 조회 정보 (학명 = iNat/GBIF 검색용 · 한글 키워드 = 검색 크롤링용)

| 종 | 학명 (API 조회) | 검색 키워드 |
|---|---|---|
| 감성돔 | Acanthopagrus schlegelii | 감성돔 낚시 |
| 농어 | Lateolabrax japonicus | 농어 낚시 |
| 돌돔 | Oplegnathus fasciatus | 돌돔 낚시 |
| 벵에돔 | Girella punctata | 벵에돔 낚시 |
| 우럭(조피볼락) | Sebastes schlegelii | 우럭 낚시, 조피볼락 |
| 참돔 | Pagrus major | 참돔 낚시 |
| 광어(넙치) | Paralichthys olivaceus | 광어 낚시, 넙치 |
| 볼락 | Sebastes inermis | 볼락 낚시 |
| 갈치 | Trichiurus lepturus | 갈치 낚시 |
| 고등어 | Scomber japonicus | 고등어 낚시 |
| 삼치 | Scomberomorus niphonius | 삼치 낚시 |
| 방어 | Seriola quinqueradiata | 방어 낚시 |
| 전갱이 | Trachurus japonicus | 전갱이 낚시 |
| 숭어 | Mugil cephalus | 숭어 낚시 |
| 붕어 | Carassius carassius | 붕어 낚시 |
| 잉어 | Cyprinus carpio | 잉어 낚시 |
| 쏘가리 | Siniperca scherzeri | 쏘가리 낚시 |
| 배스(큰입배스) | Micropterus salmoides | 배스 낚시, 배스 조황 |
| 블루길 | Lepomis macrochirus | 블루길 |
| 가물치 | Channa argus | 가물치 낚시 |
| 메기 | Silurus asotus | 메기 낚시 |
| 송어(무지개송어) | Oncorhynchus mykiss | 송어 낚시, 무지개송어 |
| 피라미 | Zacco platypus | 피라미 |
| 동자개 | Tachysurus fulvidraco | 동자개, 빠가사리 |

> 이 표는 `src/config.py`의 `SPECIES`에 그대로 옮겨져 있다. **스크립트는 config에서 읽어 쓸 것** — 표를 두 군데 관리하지 않는다.

### A.6 라이선스 메모 (공모전=비상업)

- 비상업/연구·학습 목적이므로 iNat의 CC-BY-NC 등도 사용 가능. 단, **각 이미지의 `license_code`·출처를 csv에 기록**해 둘 것(공모전 제출 시 데이터 출처 근거로 유용).
- **크롤링 원본 이미지 자체를 재배포·공개하지 말 것.** 학습된 모델만 배포.
- 나중에 상업 전환 시 → CC0/CC-BY만 남기고 재필터링 필요.

### A.7 검수 속도 팁

- 폴더별 썸네일 뷰(윈도우 탐색기 '큰 아이콘')로 훑으며 요리사진·여러마리·오라벨만 빠르게 삭제. 사람이 시간당 ~1,000장 정도 검수 가능.
- 헷갈리는 쌍(우럭/볼락, 붕어/잉어)은 **iNat research-grade 위주**로 채우면 검수 부담이 준다(이미 전문가 검증됨).
- MVP는 종당 200~300장으로 먼저 돌리고, 정확도 낮은 종만 추가 수집(전량 완벽 수집 후 학습 ❌ → 반복 개선 ⭕).
