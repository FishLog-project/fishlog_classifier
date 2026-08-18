# 로드맵 — 지금 뭘 할 차례인가

> 이 파일이 **진행 상황의 단일 기준**이다. 작업을 끝낼 때마다 체크박스와 "현재 위치"를 갱신한다.

## 현재 위치 (2026-08-18 기준)

**Phase 2 완료 → Phase 3(학습) 착수.** 데이터 6,999장 분할 완료 (train 4,904 / val 1,048 / test 1,047).

- 있는 것: `src/config.py`(25클래스), `src/dataset.py`, `src/train.py`,
  `scripts/*` 8종(수집 3 + 정제 3 + 검수 보조 2), **분할된 데이터 6,999장**, `server/labels.json`
- 없는 것: `src/evaluate.py`, `src/export_onnx.py`, `server/main.py`·`inference.py`, `Dockerfile`
- 확정: 25클래스(어종 24 + `기타`) / MVP 종당 250장 / 학습은 Colab T4 ([decisions.md](decisions.md) A-10~12)
- 미결정: 공모전 마감일, 배포처, 24종 밖 어종 앱 UX ([decisions.md](decisions.md) B섹션)

### 데이터 현황 (2026-08-18 실측)

| | 장수 |
|---|---|
| train / val / test | 4,904 / 1,048 / 1,047 (합 **6,999**) |
| 소스 | iNaturalist 약 4,900 + 검색(DDG·Bing) 약 2,100 |
| 수집 원본 `data/raw` | 9,005 → 정제 후 `data/clean` 6,999 |
| 격리분 | 1,901장 (자동 필터 `data/reject` 1,221 + 사람 검수 `data/review` 680) + 중복 제거 약 105 |

**목표 미달 3종**: 볼락 146, 삼치 151, 전갱이 189 (다음으로 적은 종은 동자개 216).
iNat 관측 수 자체가 적고(볼락 39건), 검색 결과는 어시장·조리 사진이 대부분이라 검수에서 60~80% 탈락했다.
→ A-11 방침대로 **이대로 1차 학습**하고, 평가에서 이 3종의 recall을 보고 보강 여부를 정한다.

### ▶ 다음에 할 일 (Phase 3)

Phase 3 사전 준비는 2026-08-18에 끝났다:
- [x] `python -m src.dataset --check` — 4,904 / 1,048 / 1,047 확인
- [x] `python -m src.dataset --preview` — `reports/aug_preview.jpg` 육안 확인 (증강 정상)
- [x] `python -m scripts.package_data` — `data/splits.zip` (1,067 MB, 한글 경로 UTF-8 검증 통과)
- [x] `python -m src.train --smoke` — 2단계 파인튜닝·체크포인트 저장까지 통과 (스모크 산출물은 삭제함)

1차 학습도 끝났다 (val_top3 0.8721, 아래 "베이스라인 결과" 참조).

### ▶ 다음에 할 일 (Phase 4 — 진단)

베이스라인이 합격선에 -2.8pp 미달이고 **과적합이 확실**하다. 튜닝을 더 하기 전에
"어디서 틀리는가"를 먼저 봐야 한다 ([evaluation.md](evaluation.md) "목표 미달 시 대응 순서" 1번).

1. `src/evaluate.py` 작성 → test 1,047장으로 산출물 5종 ([evaluation.md](evaluation.md) "산출물")
2. **종별 recall** — 볼락·삼치·전갱이·동자개(데이터 부족 4종)가 무너졌는지 확인
3. **혼동행렬** — 6그룹이 *대칭* 혼동이면 데이터 부족, *한쪽 쏠림*이면 라벨 오염
4. **`worst_cases/`** — 확신했는데 틀린 50장. 어시장·조리 사진이 나오면 라벨 오염 확진
5. 진단 결과에 따라 분기:
   - 라벨 오염 → 검수 2차 후 재학습
   - 데이터 부족 → 해당 종 보강 수집
   - 둘 다 아니면 → 정규화 강화(mixup/cutmix) 또는 `convnext_tiny`

상세 → [training.md](training.md) / [data-pipeline.md](data-pipeline.md)

---

## Phase 체크리스트

### Phase 1 — 환경·레포·config ✅ (완료)
- [x] 폴더 구조 생성 (`data/{raw,clean,splits}/<24종>`, `src/`, `server/`, `models/`, `reports/`, `scripts/`)
- [x] `requirements.txt` / `server/requirements.txt`(슬림 배포용)
- [x] `src/config.py` — 24종 고정 순서 + 학명·키워드·난이도·목표량
- [x] `src/dataset.py` — Dataset/증강/불균형 샘플러/점검 CLI
- [x] `src/train.py` — 2단계 파인튜닝 루프
- [x] `README.md`, `.gitignore`, `CLAUDE.md`, `docs/`
- [x] **Python 3.11 venv + 의존성 설치** (3.11.9 / torch 2.13.0+cpu / timm 1.0.28 / albumentations 2.0.8)
      `stringzilla==5.1.1` 핀 필요 — 5.1.2는 Windows cp311 휠이 없어 MSVC 빌드를 요구한다
- [x] `python -m src.train --smoke` 로 파이프라인 검증 통과 (2단계 파인튜닝·체크포인트·한글 경로 OK)

### Phase 2 — 데이터 수집·정제·분할 ✅ (완료, 2026-08-14)
- [x] `scripts/crawl_inat.py` — iNaturalist API (25클래스 taxon 매칭 확인, 학명 3건 정정)
- [x] `scripts/crawl_ddg.py` — **DuckDuckGo 검색 (실사용 사진 주 소스)**
- [x] `scripts/crawl_search.py` — Bing 크롤링 (보조. 키워드당 20~60장이면 고갈)
- [x] `scripts/crawl_naver.py` — 네이버 API (미사용: 신규 앱에 '검색' API 권한이 없음)
- [x] `scripts/dedup.py` — phash 중복 제거 + 거부 목록 영구화
- [x] `scripts/prefilter.py` — ImageNet 어류 확률 필터 + `make_scorer`(크롤링 중 실시간 판정)
- [x] `scripts/review.py` — 검수 대상만 모아 보여주고 삭제 결과 확정
- [x] `scripts/refsheet.py` — 혼동 쌍 비교 참고표 (검수 보조)
- [x] `scripts/split.py` — 70/15/15 + obsid/phash 그룹 누수 방지
- [x] 사람 검수 — 혼동 쌍 web 사진 1,370장 검토 → 680장 격리(약 50%)
- [x] `기타` 클래스 250장 (4버킷 배분)
- 상세 → [data-pipeline.md](data-pipeline.md)

### Phase 3 — 학습 🔶 (베이스라인 확보, 2026-08-18)
- [x] `python -m src.dataset --preview` 로 증강 결과 눈으로 확인
- [x] 베이스라인: `efficientnet_b0`, 224px, T4에서 **16.9분** / 26 epoch에서 조기종료
- [x] `models/best.pt` 확보 (16.4MB, Drive `MyDrive/fishlog/models/`)
- [x] 학습 곡선 확인 → **과적합 확진**
- [ ] (선택) `convnext_tiny` 비교 — Phase 4 진단 후 판단
- 상세 → [training.md](training.md)

#### 베이스라인 결과 (val 1,048장)

| 지표 | 결과 | 합격 기준 | 판정 |
|---|---|---|---|
| val Top-3 | **0.8721** (E18) | 0.90 | ❌ -2.8pp |
| val Top-1 | 0.7538 (E25) | 0.75 | ✅ |

**과적합.** train_top1 0.9895 vs val_top1 0.7538 → 간격 24pp.
epoch 10 이후 train loss만 계속 떨어지고 val loss는 1.42에서 평평 → 조기종료(patience 8) 정당.
**더 오래 돌려도 오르지 않는다.** epoch/LR 튜닝이 아니라 데이터 쪽 문제다.

원인 후보 3가지는 현재 데이터로 구분 불가 → Phase 4 진단이 먼저다:
1. 데이터 부족 (종당 250장은 25클래스 구분에 빠듯)
2. **라벨 오염** (웹 크롤링분에 어시장·조리 사진 혼입)
3. 혼동 쌍 6그룹의 본질적 난이도

### Phase 4 — 평가 ⬜ (1일)
- [ ] `src/evaluate.py` 작성 (미작성)
- [ ] test셋 Top-1 / Top-3, per-class precision·recall
- [ ] 혼동행렬 → 혼동 그룹 6쌍 집중 점검
- [ ] 목표 미달 종 데이터 보강 후 재학습
- 상세 → [evaluation.md](evaluation.md)

### Phase 5 — ONNX Export ⬜ (0.5일)
- [ ] `src/export_onnx.py` 작성 (미작성)
- [ ] `.pt` vs `.onnx` 출력 수치 일치 검증 (max abs diff < 1e-4)
- 상세 → [serving.md](serving.md)

### Phase 6 — FastAPI + Docker ⬜ (2일)
- [ ] `server/inference.py`, `server/main.py`
- [ ] `uncertain` 임계값 튜닝
- [ ] `Dockerfile` 빌드·실행
- [ ] 배포 (호스팅 미정 → decisions.md B-2)
- 상세 → [serving.md](serving.md)

### Phase 7 — 앱 연동 ⬜
- [ ] 백엔드 ↔ 모델 서버 계약 확정, 엣지케이스 테스트
- [ ] 사용자 확정 결과 로깅 → 재학습 데이터 축적
- 상세 → [integration.md](integration.md)

---

## 2주 타임라인 (계획서 기준)

| 일차 | 작업 |
|---|---|
| D1 | Phase 1 + 수집 착수 |
| D2~D4 | 수집·정제·분할 |
| D5~D7 | 베이스라인 학습 |
| D8~D9 | 평가 + 보강 재학습 |
| D10 | ONNX export |
| D11~D12 | FastAPI + Docker |
| D13~D14 | 앱 연동, 임계값 튜닝, 버퍼 |

## 리스크

| 리스크 | 대응 |
|---|---|
| 수집이 가장 오래 걸림 | MVP 종당 250장으로 먼저 학습, 부족한 종만 보강 |
| 혼동 쌍 6그룹 | 데이터 넉넉히 + Top-3 서빙으로 흡수 |
| 웹 사진 vs 실사용 사진 도메인 갭 | 손에 든/바닥/젖은 사진 반드시 학습셋에 포함 |
| 24종 밖 어종·비물고기 입력 | `기타` 클래스로 흡수 (A-10). 데이터 다양성이 관건 |
| 로컬 GPU 부족(MX450 2GB) | Colab T4로 학습, 로컬은 코드 검증용 |
| Colab 세션 끊김 | `models/`를 Drive에 링크, `--resume models/last.pt` |
