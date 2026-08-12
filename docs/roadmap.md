# 로드맵 — 지금 뭘 할 차례인가

> 이 파일이 **진행 상황의 단일 기준**이다. 작업을 끝낼 때마다 체크박스와 "현재 위치"를 갱신한다.

## 현재 위치 (2026-08-12 기준)

**Phase 1 완료 → Phase 2(데이터 수집) 착수 대기.**

- 있는 것: `src/config.py`(25클래스 확정), `src/dataset.py`, `src/train.py`, 폴더 구조, `server/labels.json`, `docs/`
- 없는 것: `scripts/*`(수집 4종), `src/evaluate.py`, `src/export_onnx.py`, `server/main.py`·`inference.py`, `Dockerfile`
- 확정: 25클래스(어종 24 + `기타`) / MVP 종당 250장 / 학습은 Colab T4 ([decisions.md](decisions.md) A-10~12)
- 미결정: 공모전 마감일, 배포처, 24종 밖 어종 앱 UX ([decisions.md](decisions.md) B섹션)

### ▶ 다음에 할 일

1. `scripts/crawl_inat.py` 작성 — `config.SPECIES`의 학명으로 iNaturalist 수집 + `data/licenses.csv` 기록
   (`기타`는 `scientific == ""` 이므로 건너뛴다)
2. `scripts/crawl_search.py` — 한글 키워드 크롤링 (`기타` 클래스 포함)
3. `scripts/dedup.py` → `scripts/split.py`
4. 병행: Python 3.11 venv + `pip install -r requirements.txt` → `python -m src.train --smoke`

상세 → [data-pipeline.md](data-pipeline.md)

---

## Phase 체크리스트

### Phase 1 — 환경·레포·config ✅
- [x] 폴더 구조 생성 (`data/{raw,clean,splits}/<24종>`, `src/`, `server/`, `models/`, `reports/`, `scripts/`)
- [x] `requirements.txt` / `server/requirements.txt`(슬림 배포용)
- [x] `src/config.py` — 24종 고정 순서 + 학명·키워드·난이도·목표량
- [x] `src/dataset.py` — Dataset/증강/불균형 샘플러/점검 CLI
- [x] `src/train.py` — 2단계 파인튜닝 루프
- [x] `README.md`, `.gitignore`, `CLAUDE.md`, `docs/`
- [ ] **Python 3.11 venv + 의존성 설치** ← 다음 액션
- [ ] `python -m src.train --smoke` 로 파이프라인 검증 (더미 이미지 몇 장으로)

### Phase 2 — 데이터 수집·정제·분할 ⬜ (병목, MVP 1~2일)
- [ ] `scripts/crawl_inat.py` — iNaturalist API 수집 + 라이선스 CSV 기록
- [ ] `scripts/crawl_search.py` — Bing/Google 키워드 크롤링(실사용 유사 사진)
- [ ] `기타` 클래스 수집 — 회 접시·어시장·낚시터 풍경·사람·24종 밖 어종(향어·학꽁치 등)
- [ ] `scripts/dedup.py` — phash 중복 제거
- [ ] `scripts/prefilter.py` — 사전학습 모델로 비물고기 1차 필터
- [ ] `scripts/split.py` — 70/15/15 stratified 분할
- [ ] 사람 검수 (요리사진·여러마리·오라벨 제거)
- [ ] `python -m src.dataset --check` 로 종별 분포 확인, 부족한 종 추가 수집
- 상세 → [data-pipeline.md](data-pipeline.md)

### Phase 3 — 학습 ⬜ (2~3일)
- [ ] 베이스라인: `efficientnet_b0`, 224px, 30 epoch
- [ ] `python -m src.dataset --preview` 로 증강 결과 눈으로 확인
- [ ] 학습 곡선 확인(`models/history.csv`), 과적합/과소적합 대응
- [ ] (선택) `convnext_tiny` 비교
- [ ] `models/best.pt` 확보
- 상세 → [training.md](training.md)

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
