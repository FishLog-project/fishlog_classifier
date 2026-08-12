# fishilog-ai — 어류 분류 모델 (24종 + 기타)

낚시 앱용 어종 인식 모델. 사진 1장 → **Top-3 후보 + confidence** 반환 → 사용자가 확정.
전이학습(PyTorch + timm)으로 학습하고, ONNX로 변환해 FastAPI 서버로 서빙한다.
출력은 25클래스 — 어종 24 + 비물고기·24종 밖을 흡수하는 `기타`.

## 문서

작업 전에 [CLAUDE.md](CLAUDE.md)에서 하려는 일에 맞는 문서로 이동한다.

- 진행 상황·다음 할 일 → [docs/roadmap.md](docs/roadmap.md)
- 정해야 할 것·논의 필요 → [docs/decisions.md](docs/decisions.md)
- 데이터 / 학습 / 평가 / 서빙 / 연동 / 규약 → `docs/`

## 세팅

> **Python 3.11 / 3.12 권장.** 3.14는 torch·albumentations 휠이 아직 없다.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# CUDA GPU가 있다면 torch를 CUDA 빌드로 재설치 (드라이버 525 이상 필요)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 폴더 구조

```
fishilog_ai/
├── data/
│   ├── raw/<종명>/      # 수집 원본 (24개 폴더)
│   ├── clean/<종명>/    # 중복·비물고기 제거 후
│   └── splits/{train,val,test}/<종명>/   # 70/15/15 stratified
├── src/
│   ├── config.py        # ★ 클래스 정의 단일 진실 소스 (순서 = 모델 출력 인덱스)
│   ├── dataset.py       # Dataset/DataLoader + albumentations 증강
│   ├── train.py         # 2단계 파인튜닝 학습 루프
│   ├── evaluate.py      # (Phase 4) Top-1/Top-3, 혼동행렬
│   └── export_onnx.py   # (Phase 5) best.pt → model.onnx
├── scripts/             # (Phase 2) 크롤링·정제·분할 스크립트
├── server/              # (Phase 6) FastAPI 추론 서버
├── models/              # best.pt / last.pt / history.csv
└── reports/             # 혼동행렬·증강 미리보기 등 산출물
```

## 자주 쓰는 명령

```bash
python -m src.config --summary     # 24종 목록 / 수집 목표 / 현재 수집량
python -m src.config --init        # 폴더 구조 + server/labels.json 재생성

python -m src.dataset --check      # split별 클래스 분포 점검
python -m src.dataset --preview    # 증강 결과를 reports/aug_preview.jpg 로 저장
python -m src.dataset --batch      # 배치 1개 로드 테스트

python -m src.train --smoke        # 파이프라인 점검 (배치 3개 × 2 epoch)
python -m src.train                # 기본 학습 (efficientnet_b0, 30 epoch)
python -m src.train --backbone convnext_tiny --batch-size 16
```

## 주의사항

- **`src/config.py`의 `SPECIES` 순서를 절대 바꾸지 말 것.** 삽입 순서가 그대로 모델
  출력 인덱스이고, 체크포인트·`labels.json`·ONNX가 모두 이 순서에 묶여 있다.
  `기타`는 항상 마지막 인덱스이며, 종 추가는 그 바로 앞에 넣고 재학습한다.
- 수집한 원본 이미지는 **커밋·재배포 금지** (`.gitignore`로 차단). 공모전 제출물은
  학습된 모델만. 이미지별 출처·라이선스는 CSV로 기록해 둘 것.
- 헷갈리는 그룹(붕어↔잉어, 우럭↔볼락, 감성돔↔벵에돔, 고등어↔전갱이↔삼치,
  숭어↔농어)은 데이터를 넉넉히 모으고 Top-3 서빙으로 흡수한다.
