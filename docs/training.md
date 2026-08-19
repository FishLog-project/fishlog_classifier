# Phase 3 — 학습

구현: `src/train.py` (루프) · `src/dataset.py` (증강) · `src/config.py::TrainConfig` (기본값)

## 전략: 2단계 파인튜닝

| 단계 | epoch | 학습 대상 | LR | 비고 |
|---|---|---|---|---|
| Stage 1 | 0 ~ 2 (`freeze_epochs=3`) | 분류 헤드만 | `head_lr=1e-3` | 백본 freeze + **BatchNorm 통계도 동결** |
| Stage 2 | 3 ~ 29 | 전체 | `lr=3e-4` | 워밍업 1 epoch → 코사인 감쇠 |

Stage 1을 건너뛰고 바로 전체 학습하면, 랜덤 초기화된 헤드의 큰 그래디언트가 사전학습 백본을 망가뜨린다(catastrophic forgetting).

## 기본 하이퍼파라미터

```
backbone      efficientnet_b0   # 대안: convnext_tiny
img_size      224               # ImageNet 정규화
batch_size    32                # VRAM 2GB면 16 이하
epochs        30 (freeze 3)
optimizer     AdamW, weight_decay 1e-4 (bias/norm 제외)
loss          CrossEntropy, label_smoothing 0.1
scheduler     cosine + warmup 1 epoch, min_lr 1e-6
grad_clip     1.0
불균형 대응    WeightedRandomSampler (--class-weights 로 손실 가중치 방식 전환 가능)
early stop    patience 8, monitor = val_top3
AMP           CUDA일 때 자동 on
```

**monitor가 `top3`인 이유**: 서비스가 Top-3 후보를 보여주므로 실제 합격 기준과 일치한다. Top-1로 모니터링하면 혼동 쌍을 억지로 가르는 방향으로 최적화된다.

## 증강 (albumentations 2.x)

```
RandomResizedCrop(scale 0.55~1.0) → HFlip → Affine(회전 ±15°, 스케일 ±10%, 이동 ±5%)
→ 밝기/대비 ±25% → 색조/채도/명도 지터
→ OneOf[MotionBlur, GaussianBlur, GaussNoise] p=0.25
→ ImageCompression(품질 55~100) p=0.3
→ Normalize → ToTensor
```

블러·JPEG 압축·색조 지터는 **웹 사진 ↔ 폰카 실사용 사진의 도메인 갭**을 메우기 위한 것이다.
`python -m src.dataset --preview` 로 항상 눈으로 먼저 확인할 것 — 증강이 과하면 물고기 특징(줄무늬·색)이 뭉개진다.

⚠️ **수직 뒤집기(VerticalFlip)는 쓰지 않는다.** 물고기는 등/배 방향이 종 판별의 핵심 단서다.

## 실행

```bash
python -m src.train --smoke                  # 배치 3개 × 2 epoch, 파이프라인 점검
python -m src.train                          # 기본
python -m src.train --backbone convnext_tiny --batch-size 16
python -m src.train --resume models/last.pt  # 이어서
python -m src.train --class-weights          # 샘플러 대신 손실 가중치
```

산출물: `models/best.pt`, `models/last.pt`, `models/history.csv`

체크포인트에는 `classes`·`img_size`·`mean/std`·`backbone`이 함께 저장된다 → 평가·ONNX 변환이 설정을 추측할 필요가 없다.

## 학습 곡선 읽기 (`models/history.csv`)

| 증상 | 해석 | 대응 |
|---|---|---|
| train top1 ↑, val top1 정체/하락 | 과적합 | 증강 강화, `drop_rate` ↑, 데이터 추가, epoch 축소 |
| train/val 둘 다 낮음 | 과소적합 | Stage 2 LR ↑, epoch ↑, 더 큰 백본 |
| val이 심하게 출렁임 | batch 너무 작음 / LR 과다 | batch ↑, LR ↓ |
| val_top1은 낮은데 val_top3는 높음 | 혼동 쌍 문제 | **정상**. 서비스 기준으론 합격. 해당 종 데이터 보강 |
| Stage 2 시작 직후 급락 | LR 과다 | `--lr 1e-4` |

## 튜닝 순서 (효과 큰 것부터)

1. **데이터 추가·정제** — 다른 무엇보다 효과가 크다
2. 증강 강도 조절
3. 백본 교체 (`efficientnet_b0` → `convnext_tiny` / `efficientnet_b2`)
4. 입력 해상도 224 → 288 (혼동 쌍의 미세한 무늬 구분에 유효, 속도는 손해)
5. LR·epoch 미세 조정

## 급할 때 대안

```bash
yolo classify train model=yolov8n-cls.pt data=data/splits epochs=30 imgsz=224
```
데이터 폴더 구조가 이미 호환된다. 단 ONNX export·서빙 코드는 별도 경로가 되므로, 베이스라인 확인용으로만 쓰고 본선은 timm 경로로 간다.

## 정규화 (mixup / cutmix) — 2026-08-18 추가

1차 학습에서 과적합이 확인됐다: `train_top1 0.9895` vs `val_top1 0.7538` (간격 24pp).
4,904장으로 404만 파라미터를 학습하니 일반화보다 암기가 쉬웠다.

mixup(두 사진의 픽셀 가중합)과 cutmix(사각형 패치 교체)는 라벨까지 함께 섞어
"이 사진 = 이 정답"이라는 암기 경로를 물리적으로 막는다. 기본값으로 켜져 있다:

| 인자 | 기본값 |
|---|---|
| `--mixup-alpha` | 0.2 |
| `--cutmix-alpha` | 1.0 |
| `--mixup-prob` | 0.5 (배치 단위 적용 확률) |
| `--no-mixup` | 완전히 끄기 (이전 동작 재현) |

**읽을 때 주의할 점 2가지:**

1. **`train_top1`이 이전보다 낮게 나오는 게 정상이다.** 손실은 섞인 soft 타깃으로
   계산하지만 정확도는 원본 라벨로 재기 때문이다. 봐야 할 것은 절대값이 아니라
   **train과 val의 간격**이 좁아졌는지다.
2. **`train_loss`를 이전 실험과 직접 비교하면 안 된다.** SoftTargetCrossEntropy와
   CrossEntropy는 스케일이 다르다. `val_loss`/`val_top3`만 비교 대상이다.

`--class-weights` 와는 함께 쓸 수 없다(SoftTargetCrossEntropy가 클래스 가중치를
받지 못한다). 동시에 주면 mixup이 꺼지고 경고가 나온다.

## 사진 품질 필터 — 2026-08-18 추가

`scripts/quality_audit.py` 로 test셋에서 검증한 결과, **품질 점수(fish_prob)와
정확도의 상관이 뚜렷했다**:

| fish_prob 10분위 | Top-3 |
|---|---|
| 1 (0.001~0.052) | 76.9% |
| 2 (0.052~0.288) | 77.9% |
| 5 (0.644~0.711) | 93.3% |
| 10 (0.906~0.996) | 96.4% |

출처별로도 iNat 86.5% / web 90.6% 로, 품질 필터를 안 돌린 iNat 쪽이 낮다.

### 쓰는 법

```bash
python -m scripts.score_quality              # 1회. → data/quality_scores.csv
python -m src.train --min-fish-prob 0.30     # 임계값만 바꿔가며 실험
```

점수는 CSV에 캐시되므로 **임계값을 바꿔도 재채점·재분할이 필요 없다.**
(재분할하면 train/test 경계가 바뀌어 이전 실험과 비교가 불가능해진다.)

### ⚠️ 학습셋에만 적용된다

`build_dataloader` 가 `split == "train"` 일 때만 필터를 넘긴다. 이건 의도된 것이다.

**val/test에 필터를 걸면 안 된다.** 어려운 문제를 빼고 채점하는 셈이라 모델은
그대로인데 점수만 올라간다. `quality_audit` 의 "임계값 후보" 표에 나오는
`남은 것 Top-3 90.2%` 같은 숫자는 **필터 후 성능 예측이 아니라** '쉬운 문제만
골라 채점한 값'이므로 목표 달성 근거로 쓰면 안 된다.

시험셋에서도 이런 사진을 빼는 것은 별개의 판단이다 — "우리 test셋이 실사용
입력을 대표하는가"라는 제품 정의 문제이며, 바꾸려면 그 근거를 따로 남길 것.
