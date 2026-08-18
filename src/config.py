"""프로젝트 전역 설정.

이 파일이 24종 정의의 **단일 진실 소스(single source of truth)** 다.
`SPECIES` 딕셔너리의 **삽입 순서 = 모델 출력 인덱스 순서**이며,
한 번 학습을 시작한 뒤에는 절대 순서를 바꾸지 말 것.
(순서를 바꾸면 기존 체크포인트/ONNX/labels.json 의 라벨이 전부 어긋난다.)

사용 예:
    python -m src.config --init      # 폴더 구조 생성 + server/labels.json 갱신
    python -m src.config --summary   # 클래스/수집 목표 요약 출력
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# 콘솔 인코딩
# ---------------------------------------------------------------------------
def _force_utf8_console() -> None:
    """Windows 기본 콘솔(cp949)에서 UnicodeEncodeError로 죽는 것을 막는다.

    이 프로젝트의 로그는 한글 + `—`, `≥`, `→` 같은 기호를 쓴다. cp949는 한글은
    되지만 이런 기호에서 터진다. 학습을 30 epoch 돌린 뒤 마지막 요약 print에서
    죽는 사고가 실제로 가능하므로, 모든 진입점이 import하는 이 모듈에서 한 번에 막는다.
    (`errors="replace"` — 표시가 깨질지언정 프로세스가 죽지는 않게)
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and getattr(stream, "encoding", "").lower() != "utf-8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # 파이프/리다이렉트 등 reconfigure 불가 환경 — 무시


_force_utf8_console()


# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # 수집 원본 (종별 폴더)
CLEAN_DIR = DATA_DIR / "clean"      # 정제 후 (중복/비물고기 제거)
SPLITS_DIR = DATA_DIR / "splits"    # train/val/test 분할 결과
TRAIN_DIR = SPLITS_DIR / "train"
VAL_DIR = SPLITS_DIR / "val"
TEST_DIR = SPLITS_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"          # 체크포인트(.pt)
SERVER_DIR = PROJECT_ROOT / "server"
REPORTS_DIR = PROJECT_ROOT / "reports"        # 혼동행렬/지표 산출물

BEST_CKPT = MODELS_DIR / "best.pt"
LAST_CKPT = MODELS_DIR / "last.pt"
HISTORY_CSV = MODELS_DIR / "history.csv"
LABELS_JSON = SERVER_DIR / "labels.json"
ONNX_PATH = SERVER_DIR / "model.onnx"

SPLIT_DIRS = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ---------------------------------------------------------------------------
# 어종 메타데이터
# ---------------------------------------------------------------------------
DIFFICULTY_TARGET = {"easy": 350, "medium": 600, "hard": 1000}  # 종당 권장 이미지 수


MVP_TARGET_IMAGES = 250  # MVP 1차 수집 목표(종당). 평가 후 부족한 종만 목표량까지 보강한다.


@dataclass(frozen=True)
class Species:
    """한 어종에 대한 수집/분류 메타데이터."""

    korean: str                       # 표시용 한글명 (폴더명과 동일)
    scientific: str                   # iNaturalist/GBIF API 조회용 학명 ("" = API 조회 대상 아님)
    difficulty: str                   # easy | medium | hard (시각적 구분 난이도)
    habitat: str                      # sea | fresh | other
    keywords: tuple[str, ...] = ()    # 검색엔진 크롤링 키워드
    confusable_with: tuple[str, ...] = ()  # 헷갈리는 상대 종
    aliases: tuple[str, ...] = ()     # 이명/속명
    target: int | None = None         # 수집 목표 직접 지정(없으면 난이도 기본값)

    @property
    def target_images(self) -> int:
        return self.target if self.target is not None else DIFFICULTY_TARGET[self.difficulty]

    @property
    def is_fish(self) -> bool:
        return self.habitat != "other"


def _s(korean, scientific, difficulty, habitat, keywords=(), confusable=(), aliases=(),
       target=None):
    return Species(korean, scientific, difficulty, habitat, tuple(keywords),
                   tuple(confusable), tuple(aliases), target)


# ⚠️ 순서 고정: 이 순서가 곧 모델 출력 인덱스(0~23). 바다 14종 → 담수 10종.
SPECIES: dict[str, Species] = {
    s.korean: s
    for s in [
        # ---------------- 바다 (14) ----------------
        _s("감성돔", "Acanthopagrus schlegelii", "hard", "sea",
           ["감성돔 낚시", "감성돔 조황", "감성돔 찌낚시", "감성돔 갯바위"], ["벵에돔"], ["구로다이"]),
        _s("농어", "Lateolabrax japonicus", "medium", "sea",
           ["농어 낚시", "농어 루어", "농어 조황", "농어 웨이딩"], ["숭어"], []),
        _s("돌돔", "Oplegnathus fasciatus", "easy", "sea",
           ["돌돔 낚시", "돌돔 조황", "돌돔 선상낚시"], [], ["줄돔"]),
        _s("벵에돔", "Girella punctata", "medium", "sea",
           ["벵에돔 낚시", "벵에돔 조황", "긴꼬리벵에돔", "벵에돔 갯바위"], ["감성돔"], []),
        _s("우럭", "Sebastes schlegelii", "hard", "sea",
           ["우럭 낚시", "조피볼락", "우럭 조황", "우럭 선상낚시",
            "우럭 배낚시", "서해 우럭", "우럭 대물"], ["볼락"], ["조피볼락"]),
        _s("참돔", "Pagrus major", "easy", "sea",
           ["참돔 낚시", "참돔 타이라바", "참돔 조황", "참돔 선상낚시"], [], []),
        _s("광어", "Paralichthys olivaceus", "easy", "sea",
           ["광어 낚시", "넙치", "광어 다운샷", "광어 조황", "넙치 낚시",
            "광어 배낚시", "서해 광어", "광어 대물"], [], ["넙치"]),
        _s("볼락", "Sebastes inermis", "hard", "sea",
           ["볼락 낚시", "볼락 루어", "볼락 조황", "볼락 다운샷", "볼락 웜",
            "볼락 밤낚시", "통영 볼락", "볼락 대물"], ["우럭"], ["뽈락"]),
        _s("갈치", "Trichiurus lepturus", "easy", "sea",
           ["갈치 낚시", "갈치 조황", "갈치 선상낚시"], [], []),
        _s("고등어", "Scomber japonicus", "medium", "sea",
           ["고등어 낚시", "고등어 조황"], ["전갱이", "삼치"], []),
        _s("삼치", "Scomberomorus niphonius", "medium", "sea",
           ["삼치 낚시", "삼치 루어", "삼치 조황", "삼치 캐스팅",
            "삼치 지깅", "가을 삼치", "삼치 대물"], ["고등어"], []),
        _s("방어", "Seriola quinqueradiata", "easy", "sea",
           ["방어 낚시", "방어 조황", "부시리 낚시"], [], ["부시리"]),
        _s("전갱이", "Trachurus japonicus", "medium", "sea",
           ["전갱이 낚시", "아지", "전갱이 조황", "아지 지깅", "전갱이 선상",
            "전갱이 사비키", "전갱이 카고", "전갱이 대물"], ["고등어"], ["아지"]),
        _s("숭어", "Mugil cephalus", "medium", "sea",
           ["숭어 낚시", "숭어 조황"], ["농어"], []),
        # ---------------- 담수 (10) ----------------
        _s("붕어", "Carassius carassius", "hard", "fresh",
           ["붕어 낚시", "붕어 조황"], ["잉어"], []),
        _s("잉어", "Cyprinus carpio", "hard", "fresh",
           ["잉어 낚시"], ["붕어"], ["향어"]),
        _s("쏘가리", "Siniperca scherzeri", "easy", "fresh",
           ["쏘가리 낚시", "쏘가리 루어", "쏘가리 조황", "황쏘가리", "쏘가리 웨이딩",
            "임진강 쏘가리", "쏘가리 대물", "쏘가리 밤낚시"], [], []),
        _s("배스", "Micropterus salmoides", "easy", "fresh",
           ["배스 낚시", "배스 조황"], [], ["큰입배스", "큰입우럭"]),
        _s("블루길", "Lepomis macrochirus", "easy", "fresh",
           ["블루길", "블루길 낚시", "블루길 조과", "블루길 대물"], [], ["파랑볼우럭"]),
        _s("가물치", "Channa argus", "easy", "fresh",
           ["가물치 낚시"], [], []),
        _s("메기", "Silurus asotus", "easy", "fresh",
           ["메기 낚시"], ["동자개"], []),
        _s("송어", "Oncorhynchus mykiss", "easy", "fresh",
           ["송어 낚시", "무지개송어"], [], ["무지개송어"]),
        _s("피라미", "Zacco platypus", "easy", "fresh",
           ["피라미", "피라미 낚시"], [], []),
        # iNat 활성 학명은 Tachysurus sinensis (common name "Korean Bullhead").
        # Tachysurus/Pelteobagrus fulvidraco 는 동의어로 취급되어 검색이 안 잡힌다.
        _s("동자개", "Tachysurus sinensis", "easy", "fresh",
           ["동자개", "빠가사리", "빠가사리 낚시", "동자개 낚시",
            "빠가사리 조과", "빠가사리 대물", "동자개 물고기"], ["메기"],
           ["빠가사리", "Tachysurus fulvidraco"]),
        # ---------------- OOD 처리용 (24종 아님, 항상 맨 뒤) ----------------
        # 비물고기/24종 밖 어종을 흡수하는 25번째 클래스. 이게 없으면 고양이 사진도
        # 24종 중 하나로 강제 분류된다. (decisions.md A-10)
        _s("기타", "", "hard", "other",
           ["회 접시", "어시장", "낚시터 풍경"], [], ["비물고기", "unknown"], target=1000),
    ]
}

# 모델 출력 인덱스 ↔ 클래스명 ('기타' 포함, 총 25)
CLASSES: list[str] = list(SPECIES.keys())
NUM_CLASSES: int = len(CLASSES)

OTHER_CLASS = "기타"                # 이게 Top-1이면 서버는 uncertain 처리한다
OTHER_IDX = len(CLASSES) - 1
FISH_CLASSES: list[str] = [c for c in CLASSES if c != OTHER_CLASS]  # 실제 어종 24종
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: name for i, name in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# `기타`(OOD) 클래스 수집 명세
# ---------------------------------------------------------------------------
# 한 종류에 쏠리면 OOD 흡수가 안 된다(요리 사진만 1000장 → 고양이를 못 거른다).
# 4개 버킷을 비중대로 섞는다. 크롤러는 이 비중으로 버킷별 목표량을 나눈다.
# 상세 근거 → docs/data-pipeline.md "기타 클래스 수집"
OTHER_BUCKET_RATIO: dict[str, float] = {
    "other_fish": 0.40,   # 24종 밖 어종 (가장 중요: 실제 오분류의 주범)
    "cooked": 0.20,       # 조리·시장 사진
    "gear": 0.20,         # 낚시 관련 비물고기
    "misc": 0.20,         # 일반 오촬영
}

OTHER_KEYWORDS: dict[str, tuple[str, ...]] = {
    # ⚠️ '향어'는 넣지 않는다 — 잉어(Cyprinus carpio)의 품종이라 15번 클래스와 같은 종이다.
    #    (계획서엔 24종 밖 예시로 적혀 있지만 config의 잉어 aliases에 이미 포함돼 있다.)
    # ⚠️ 24종과 육안 구분이 안 되는 종(밀치=가숭어, 우럭볼락)은 넣지 않는다.
    #    기타에 넣으면 숭어/우럭 재현율을 깎는다. 오분류를 감수하는 편이 낫다.
    "other_fish": ("학꽁치", "붕장어", "대구 생선", "갑오징어", "문어", "꽃게",
                   "쥐노래미", "황어", "빙어", "가오리"),
    "cooked": ("회 접시", "생선구이", "어시장", "수산시장 생선", "매운탕", "초밥"),
    "gear": ("낚시터 풍경", "낚싯대", "루어 채비", "낚시 릴", "뜰채", "낚시 쿨러",
             "갯바위 낚시"),
    "misc": ("사람 얼굴", "손 사진", "바닥 콘크리트", "하늘 구름", "강아지", "고양이",
             "음식 사진"),
}

# `기타`의 other_fish 버킷은 학명이 있으므로 iNaturalist에서 깨끗하게 긁을 수 있다.
# (검색 크롤링보다 라벨 오염이 훨씬 적어 여기부터 채운다.)
# 학명은 **iNaturalist 활성 이름** 기준. 개정된 이름을 쓰면 검색이 0건이 된다
# (`python -m scripts.crawl_inat --dry-run` 으로 항상 먼저 확인할 것).
OTHER_INAT_TAXA: tuple[str, ...] = (
    "Hypophthalmichthys nobilis",   # 대두어(백연어) — 잉어와 헷갈리는 대형 담수어
    "Hyporhamphus sajori",          # 학꽁치
    "Conger myriaster",             # 붕장어
    "Gadus macrocephalus",          # 대구
    "Acanthosepion esculentum",     # 갑오징어 (구명 Sepia esculenta)
    "Octopus vulgaris",             # 문어
    "Portunus trituberculatus",     # 꽃게
    "Hexagrammos otakii",           # 쥐노래미
    "Pseudaspius hakonensis",       # 황어 (구명 Tribolodon hakonensis)
    "Pseudorasbora parva",          # 참붕어 (붕어 오분류 유발)
)

# 종당 iNat 관측 수가 부족한 클래스를 같은/근연 분류군으로 보충한다.
# 넣기 전 기준: **낚시인이 같은 이름으로 부르는 수준**이어야 한다. 애매하면 넣지 말고
# 검색 크롤링으로 채운다 (라벨 노이즈가 혼동 쌍 정확도를 직접 깎는다).
EXTRA_INAT_TAXA: dict[str, tuple[str, ...]] = {
    # iNat의 Carassius carassius는 유럽 붕어. 한국 붕어에 더 가까운 은붕어류를 보충.
    "붕어": ("Carassius gibelio", "Carassius langsdorfii"),
    # config가 부시리(S. lalandi)를 방어의 이명으로 묶었다 → iNat 36건 → 1,000건대로.
    "방어": ("Seriola lalandi",),
    # 일본에서 메바루 3종으로 쪼갠 것들. 한국에선 다 '볼락'.
    "볼락": ("Sebastes cheni", "Sebastes ventricosus"),
}

# 평가 시 집중 점검할 혼동 쌍/그룹 (Phase 4 혼동행렬에서 따로 뽑아본다)
CONFUSABLE_GROUPS: list[tuple[str, ...]] = [
    ("붕어", "잉어"),
    ("우럭", "볼락"),
    ("감성돔", "벵에돔"),
    ("고등어", "전갱이", "삼치"),
    ("숭어", "농어"),
    ("메기", "동자개"),
]


# ---------------------------------------------------------------------------
# 하이퍼파라미터
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    """학습 기본값. train.py 의 CLI 인자로 덮어쓸 수 있다."""

    # 모델
    backbone: str = "efficientnet_b0"   # 대안: convnext_tiny (정확도↑, 무거움)
    pretrained: bool = True
    drop_rate: float = 0.2              # classifier dropout
    drop_path_rate: float = 0.1         # stochastic depth

    # 입력
    img_size: int = 224
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    # 2단계 파인튜닝
    freeze_epochs: int = 3              # 1단계: 백본 freeze, 헤드만 학습
    epochs: int = 30                    # 총 epoch (freeze_epochs 포함)
    head_lr: float = 1e-3               # 1단계 LR
    lr: float = 3e-4                    # 2단계 LR (전체 unfreeze)
    min_lr: float = 1e-6
    warmup_epochs: int = 1              # 2단계 시작 시 워밍업
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    grad_clip: float = 1.0

    # 정규화 — 과적합 대응 (2026-08-18 진단: train_top1 0.99 vs val 0.75, 간격 24pp)
    # 두 사진과 라벨을 섞어 학습시켜 "사진 자체를 외우는" 경로를 물리적으로 막는다.
    # 0으로 두면 해당 기법이 꺼진다. mixup_prob=0 이면 둘 다 끈다(= 이전 동작).
    mixup_alpha: float = 0.2            # 픽셀 가중합 (Beta 분포 파라미터)
    cutmix_alpha: float = 1.0           # 사각형 패치 교체
    mixup_prob: float = 0.5             # 배치 단위 적용 확률

    # 데이터 로더
    batch_size: int = 32                # VRAM 2GB면 16 이하 권장
    num_workers: int = 4                # Windows에서 문제 생기면 0
    balanced_sampler: bool = True       # 클래스 불균형 → WeightedRandomSampler

    # 기타
    amp: bool = True                    # mixed precision (CUDA일 때만 적용)
    seed: int = 42
    patience: int = 8                   # early stopping (val 기준)
    monitor: str = "top3"               # top1 | top3 — 서비스 기준은 top3
    ema: bool = False                   # (예약) 미사용


# ---------------------------------------------------------------------------
# 서빙 설정
# ---------------------------------------------------------------------------
TOP_K = 3                    # 앱에 돌려줄 후보 개수
CONFIDENCE_THRESHOLD = 0.45  # top1 < 임계값이면 uncertain=true → "다시 촬영" 유도


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def validate() -> None:
    """설정 무결성 검사 (import 시 자동 실행)."""
    assert len(FISH_CLASSES) == 24, f"어종이 24가 아님: {len(FISH_CLASSES)}"
    assert NUM_CLASSES == 25, f"클래스 수가 25(24종+기타)가 아님: {NUM_CLASSES}"
    assert CLASSES[-1] == OTHER_CLASS, "'기타'는 항상 마지막 인덱스여야 한다"
    assert len(set(CLASSES)) == NUM_CLASSES, "중복된 종명이 있음"
    sci = [s.scientific for s in SPECIES.values() if s.scientific]
    assert len(set(sci)) == len(sci), "중복된 학명이 있음"
    for s in SPECIES.values():
        assert s.difficulty in DIFFICULTY_TARGET, f"{s.korean}: 잘못된 difficulty"
        assert s.habitat in ("sea", "fresh", "other"), f"{s.korean}: 잘못된 habitat"
        for other in s.confusable_with:
            assert other in SPECIES, f"{s.korean}: 알 수 없는 혼동종 {other}"


def ensure_dirs() -> None:
    """레포 폴더 구조를 생성한다 (이미 있으면 그대로 둔다)."""
    for d in (RAW_DIR, CLEAN_DIR, MODELS_DIR, SERVER_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for split_dir in SPLIT_DIRS.values():
        for name in CLASSES:
            (split_dir / name).mkdir(parents=True, exist_ok=True)
    for name in CLASSES:
        (RAW_DIR / name).mkdir(parents=True, exist_ok=True)
        (CLEAN_DIR / name).mkdir(parents=True, exist_ok=True)


def write_labels_json(path: Path = LABELS_JSON) -> Path:
    """추론 서버가 쓰는 인덱스→종명 매핑을 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,  # 인덱스 = 리스트 위치
        "other_class": OTHER_CLASS,   # Top-1이 이거면 서버는 uncertain 처리
        "other_index": OTHER_IDX,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "top_k": TOP_K,
        "species": {
            name: {
                "scientific": s.scientific,
                "habitat": s.habitat,
                "difficulty": s.difficulty,
            }
            for name, s in SPECIES.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def total_target_images() -> int:
    return sum(s.target_images for s in SPECIES.values())


def count_images(root: Path) -> dict[str, int]:
    """`root/<종명>/` 아래 이미지 개수를 센다. 폴더가 없으면 0."""
    counts = {}
    for name in CLASSES:
        d = root / name
        counts[name] = (
            sum(1 for p in d.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS)
            if d.is_dir()
            else 0
        )
    return counts


def _print_summary() -> None:
    print(f"클래스 수: {NUM_CLASSES} (어종 {len(FISH_CLASSES)} + '{OTHER_CLASS}')  |  "
          f"MVP 목표: {MVP_TARGET_IMAGES * NUM_CLASSES:,}장  |  "
          f"최종 목표: {total_target_images():,}장\n")
    raw = count_images(RAW_DIR)
    header = (f"{'idx':>3}  {'종명':<8} {'난이도':<7} {'서식':<5} "
              f"{'MVP':>5} {'최종':>6} {'raw':>6}  학명")
    print(header)
    print("-" * (len(header) + 12))
    for i, name in enumerate(CLASSES):
        s = SPECIES[name]
        print(f"{i:>3}  {name:<8} {s.difficulty:<7} {s.habitat:<5} "
              f"{MVP_TARGET_IMAGES:>5} {s.target_images:>6} {raw[name]:>6}  {s.scientific}")
    print("\n[혼동 주의 그룹]")
    for g in CONFUSABLE_GROUPS:
        print("  - " + " ↔ ".join(g))


validate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="fish-classifier 설정 유틸")
    ap.add_argument("--init", action="store_true", help="폴더 구조 생성 + labels.json 작성")
    ap.add_argument("--summary", action="store_true", help="클래스/수집 현황 요약")
    args = ap.parse_args()

    if args.init:
        ensure_dirs()
        p = write_labels_json()
        print(f"[OK] 폴더 구조 생성 완료: {PROJECT_ROOT}")
        print(f"[OK] labels.json 작성: {p}")
    if args.summary or not args.init:
        _print_summary()
