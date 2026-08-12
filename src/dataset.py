"""Dataset / DataLoader / 증강 파이프라인.

핵심 설계
---------
1. **라벨 인덱스는 폴더 정렬 순서가 아니라 `config.CLASSES` 순서를 따른다.**
   (torchvision ImageFolder 는 폴더명 정렬로 인덱스를 매기는데, 한글 폴더명은
    OS/로케일에 따라 정렬이 달라질 수 있어 라벨이 어긋날 위험이 있다.)
2. **한글 경로 안전 로딩.** `cv2.imread` 는 Windows에서 비ASCII 경로를 못 읽는다.
   → `np.fromfile` + `cv2.imdecode` 로 우회.
3. 손상 파일은 학습을 멈추지 않고 건너뛴다(같은 클래스의 다른 샘플로 대체).

사용 예:
    python -m src.dataset --check              # 분할 현황 점검
    python -m src.dataset --preview            # 증강 결과 미리보기 이미지 저장
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src import config
from src.config import CLASS_TO_IDX, CLASSES, VALID_IMAGE_EXTS, TrainConfig

cv2.setNumThreads(0)  # DataLoader worker와 OpenCV 내부 스레드 경합 방지


# ---------------------------------------------------------------------------
# I/O 유틸
# ---------------------------------------------------------------------------
def imread_unicode(path: Path) -> np.ndarray | None:
    """한글/유니코드 경로에서도 동작하는 이미지 로더. 실패 시 None."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR, 알파/그레이스케일도 3채널로
    except Exception:
        return None
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_unicode(path: Path, img_rgb: np.ndarray) -> bool:
    """한글 경로 안전 저장 (RGB 입력)."""
    ok, buf = cv2.imencode(path.suffix, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def scan_split(root: Path) -> list[tuple[Path, int]]:
    """`root/<종명>/*.jpg` 를 훑어 (경로, 라벨인덱스) 목록을 만든다."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"데이터 폴더가 없다: {root}\n"
            "  → Phase 2(수집·정제·분할)를 먼저 끝내고 data/splits/ 를 채울 것."
        )
    samples: list[tuple[Path, int]] = []
    for name in CLASSES:  # config 순서 고정
        class_dir = root / name
        if not class_dir.is_dir():
            continue
        idx = CLASS_TO_IDX[name]
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in VALID_IMAGE_EXTS and p.is_file():
                samples.append((p, idx))
    return samples


# ---------------------------------------------------------------------------
# 증강
# ---------------------------------------------------------------------------
def build_transforms(split: str, cfg: TrainConfig) -> A.Compose:
    """split('train'|'val'|'test')별 전처리/증강 파이프라인 (albumentations 2.x API)."""
    size = cfg.img_size
    normalize = [A.Normalize(mean=cfg.mean, std=cfg.std), ToTensorV2()]

    if split != "train":
        # 평가: 짧은 변을 살짝 크게 리사이즈 후 중앙 크롭 (aspect ratio 보존)
        return A.Compose([
            A.SmallestMaxSize(max_size=int(size * 1.14)),
            A.CenterCrop(height=size, width=size),
            *normalize,
        ])

    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.55, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), scale=(0.9, 1.1),
                 translate_percent=(-0.05, 0.05), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=25,
                             val_shift_limit=15, p=0.5),
        # 실사용 사진(손에 든/젖은/역광/폰카메라) 도메인 갭 흡수용
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.GaussNoise(std_range=(0.03, 0.12)),
        ], p=0.25),
        A.ImageCompression(quality_range=(55, 100), p=0.3),
        *normalize,
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FishDataset(Dataset):
    """24종 어류 분류 데이터셋.

    Args:
        root: `data/splits/train` 같은 분할 폴더.
        transform: albumentations Compose. None이면 원본 RGB ndarray를 반환.
        max_retry: 손상 파일을 만났을 때 다른 샘플로 대체 시도 횟수.
    """

    def __init__(self, root: Path, transform: A.Compose | None = None, max_retry: int = 5):
        self.root = Path(root)
        self.samples = scan_split(self.root)
        if not self.samples:
            raise RuntimeError(f"이미지가 하나도 없다: {self.root}")
        self.transform = transform
        self.max_retry = max_retry
        self.targets = [label for _, label in self.samples]
        self.class_counts = Counter(self.targets)
        self._broken: set[int] = set()

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, i: int) -> np.ndarray | None:
        path, _ = self.samples[i]
        img = imread_unicode(path)
        if img is None and i not in self._broken:
            self._broken.add(i)
            print(f"[warn] 손상/판독불가 이미지 건너뜀: {path}")
        return img

    def __getitem__(self, i: int):
        img = self._load(i)
        label = self.samples[i][1]

        # 손상 파일 → 같은 클래스의 다른 샘플로 대체 (라벨 분포 유지)
        retry = 0
        while img is None and retry < self.max_retry:
            j = random.randrange(len(self.samples))
            if self.samples[j][1] == label:
                img = self._load(j)
                retry += 1
        if img is None:
            # 최후의 수단: 검은 이미지 (transform이 알아서 리사이즈한다)
            img = np.zeros((256, 256, 3), dtype=np.uint8)

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label

    def counts_by_name(self) -> dict[str, int]:
        return {name: self.class_counts.get(CLASS_TO_IDX[name], 0) for name in CLASSES}


# ---------------------------------------------------------------------------
# 클래스 불균형 대응
# ---------------------------------------------------------------------------
def make_weighted_sampler(targets: list[int]) -> WeightedRandomSampler:
    """클래스별 등장 확률을 균등하게 맞추는 샘플러 (오버/언더샘플링 효과)."""
    counts = Counter(targets)
    per_class = {c: 1.0 / n for c, n in counts.items() if n > 0}
    weights = [per_class[t] for t in targets]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(targets),
        replacement=True,
    )


def compute_class_weights(targets: list[int], num_classes: int = len(CLASSES)) -> torch.Tensor:
    """CrossEntropy용 클래스 가중치 (샘플러 대신 쓸 때). 없는 클래스는 0."""
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    weights = np.zeros_like(counts)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (nonzero.sum() * counts[nonzero])
    return torch.as_tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# DataLoader 조립
# ---------------------------------------------------------------------------
def build_dataloader(split: str, cfg: TrainConfig, shuffle: bool | None = None) -> DataLoader:
    root = config.SPLIT_DIRS[split]
    ds = FishDataset(root, transform=build_transforms(split, cfg))
    is_train = split == "train"

    sampler = None
    if is_train and cfg.balanced_sampler:
        sampler = make_weighted_sampler(ds.targets)
    if shuffle is None:
        shuffle = is_train and sampler is None

    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )


def build_dataloaders(cfg: TrainConfig, splits=("train", "val")) -> dict[str, DataLoader]:
    return {s: build_dataloader(s, cfg) for s in splits}


# ---------------------------------------------------------------------------
# 점검 도구
# ---------------------------------------------------------------------------
def check_splits() -> None:
    """각 split의 클래스별 이미지 수와 불균형 정도를 출력."""
    print(f"{'종명':<8} {'train':>7} {'val':>6} {'test':>6} {'합계':>7} {'목표':>7}")
    print("-" * 52)
    per_split = {s: config.count_images(d) for s, d in config.SPLIT_DIRS.items()}
    totals = Counter()
    missing = []
    for name in CLASSES:
        row = {s: counts.get(name, 0) for s, counts in per_split.items()}
        total = sum(row.values())
        target = config.SPECIES[name].target_images
        flag = "" if total >= target * 0.5 else "  ← 부족"
        print(f"{name:<8} {row['train']:>7} {row['val']:>6} {row['test']:>6} "
              f"{total:>7} {target:>7}{flag}")
        for k, v in row.items():
            totals[k] += v
        if total == 0:
            missing.append(name)
    print("-" * 52)
    print(f"{'합계':<8} {totals['train']:>7} {totals['val']:>6} {totals['test']:>6} "
          f"{sum(totals.values()):>7} {config.total_target_images():>7}")
    if missing:
        print(f"\n[warn] 이미지가 0장인 종 {len(missing)}개: {', '.join(missing)}")


def save_augmentation_preview(cfg: TrainConfig, n: int = 16,
                              out: Path | None = None) -> Path | None:
    """증강 결과를 4x4 타일 이미지로 저장해 눈으로 확인한다."""
    out = out or (config.REPORTS_DIR / "aug_preview.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = FishDataset(config.TRAIN_DIR, transform=build_transforms("train", cfg))

    mean = np.array(cfg.mean, dtype=np.float32)
    std = np.array(cfg.std, dtype=np.float32)
    tiles = []
    for i in random.sample(range(len(ds)), min(n, len(ds))):
        x, y = ds[i]
        img = x.permute(1, 2, 0).numpy() * std + mean
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        tiles.append(img)

    cols = 4
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles) - cols + 1, cols)]
    if not rows:
        print("[warn] 미리보기를 만들 이미지가 부족하다.")
        return None
    grid = np.vstack(rows)
    imwrite_unicode(out, grid)
    print(f"[OK] 증강 미리보기 저장: {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="데이터셋 점검 유틸")
    ap.add_argument("--check", action="store_true", help="split별 클래스 분포 출력")
    ap.add_argument("--preview", action="store_true", help="증강 결과 미리보기 저장")
    ap.add_argument("--batch", action="store_true", help="train 배치 1개 로드 테스트")
    ap.add_argument("--img-size", type=int, default=TrainConfig.img_size)
    args = ap.parse_args()

    cfg = TrainConfig(img_size=args.img_size, num_workers=0)
    if args.batch:
        loader = build_dataloader("train", cfg)
        x, y = next(iter(loader))
        print(f"batch x: {tuple(x.shape)} {x.dtype}  |  y: {tuple(y.shape)} "
              f"| 라벨 예시: {[CLASSES[i] for i in y[:5].tolist()]}")
    if args.preview:
        save_augmentation_preview(cfg)
    if args.check or not (args.preview or args.batch):
        check_splits()
