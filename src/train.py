"""학습 루프 — timm 백본 전이학습 (2단계 파인튜닝).

전략
----
Stage 1 (epoch 0 ~ freeze_epochs-1): 백본 freeze, 분류 헤드만 높은 LR로 학습.
    BatchNorm running stats도 함께 동결한다(작은 데이터셋에서 통계가 망가지는 것 방지).
Stage 2 (freeze_epochs ~ end): 전체 unfreeze, 낮은 LR + 워밍업 + 코사인 감쇠.

모니터링 지표는 기본이 **val Top-3**다. 서비스가 Top-3 후보를 보여주고
사용자가 확정하는 구조이므로, Top-1보다 Top-3가 실제 합격 기준에 가깝다.

사용 예:
    python -m src.train                                  # 기본값(efficientnet_b0)
    python -m src.train --backbone convnext_tiny --batch-size 16
    python -m src.train --smoke                          # 배치 몇 개만 돌려 파이프라인 점검
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src import config
from src.config import CLASSES, NUM_CLASSES, TrainConfig
from src.dataset import build_dataloader, compute_class_weights


# ---------------------------------------------------------------------------
# 준비
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # 입력 크기 고정이라 켜두면 빠르다


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: TrainConfig) -> nn.Module:
    return timm.create_model(
        cfg.backbone,
        pretrained=cfg.pretrained,
        num_classes=NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
    )


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """백본 freeze/unfreeze. 분류 헤드는 항상 학습한다."""
    for p in model.parameters():
        p.requires_grad = trainable
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def freeze_bn_stats(model: nn.Module) -> None:
    """BatchNorm의 running mean/var 갱신을 막는다 (Stage 1 전용)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """bias/norm 파라미터에는 weight decay를 걸지 않는다."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_stage(model: nn.Module, cfg: TrainConfig, stage: int):
    """stage(1|2)에 맞는 (optimizer, scheduler, 에폭 수)를 만든다."""
    if stage == 1:
        set_backbone_trainable(model, False)
        epochs = max(cfg.freeze_epochs, 0)
        opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.head_lr)
        sched = CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=cfg.min_lr)
        return opt, sched, epochs

    set_backbone_trainable(model, True)
    epochs = max(cfg.epochs - cfg.freeze_epochs, 1)
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr)
    warmup = min(cfg.warmup_epochs, max(epochs - 1, 0))
    cosine = CosineAnnealingLR(opt, T_max=max(epochs - warmup, 1), eta_min=cfg.min_lr)
    if warmup > 0:
        sched = SequentialLR(
            opt,
            schedulers=[LinearLR(opt, start_factor=0.1, total_iters=warmup), cosine],
            milestones=[warmup],
        )
    else:
        sched = cosine
    return opt, sched, epochs


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def topk_correct(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 3)) -> dict[int, int]:
    """배치 내 Top-k 정답 개수."""
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)                  # (B, maxk)
    hit = pred.eq(targets.view(-1, 1))
    return {k: hit[:, :min(k, maxk)].any(dim=1).sum().item() for k in ks}


# ---------------------------------------------------------------------------
# 에폭 루프
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, *, optimizer=None, scaler=None,
              cfg: TrainConfig, desc: str, max_batches: int | None = None,
              frozen: bool = False, mixup_fn=None) -> dict[str, float]:
    """한 epoch 실행.

    mixup_fn 이 주어지면 학습 배치의 이미지/라벨을 섞는다. 이때 loss는 섞인 soft
    타깃에 대해 계산하지만, **정확도는 원본 라벨 y 로 잰다** — 섞인 라벨에 대한
    정확도는 해석이 불가능하고, 우리가 보려는 건 과적합 간격이기 때문이다.
    (그래서 mixup을 켜면 train_top1 이 이전보다 낮게 나오는 게 정상이다.)
    """
    train_mode = optimizer is not None
    model.train(train_mode)
    if train_mode and frozen:
        freeze_bn_stats(model)

    use_amp = cfg.amp and device.type == "cuda"
    total, loss_sum = 0, 0.0
    hits = {1: 0, 3: 0}

    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for step, (x, y) in enumerate(bar):
        if max_batches is not None and step >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_loss = y
        if mixup_fn is not None:
            x, y_loss = mixup_fn(x, y)   # y_loss는 soft 타깃, y는 정확도 측정용 원본

        with torch.set_grad_enabled(train_mode):
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y_loss)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if cfg.grad_clip:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()

        bs = y.size(0)
        total += bs
        loss_sum += loss.item() * bs
        batch_hits = topk_correct(logits.detach().float(), y)
        for k in hits:
            hits[k] += batch_hits[k]
        bar.set_postfix(loss=f"{loss_sum / total:.3f}", top1=f"{hits[1] / total:.3f}")

    total = max(total, 1)
    return {
        "loss": loss_sum / total,
        "top1": hits[1] / total,
        "top3": hits[3] / total,
    }


# ---------------------------------------------------------------------------
# 체크포인트
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, model: nn.Module, cfg: TrainConfig,
                    epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": cfg.backbone,
            "num_classes": NUM_CLASSES,
            "classes": CLASSES,            # 라벨 순서를 체크포인트에 박아둔다
            "img_size": cfg.img_size,
            "mean": cfg.mean,
            "std": cfg.std,
            "epoch": epoch,
            "metrics": metrics,
            "config": asdict(cfg),
        },
        path,
    )


def append_history(row: dict) -> None:
    new_file = not config.HISTORY_CSV.exists()
    config.HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with config.HISTORY_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    d = TrainConfig()
    ap = argparse.ArgumentParser(description="24종 어류 분류 학습")
    ap.add_argument("--backbone", default=d.backbone)
    ap.add_argument("--img-size", type=int, default=d.img_size)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--freeze-epochs", type=int, default=d.freeze_epochs)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--lr", type=float, default=d.lr, help="stage 2 LR")
    ap.add_argument("--head-lr", type=float, default=d.head_lr, help="stage 1 LR")
    ap.add_argument("--weight-decay", type=float, default=d.weight_decay)
    ap.add_argument("--label-smoothing", type=float, default=d.label_smoothing)
    ap.add_argument("--mixup-alpha", type=float, default=d.mixup_alpha)
    ap.add_argument("--cutmix-alpha", type=float, default=d.cutmix_alpha)
    ap.add_argument("--mixup-prob", type=float, default=d.mixup_prob,
                    help="배치 단위 mixup/cutmix 적용 확률. 0이면 끈다")
    ap.add_argument("--no-mixup", action="store_true", help="mixup/cutmix 완전히 끄기")
    ap.add_argument("--num-workers", type=int, default=d.num_workers)
    ap.add_argument("--patience", type=int, default=d.patience)
    ap.add_argument("--monitor", choices=["top1", "top3"], default=d.monitor)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--no-amp", action="store_true", help="mixed precision 끄기")
    ap.add_argument("--no-pretrained", action="store_true", help="사전학습 가중치 없이(비권장)")
    ap.add_argument("--class-weights", action="store_true",
                    help="샘플러 대신 CrossEntropy 클래스 가중치 사용")
    ap.add_argument("--no-balanced-sampler", action="store_true")
    ap.add_argument("--resume", type=Path, default=None, help="체크포인트에서 이어서 학습")
    ap.add_argument("--out", type=Path, default=config.BEST_CKPT)
    ap.add_argument("--smoke", action="store_true",
                    help="배치 3개만 돌려 파이프라인 점검 (2 epoch)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(
        TrainConfig(),
        backbone=args.backbone,
        img_size=args.img_size,
        epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mixup_prob=0.0 if args.no_mixup else args.mixup_prob,
        num_workers=args.num_workers,
        patience=args.patience,
        monitor=args.monitor,
        seed=args.seed,
        amp=not args.no_amp,
        pretrained=not args.no_pretrained,
        balanced_sampler=not (args.no_balanced_sampler or args.class_weights),
    )
    if args.smoke:
        cfg = replace(cfg, epochs=2, freeze_epochs=1, warmup_epochs=0, num_workers=0)
    max_batches = 3 if args.smoke else None

    set_seed(cfg.seed)
    device = get_device()
    print(f"[env] device={device} | torch={torch.__version__} | "
          f"amp={'on' if cfg.amp and device.type == 'cuda' else 'off'}")
    print(f"[cfg] {cfg.backbone} | img={cfg.img_size} | bs={cfg.batch_size} | "
          f"epochs={cfg.epochs}(freeze {cfg.freeze_epochs}) | monitor=val_{cfg.monitor}")

    # --- 데이터 ---
    train_loader = build_dataloader("train", cfg)
    val_loader = build_dataloader("val", cfg)
    train_ds = train_loader.dataset
    print(f"[data] train={len(train_ds):,}장 | val={len(val_loader.dataset):,}장 | "
          f"클래스={NUM_CLASSES}")
    counts = train_ds.counts_by_name()
    lo = min(counts.values())
    hi = max(counts.values())
    print(f"[data] 클래스별 최소 {lo}장 / 최대 {hi}장 (불균형비 {hi / max(lo, 1):.1f}x)"
          f"{' — 샘플러로 보정' if cfg.balanced_sampler else ''}")

    # --- 모델/손실 ---
    model = build_model(cfg).to(device)
    weights = None
    if args.class_weights:
        weights = compute_class_weights(train_ds.targets).to(device)
        print("[loss] 클래스 가중치 적용")
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)

    # mixup을 쓰면 타깃이 soft가 되므로 손실 함수도 바뀐다. 검증은 항상 원래 CE로 잰다
    # (mixup 손실과 CE 손실은 스케일이 달라 섞어서 비교하면 곡선을 잘못 읽는다).
    mixup_fn = None
    train_criterion = criterion
    if cfg.mixup_prob > 0 and (cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0):
        if args.class_weights:
            # SoftTargetCrossEntropy는 클래스 가중치를 못 받는다. 둘 중 하나만 쓴다.
            print("[warn] --class-weights 는 mixup과 함께 쓸 수 없다 → mixup을 끈다")
        else:
            mixup_fn = Mixup(
                mixup_alpha=cfg.mixup_alpha,
                cutmix_alpha=cfg.cutmix_alpha,
                prob=cfg.mixup_prob,
                label_smoothing=cfg.label_smoothing,  # Mixup이 직접 처리한다
                num_classes=NUM_CLASSES,
            )
            train_criterion = SoftTargetCrossEntropy()
            print(f"[loss] mixup α={cfg.mixup_alpha} / cutmix α={cfg.cutmix_alpha} "
                  f"/ p={cfg.mixup_prob} → train_top1은 원본 라벨 기준(낮게 나온다)")

    scaler = GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")

    start_epoch, best = 0, 0.0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best = ckpt.get("metrics", {}).get(f"val_{cfg.monitor}", 0.0)
        print(f"[resume] {args.resume} (epoch {start_epoch}부터, best={best:.4f})")

    config.write_labels_json()  # 학습과 서빙의 라벨 순서를 항상 일치시킨다

    # --- 학습 ---
    bad_epochs = 0
    stage = 0
    optimizer = scheduler = None
    t0 = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        want_stage = 1 if epoch < cfg.freeze_epochs else 2
        if want_stage != stage:
            stage = want_stage
            optimizer, scheduler, _ = build_stage(model, cfg, stage)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_p = sum(p.numel() for p in model.parameters())
            print(f"\n=== Stage {stage}: "
                  f"{'헤드만 학습' if stage == 1 else '전체 파인튜닝'} | "
                  f"학습 파라미터 {trainable / 1e6:.2f}M / {total_p / 1e6:.2f}M ===")

        lr_now = optimizer.param_groups[0]["lr"]
        tr = run_epoch(model, train_loader, train_criterion, device, optimizer=optimizer,
                       scaler=scaler, cfg=cfg, desc=f"E{epoch + 1}/{cfg.epochs} train",
                       max_batches=max_batches, frozen=(stage == 1), mixup_fn=mixup_fn)
        va = run_epoch(model, val_loader, criterion, device, cfg=cfg,
                       desc=f"E{epoch + 1}/{cfg.epochs}  val", max_batches=max_batches)
        scheduler.step()

        score = va[cfg.monitor]
        is_best = score > best
        mark = "  ← best" if is_best else ""
        print(f"E{epoch + 1:>3}/{cfg.epochs} | lr {lr_now:.2e} | "
              f"train loss {tr['loss']:.3f} top1 {tr['top1']:.3f} | "
              f"val loss {va['loss']:.3f} top1 {va['top1']:.3f} top3 {va['top3']:.3f}"
              f"{mark}")

        append_history({
            "epoch": epoch + 1, "stage": stage, "lr": round(lr_now, 8),
            "train_loss": round(tr["loss"], 4), "train_top1": round(tr["top1"], 4),
            "val_loss": round(va["loss"], 4), "val_top1": round(va["top1"], 4),
            "val_top3": round(va["top3"], 4),
        })

        metrics = {f"val_{k}": v for k, v in va.items()}
        save_checkpoint(config.LAST_CKPT, model, cfg, epoch, metrics)
        if is_best:
            best = score
            bad_epochs = 0
            save_checkpoint(args.out, model, cfg, epoch, metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"[early stop] val_{cfg.monitor}가 {cfg.patience} epoch 동안 "
                      f"개선되지 않아 중단한다.")
                break

    mins = (time.time() - t0) / 60
    print(f"\n[done] 총 {mins:.1f}분 | best val_{cfg.monitor} = {best:.4f}")
    print(f"[done] 베스트 체크포인트: {args.out}")
    print(f"[next] 평가: python -m src.evaluate --ckpt {args.out}")


if __name__ == "__main__":  # Windows의 spawn 방식 DataLoader를 위해 반드시 필요
    main()
