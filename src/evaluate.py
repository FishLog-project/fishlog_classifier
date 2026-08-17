"""Phase 4 — test셋 평가 및 오류 진단.

이 스크립트의 목적은 "정확도 몇 %"를 재는 게 아니라 **어디서 왜 틀리는지**를
찾는 것이다. 베이스라인이 합격선에 미달했을 때 다음 수(데이터 정제 / 추가 수집 /
정규화 강화)를 고르려면 아래 셋을 구분해야 한다:

- 두 종이 **대칭으로** 헷갈린다 → 데이터 부족 (둘 다 특징을 못 배웠다)
- **한쪽으로만** 쏠린다 → 클래스 불균형 또는 그 종의 라벨 오염
- 확신하고 틀린 사진이 실제로 **다른 종/어시장 사진**이다 → 라벨 오염 확진

`기타`는 어종 지표와 분리해서 계산한다. 섞으면 24종 성능을 가린다
([evaluation.md](../docs/evaluation.md) 참조).

사용 예:
    python -m src.evaluate                          # models/best.pt, test셋
    python -m src.evaluate --split val              # 임계값 튜닝은 val에서 할 것
    python -m src.evaluate --ckpt models/last.pt
    python -m src.evaluate --no-worst-cases         # 이미지 복사 생략(빠름)
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")  # 헤드리스(Colab/서버)에서 창을 띄우지 않는다
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import config
from src.config import CLASSES, CONFUSABLE_GROUPS, NUM_CLASSES, SPECIES, TrainConfig
from src.dataset import FishDataset, build_transforms
from src.train import build_model, get_device

OTHER = "기타"
OTHER_IDX = CLASSES.index(OTHER)
FISH_IDX = [i for i, name in enumerate(CLASSES) if name != OTHER]


# ---------------------------------------------------------------------------
# 한글 폰트
# ---------------------------------------------------------------------------
# matplotlib 기본 폰트(DejaVu Sans)에는 한글 글리프가 없어 혼동행렬 라벨이 전부
# 네모(두부)로 나온다. 혼동행렬은 이 스크립트의 핵심 산출물이므로 폰트를 찾고,
# 없으면 라벨을 인덱스 숫자로 낮춰서라도 읽을 수 있게 만든다.
_KOREAN_FONT_CANDIDATES = (
    "Malgun Gothic",      # Windows 기본
    "AppleGothic",        # macOS
    "NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR",  # Linux/Colab
)


def setup_korean_font() -> bool:
    """한글 렌더링이 가능하면 True. Colab은 `apt-get install -y fonts-nanum` 후 가능."""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _KOREAN_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트는 U+2212를 종종 빠뜨린다
            return True

    print("[warn] 한글 폰트를 못 찾았다 → 그래프 라벨을 인덱스 숫자로 대체한다.")
    print("       Colab: !apt-get install -y fonts-nanum "
          "&& rm -rf ~/.cache/matplotlib  (설치 후 런타임 재시작)")
    return False


def _tick_labels(has_font: bool) -> list[str]:
    """폰트가 있으면 `0:감성돔`, 없으면 `0`. 없을 때 이름은 metrics.json에서 본다."""
    return [f"{i}:{n}" for i, n in enumerate(CLASSES)] if has_font \
        else [str(i) for i in range(NUM_CLASSES)]


# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, TrainConfig]:
    """체크포인트를 읽어 모델을 복원한다. 라벨 순서가 어긋나면 즉시 멈춘다."""
    if not path.exists():
        raise SystemExit(f"[FAIL] 체크포인트가 없다: {path}\n  → `python -m src.train` 을 먼저 돌릴 것.")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # 학습 당시의 클래스 순서와 지금 config가 다르면 모든 지표가 조용히 틀어진다.
    saved = ckpt.get("classes")
    if saved is not None and list(saved) != CLASSES:
        raise SystemExit(
            "[FAIL] 체크포인트의 클래스 순서가 현재 config.SPECIES 와 다르다.\n"
            f"  ckpt: {saved}\n  now : {CLASSES}\n"
            "  → config를 되돌리거나 재학습할 것. (CLAUDE.md 불변 규칙)"
        )

    cfg = TrainConfig(**{k: v for k, v in ckpt.get("config", {}).items()
                         if k in TrainConfig.__dataclass_fields__})
    cfg = TrainConfig(**{**cfg.__dict__, "backbone": ckpt.get("backbone", cfg.backbone),
                         "img_size": ckpt.get("img_size", cfg.img_size)})

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    ep = ckpt.get("epoch", "?")
    met = ckpt.get("metrics", {})
    print(f"[ckpt] {path.name} | {cfg.backbone} | epoch {ep} | 학습 당시 {met}")
    return model, cfg


@torch.no_grad()
def predict_all(model, ds: FishDataset, cfg: TrainConfig, device: torch.device,
                batch_size: int = 64, num_workers: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """전체 데이터셋에 대한 softmax 확률과 정답 라벨을 반환한다."""
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    probs, targets = [], []
    for x, y in tqdm(loader, desc="predict", leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int,
                  mask: np.ndarray | None = None) -> float:
    if mask is not None:
        probs, y = probs[mask], y[mask]
    if len(y) == 0:
        return float("nan")
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def per_class_metrics(pred: np.ndarray, y: np.ndarray, probs: np.ndarray) -> dict[str, dict]:
    """종별 precision / recall / F1 / top3 recall / support."""
    out: dict[str, dict] = {}
    top3 = np.argsort(-probs, axis=1)[:, :3]
    for i, name in enumerate(CLASSES):
        tp = int(((pred == i) & (y == i)).sum())
        fp = int(((pred == i) & (y != i)).sum())
        fn = int(((pred != i) & (y == i)).sum())
        support = int((y == i).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        in3 = (top3[y == i] == i).any(axis=1)
        out[name] = {
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "top3_recall": round(float(in3.mean()) if support else 0.0, 4),
            "difficulty": SPECIES[name].difficulty,
        }
    return out


def confusion_matrix(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


# ---------------------------------------------------------------------------
# 산출물
# ---------------------------------------------------------------------------
def plot_confusion(cm: np.ndarray, out: Path, has_font: bool) -> None:
    """행 정규화(=recall 기준) 혼동행렬. 대각선이 진할수록 좋다."""
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    labels = _tick_labels(has_font)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    if has_font:
        ax.set_xlabel("예측"); ax.set_ylabel("정답")
        ax.set_title("혼동행렬 (행 정규화 = recall)")
    else:
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title("Confusion matrix (row-normalized = recall)")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if norm[i, j] >= 0.05:
                ax.text(j, i, f"{norm[i, j]*100:.0f}", ha="center", va="center",
                        fontsize=6, color="white" if norm[i, j] < 0.6 else "black")

    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_confusable_report(cm: np.ndarray, metrics: dict[str, dict], out: Path) -> None:
    """혼동 그룹별 상호 오분류율. 대칭/비대칭 판정이 이 파일의 핵심이다."""
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)

    lines = [
        "# 혼동 그룹 리포트",
        "",
        "각 셀 = `정답 행 → 예측 열` 비율(%). 대각선은 recall.",
        "",
        "**읽는 법**: 두 종이 서로 비슷한 비율로 헷갈리면(대칭) 데이터 부족,",
        "한쪽으로만 쏠리면(비대칭) 클래스 불균형 또는 그 종의 라벨 오염이다.",
        "",
    ]

    for group in CONFUSABLE_GROUPS:
        idx = [CLASSES.index(n) for n in group]
        lines += [f"## {' ↔ '.join(group)}", "",
                  "| 정답＼예측 | " + " | ".join(group) + " | 그룹 밖 |",
                  "|---" * (len(group) + 2) + "|"]
        for name, i in zip(group, idx):
            cells = [f"**{norm[i, j]*100:.1f}**" if i == j else f"{norm[i, j]*100:.1f}"
                     for j in idx]
            outside = (1.0 - sum(norm[i, j] for j in idx)) * 100
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {outside:.1f} |")

        # 대칭/비대칭 자동 판정 (쌍 단위)
        notes = []
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                ia, ib = idx[a], idx[b]
                ab, ba = norm[ia, ib] * 100, norm[ib, ia] * 100
                if max(ab, ba) < 5:
                    continue
                if min(ab, ba) > 0 and max(ab, ba) / max(min(ab, ba), 1e-9) < 2:
                    notes.append(f"- {group[a]} ↔ {group[b]}: **대칭 혼동** "
                                 f"({ab:.1f}% / {ba:.1f}%) → 데이터 부족 쪽을 의심")
                else:
                    hi, lo = (group[a], group[b]) if ab > ba else (group[b], group[a])
                    notes.append(f"- {group[a]} ↔ {group[b]}: **비대칭** "
                                 f"({ab:.1f}% / {ba:.1f}%) → `{hi}` 를 `{lo}` 로 미는 경향. "
                                 f"`{hi}` 의 라벨 오염 의심")
        lines += ["", *(notes or ["- 유의미한 상호 혼동 없음(5% 미만)"]), ""]

        lines.append("| 종 | recall | top3_recall | support |")
        lines.append("|---|---|---|---|")
        for name in group:
            m = metrics[name]
            lines.append(f"| {name} | {m['recall']*100:.1f}% | "
                         f"{m['top3_recall']*100:.1f}% | {m['support']} |")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def dump_worst_cases(ds: FishDataset, probs: np.ndarray, y: np.ndarray,
                     out_dir: Path, n: int = 50) -> list[dict]:
    """확신했는데 틀린 상위 n장을 복사한다. 라벨 오염을 눈으로 잡는 가장 빠른 길.

    파일명: `{확신도}_정답-{정답종}_예측-{예측종}_{원본명}` — 정렬하면 확신 순이다.
    """
    pred = probs.argmax(axis=1)
    wrong = np.flatnonzero(pred != y)
    conf = probs[wrong, pred[wrong]]
    order = wrong[np.argsort(-conf)][:n]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, i in enumerate(order):
        src = ds.samples[i][0]
        c = float(probs[i, pred[i]])
        true_n, pred_n = CLASSES[y[i]], CLASSES[pred[i]]
        dst = out_dir / f"{rank:02d}_{c*100:.0f}_정답-{true_n}_예측-{pred_n}_{src.name}"
        try:
            shutil.copy2(src, dst)
        except OSError:
            pass
        rows.append({"path": str(src), "true": true_n, "pred": pred_n,
                     "confidence": round(c, 4)})
    return rows


def plot_threshold_curve(probs: np.ndarray, y: np.ndarray, out: Path,
                         has_font: bool) -> list[dict]:
    """`uncertain` 임계값 후보별 (거부율, 남은 것의 정확도) 트레이드오프.

    최대 확률이 임계값 미만이면 '판단 보류'로 돌려보내는 정책을 가정한다.
    """
    top1 = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    hit1, hit3 = top1 == y, (top3 == y[:, None]).any(axis=1)

    ths = np.arange(0.20, 0.85, 0.05)
    rows = []
    for t in ths:
        keep = conf >= t
        rows.append({
            "threshold": round(float(t), 2),
            "reject_rate": round(float(1 - keep.mean()), 4),
            "top1_on_kept": round(float(hit1[keep].mean()) if keep.any() else float("nan"), 4),
            "top3_on_kept": round(float(hit3[keep].mean()) if keep.any() else float("nan"), 4),
        })

    ko = {"t3": "top3 (통과분)", "t1": "top1 (통과분)", "rej": "거부율",
          "cap": "거부율 10% 상한", "x": "uncertain 임계값", "y": "비율",
          "title": "임계값 트레이드오프"}
    en = {"t3": "top3 (kept)", "t1": "top1 (kept)", "rej": "reject rate",
          "cap": "reject rate cap 10%", "x": "uncertain threshold", "y": "rate",
          "title": "Threshold trade-off"}
    L = ko if has_font else en

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ths, [r["top3_on_kept"] for r in rows], "o-", label=L["t3"])
    ax.plot(ths, [r["top1_on_kept"] for r in rows], "s-", label=L["t1"])
    ax.plot(ths, [r["reject_rate"] for r in rows], "^--", label=L["rej"])
    ax.axhline(0.10, color="r", ls=":", lw=1, label=L["cap"])
    ax.set_xlabel(L["x"]); ax.set_ylabel(L["y"])
    ax.set_title(L["title"]); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return rows


# ---------------------------------------------------------------------------
def print_summary(metrics: dict, per_class: dict[str, dict]) -> None:
    o = metrics["overall"]
    print("\n" + "=" * 62)
    print(f"  {metrics['split']}셋 {metrics['n_samples']:,}장 평가")
    print("=" * 62)

    def line(label: str, value: float, target: float | None) -> str:
        if target is None:
            return f"  {label:<28} {value*100:6.2f}%"
        mark = "✅" if value >= target else "❌"
        return f"  {label:<28} {value*100:6.2f}%   (기준 {target*100:.0f}%) {mark}"

    print(line("전체 Top-3 (25클래스)", o["top3"], 0.90))
    print(line("전체 Top-1 (25클래스)", o["top1"], 0.75))
    print(line("어종 24종만 Top-3", o["top3_fish_only"], 0.90))
    print(line("어종 24종만 Top-1", o["top1_fish_only"], None))
    print(line("기타 recall", per_class[OTHER]["recall"], 0.80))
    print(line("기타 오탐율(어종→기타)", o["other_false_positive_rate"], None)
          + f"   (기준 10% 이하 {'✅' if o['other_false_positive_rate'] <= 0.10 else '❌'})")

    print("\n  [종별 recall 하위 8종]  기준 60%")
    ranked = sorted(((n, m) for n, m in per_class.items()), key=lambda kv: kv[1]["recall"])
    for name, m in ranked[:8]:
        mark = "❌" if m["recall"] < 0.60 else "  "
        print(f"    {mark} {name:<6} recall {m['recall']*100:5.1f}%  "
              f"top3 {m['top3_recall']*100:5.1f}%  (n={m['support']}, {m['difficulty']})")

    low = [n for n, m in per_class.items() if m["recall"] < 0.60]
    if low:
        print(f"\n  ⚠ recall 60% 미만 {len(low)}종: {', '.join(low)}")
        print("     → reports/worst_cases/ 를 먼저 볼 것. 라벨 오염이면 재수집은 낭비다.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="test셋 평가 및 오류 진단")
    ap.add_argument("--ckpt", type=Path, default=config.BEST_CKPT)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--worst-n", type=int, default=50)
    ap.add_argument("--no-worst-cases", action="store_true",
                    help="worst_cases 이미지 복사 생략")
    ap.add_argument("--out-dir", type=Path, default=config.REPORTS_DIR)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"[env] device={device}")

    model, cfg = load_checkpoint(args.ckpt, device)

    root = config.SPLIT_DIRS[args.split]
    ds = FishDataset(root, transform=build_transforms(args.split, cfg))
    if len(ds) == 0:
        raise SystemExit(f"[FAIL] {root} 에 이미지가 없다.")
    print(f"[data] {args.split}셋 {len(ds):,}장")

    probs, y = predict_all(model, ds, cfg, device, args.batch_size, args.num_workers)
    pred = probs.argmax(axis=1)

    per_class = per_class_metrics(pred, y, probs)
    cm = confusion_matrix(pred, y)
    fish_mask = y != OTHER_IDX

    # 어종을 `기타`로 밀어낸 비율 — UX를 직접 망가뜨리는 지표라 따로 센다
    other_fp = float((pred[fish_mask] == OTHER_IDX).mean()) if fish_mask.any() else 0.0

    metrics = {
        "split": args.split,
        "n_samples": int(len(y)),
        "checkpoint": str(args.ckpt),
        "backbone": cfg.backbone,
        "img_size": cfg.img_size,
        "overall": {
            "top1": round(topk_accuracy(probs, y, 1), 4),
            "top3": round(topk_accuracy(probs, y, 3), 4),
            "top1_fish_only": round(topk_accuracy(probs, y, 1, fish_mask), 4),
            "top3_fish_only": round(topk_accuracy(probs, y, 3, fish_mask), 4),
            "macro_recall": round(float(np.mean([m["recall"] for m in per_class.values()])), 4),
            "other_false_positive_rate": round(other_fp, 4),
        },
        "per_class": per_class,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    has_font = setup_korean_font()
    plot_confusion(cm, args.out_dir / "confusion_matrix.png", has_font)
    write_confusable_report(cm, per_class, args.out_dir / "confusable_report.md")
    metrics["threshold_curve"] = plot_threshold_curve(
        probs, y, args.out_dir / "threshold_curve.png", has_font)

    if not args.no_worst_cases:
        metrics["worst_cases"] = dump_worst_cases(
            ds, probs, y, args.out_dir / "worst_cases", args.worst_n)

    np.save(args.out_dir / "confusion_matrix.npy", cm)
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(metrics, per_class)
    print(f"\n[out] {args.out_dir}")
    for f in ("metrics.json", "confusion_matrix.png", "confusable_report.md",
              "threshold_curve.png"):
        print(f"       {f}")
    if not args.no_worst_cases:
        print(f"       worst_cases/ ({args.worst_n}장)")
    print("\n[next] confusable_report.md 의 대칭/비대칭 판정 → worst_cases/ 육안 확인 순으로 볼 것")


if __name__ == "__main__":
    main()
