"""사전학습 모델로 '물고기 같지 않은' 사진을 골라낸다 (사람 검수 부담 줄이기).

ImageNet-1k 사전학습 모델의 **어류 클래스 확률 합**을 'fishiness' 점수로 쓴다.
정확한 종 판별은 못 하지만, 낚싯대 사진·풍경·사람·회 접시처럼 **물고기가 없는 사진**은
잘 걸러진다. 검색 크롤링분(web_*)의 절반 가까이가 이런 사진이다.

**절대 자동 삭제하지 않는다.** 기본은 리포트만 쓰고, `--apply` 를 줘야 `data/reject/`로
옮긴다(clean은 raw의 하드링크라 raw 원본은 그대로 남는다). 임계값을 잘못 잡으면
'손에 든 물고기' 사진이 대량으로 날아가므로, 먼저 리포트 CSV를 눈으로 확인할 것.

`기타` 클래스는 **일부러 비물고기를 모은 것**이므로 기본적으로 건너뛴다.

사용 예:
    python -m scripts.prefilter                       # 리포트만 (reports/prefilter.csv)
    python -m scripts.prefilter --species 볼락 --threshold 0.05
    python -m scripts.prefilter --apply               # 의심 파일을 data/reject/로 이동
    python -m scripts.prefilter --include-other       # '기타'도 점수를 매겨본다(참고용)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from src import config
from scripts._common import append_rejected, resolve_species

try:
    import timm
    from timm.data import create_transform, resolve_data_config
except ImportError:  # pragma: no cover
    raise SystemExit("timm이 없다 → pip install -r requirements.txt")

REJECT_DIR = config.DATA_DIR / "reject"
REPORT_CSV = config.REPORTS_DIR / "prefilter.csv"

# ImageNet-1k에서 어류/수생동물 인덱스. 종 식별용이 아니라 '물고기가 있나' 판정용이다.
FISH_IDX = (
    0, 1, 2, 3, 4, 5, 6,                    # tench, goldfish, sharks, rays
    389, 390, 391, 392, 393, 394, 395, 396, 397,  # barracouta, eel, coho, ... puffer
)
# 조리·식탁 정황 — '요리 사진' 후보를 따로 표시해 검수 우선순위를 높인다
FOOD_IDX = (923, 809, 532, 762, 659, 964)   # plate, soup bowl, dining table, restaurant,
                                            # mixing bowl, potpie
# 큰 글씨가 박힌 유튜브/블로그 썸네일이 여기로 몰린다(comic book, book jacket, menu…).
# 물고기가 찍혀 있어도 자막 텍스트를 특징으로 학습할 수 있어 학습셋에 좋지 않다.
TEXT_IDX = (916, 917, 918, 921, 922)        # web site, comic book, crossword, book jacket, menu


def load_model(name: str, device: torch.device):
    model = timm.create_model(name, pretrained=True).eval().to(device)
    cfg = resolve_data_config({}, model=model)
    tf = create_transform(**cfg, is_training=False)
    labels = None
    try:  # timm 버전에 따라 없을 수 있다 — 없으면 인덱스만 출력
        from timm.data import ImageNetInfo
        labels = ImageNetInfo().index_to_description
    except Exception:
        pass
    return model, tf, labels


def describe(labels, idx: int) -> str:
    if labels is None:
        return f"idx{idx}"
    try:
        return labels(idx) if callable(labels) else str(labels[idx])
    except Exception:
        return f"idx{idx}"


def make_scorer(model_name: str = "tf_efficientnet_b0.ns_jft_in1k",
                device: torch.device | None = None):
    """바이트 → fish_prob 함수를 돌려준다.

    crawl_search가 **다운로드 중 실시간으로** 쓰레기를 걸러내는 데 사용한다.
    Bing이 간헐적으로 무관한 이미지를 뿌리는 걸 막을 방법이 없으니, 받는 즉시 버린다.
    (2026-08-13: '우럭 조황' 검색에 필리핀 경찰 로고가 섞여 들어왔다)
    """
    import io
    import threading

    device = device or torch.device("cpu")
    model, tf, _ = load_model(model_name, device)
    lock = threading.Lock()   # 다운로드 스레드 여러 개가 같은 모델을 쓴다

    @torch.no_grad()
    def score(buf: bytes) -> float:
        try:
            img = Image.open(io.BytesIO(buf)).convert("RGB")
        except Exception:
            return 0.0
        x = tf(img).unsqueeze(0).to(device)
        with lock:
            probs = model(x).softmax(dim=-1)[0]
        return float(probs[list(FISH_IDX)].sum())

    return score


@torch.no_grad()
def score_dir(directory: Path, model, tf, device, batch_size: int, include_inat: bool):
    """(경로, fish_prob, food_prob, text_prob, top1_idx, top1_prob) 목록.

    기본적으로 `web_*`(검색 크롤링분)만 채점한다. iNat 사진은 학명으로 검증된 것이라
    걸러낼 이유가 없고, 실제로 수중·흐린 사진에서 fish_prob가 낮게 나와 오탐이 많다
    (돌돔 수중 사진 top-1이 'sea urchin'으로 잡히는 식).
    """
    paths = [p for p in sorted(directory.iterdir())
             if p.is_file() and p.suffix.lower() in config.VALID_IMAGE_EXTS] \
        if directory.is_dir() else []
    if not include_inat:
        paths = [p for p in paths if not p.name.startswith("inat_")]

    out = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        tensors, ok_paths = [], []
        for p in chunk:
            try:
                with Image.open(p) as img:
                    tensors.append(tf(img.convert("RGB")))
                ok_paths.append(p)
            except Exception:
                out.append((p, 0.0, 0.0, 0.0, -1, 0.0))  # 열리지 않으면 의심 처리
        if not tensors:
            continue
        x = torch.stack(tensors).to(device)
        probs = model(x).softmax(dim=-1).cpu()
        fish = probs[:, list(FISH_IDX)].sum(dim=1)
        food = probs[:, list(FOOD_IDX)].sum(dim=1)
        text = probs[:, list(TEXT_IDX)].sum(dim=1)
        top1_prob, top1_idx = probs.max(dim=1)
        for j, p in enumerate(ok_paths):
            out.append((p, float(fish[j]), float(food[j]), float(text[j]),
                        int(top1_idx[j]), float(top1_prob[j])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="사전학습 모델로 비물고기 후보 걸러내기")
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--model", default="tf_efficientnet_b0.ns_jft_in1k",
                    help="timm 모델명 (기본: 가볍고 노이즈에 강한 b0 NS 가중치)")
    # 임계 0.15의 근거 (2026-08-13 실측):
    #   정상 낚시 사진 12장 fish_prob = 0.56 ~ 0.94 (중간 0.85, 최소 0.5575)
    #   쓰로틀링으로 섞여든 무관 이미지 = 0.0003 ~ 0.08
    #   → 두 분포 사이가 텅 비어 있다. 0.15는 정상 최소값의 1/3.7 지점.
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="fish_prob가 이 값 미만이면 '의심'. 정상 사진은 0.5 이상 나온다 (기본 0.15)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--apply", action="store_true",
                    help="의심 파일을 data/reject/<종>/ 으로 이동 (기본은 리포트만)")
    ap.add_argument("--strict", action="store_true",
                    help="썸네일(overlay?)·요리(food?) 후보까지 함께 이동")
    ap.add_argument("--include-inat", action="store_true",
                    help="iNat 사진도 채점한다. 기본 제외 — 학명 검증본인데 수중 사진이 오탐된다")
    ap.add_argument("--include-other", action="store_true",
                    help="'기타' 클래스도 점수를 매긴다(일부러 비물고기를 모은 클래스라 기본 제외)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    targets = resolve_species(args.species)
    if not args.include_other:
        targets = [t for t in targets if t != config.OTHER_CLASS]

    device = torch.device(args.device)
    print(f"[env] device={device} | model={args.model}")
    model, tf, labels = load_model(args.model, device)
    print(f"[cfg] {len(targets)}클래스 | 임계 fish_prob < {args.threshold} | "
          f"{'APPLY(이동)' if args.apply else '리포트만'}\n")

    rows: list[dict] = []
    moved: list[tuple[str, str]] = []
    print(f"{'종명':<8} {'검사':>6} {'의심':>6} {'썸네일':>7} {'요리':>6}")
    print("-" * 38)

    for name in targets:
        scored = score_dir(config.CLEAN_DIR / name, model, tf, device,
                           args.batch_size, args.include_inat)
        n_sus = n_text = n_food = 0
        for p, fish, food, text, top1, top1p in scored:
            suspicious = fish < args.threshold
            text_like = text > 0.20
            food_like = food > 0.15
            n_sus += suspicious
            n_text += text_like
            n_food += food_like
            verdict = ("suspicious" if suspicious else
                       "overlay?" if text_like else
                       "food?" if food_like else "keep")
            rows.append({
                "species": name, "filename": p.name, "verdict": verdict,
                "fish_prob": round(fish, 4), "food_prob": round(food, 4),
                "text_prob": round(text, 4),
                "top1": describe(labels, top1) if top1 >= 0 else "unreadable",
                "top1_prob": round(top1p, 4),
            })
            move = suspicious or (args.strict and verdict in ("overlay?", "food?"))
            if move and args.apply:
                dst = REJECT_DIR / name / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                p.replace(dst)   # clean은 raw의 하드링크 → raw 원본은 그대로
                moved.append((name, p.name))
        print(f"{name:<8} {len(scored):>6} {n_sus:>6} {n_text:>7} {n_food:>6}")

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["species", "filename", "verdict", "fish_prob",
                                          "food_prob", "text_prob", "top1", "top1_prob"])
        w.writeheader()
        w.writerows(rows)

    n_sus_total = sum(r["verdict"] == "suspicious" for r in rows)
    print("-" * 38)
    print(f"[OK] 리포트: {REPORT_CSV}  (총 {len(rows)}장, 의심 {n_sus_total}장)")
    if args.apply:
        # dedup 재실행 때 되살아나지 않도록 거부 목록에 남긴다
        added = append_rejected(moved)
        print(f"[OK] {len(moved)}장을 {REJECT_DIR} 로 이동 (거부 목록 +{added})")
        print("     오탐이면 reject에서 clean으로 되돌리고 data/rejected.txt 에서 해당 줄을 지울 것")
    else:
        print("[next] CSV를 fish_prob 오름차순으로 정렬해 눈으로 확인 → 임계값 조정 → --apply")
    print("[next] 사람 검수 후: python -m scripts.split")


if __name__ == "__main__":
    main()
