"""실사용 사진을 배포된 서버에 던져보고 결과를 표로 본다.

val/test 성적은 **웹 수집 사진** 기준이다. 실제 앱이 받는 사진(손에 든 물고기,
젖은 바닥, 역광, 뜰채 안)은 분포가 다르므로, 그 차이는 이렇게 직접 넣어봐야 안다.
로드맵의 마지막 미검증 항목이 이것이다.

EC2에 붙으려면 SSH 터널을 먼저 연다(보안그룹은 그대로 두고 8000을 빌려온다):

    ssh -i <키>.pem -N -L 8000:localhost:8000 ubuntu@<모델서버 공인IP>

사용 예:
    python -m scripts.try_photos 사진폴더
    python -m scripts.try_photos a.jpg b.jpg --url http://localhost:8000
    python -m scripts.try_photos 사진폴더 --html reports/photo_test.html

파일명 앞에 어종을 적어두면(`붕어_01.jpg`) 정답으로 인식해 맞았는지까지 표시한다.
물고기가 아닌 사진은 `기타_01.jpg` 로 두면 된다 — `기타`는 후보에 나오지 않는 것이
설계이므로, uncertain 을 띄웠는지로 채점한다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx

from src import config

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic")


def guess_truth(name: str) -> str | None:
    """파일명에서 어종을 추론한다. `붕어_01.jpg`, `01_붕어.jpg` 둘 다 인식."""
    stem = Path(name).stem
    for cls in config.CLASSES:
        if cls in stem:
            return cls
    return None


def collect(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out += [q for q in sorted(p.iterdir()) if q.suffix.lower() in EXTS]
        elif p.suffix.lower() in EXTS:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="실사용 사진으로 배포 서버 확인")
    ap.add_argument("paths", nargs="+", type=Path, help="이미지 파일 또는 폴더")
    ap.add_argument("--url", default="http://localhost:8000", help="서버 주소")
    ap.add_argument("--html", type=Path, default=None, help="사진과 함께 HTML 리포트 저장")
    args = ap.parse_args()

    files = collect(args.paths)
    if not files:
        raise SystemExit("[FAIL] 이미지를 못 찾았다 (지원: " + " ".join(EXTS) + ")")

    with httpx.Client(base_url=args.url, timeout=30) as c:
        try:
            health = c.get("/health").json()
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"[FAIL] {args.url} 에 연결할 수 없다: {exc}\n"
                "  → EC2라면 SSH 터널을 먼저 열 것:\n"
                "     ssh -i <키>.pem -N -L 8000:localhost:8000 ubuntu@<공인IP>"
            ) from exc
        print(f"[server] {args.url} | 모델 {health.get('model_version')} | "
              f"임계값 {health.get('confidence_threshold')} | "
              f"TTA {'on' if health.get('tta') else 'off'}")
        print(f"[cfg] 사진 {len(files)}장\n")

        rows = []
        for path in files:
            t0 = time.perf_counter()
            try:
                r = c.post("/predict", files={"file": (path.name, path.read_bytes())})
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERR ] {path.name}: {exc}")
                continue
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                print(f"  [{r.status_code}] {path.name}: {body.get('error')} {body.get('detail', '')}")
                continue
            res = r.json()
            rows.append((path, res, ms))

    if not rows:
        raise SystemExit("[FAIL] 성공한 요청이 없다")

    # ---- 표 ----
    print(f"{'파일':<24} {'1순위':<18} {'2순위':<14} {'3순위':<14} "
          f"{'기타':>5} {'판정':<9} {'ms':>5}")
    print("-" * 100)
    n_unc = n_truth = n_top1 = n_top3 = 0
    for path, res, ms in rows:
        p = res["predictions"]
        truth = guess_truth(path.name)
        cands = [x["species"] for x in p]
        mark = ""
        if truth == config.OTHER_CLASS:
            # `기타`는 설계상 후보에 나오지 않는다 — 맞았다는 것은 uncertain 을 띄웠다는 뜻.
            # 후보 목록으로 채점하면 항상 오답이 되어 지표가 거짓말을 한다.
            n_truth += 1
            n_top1 += res["uncertain"]
            n_top3 += res["uncertain"]
            mark = "O" if res["uncertain"] else "X"
        elif truth:
            n_truth += 1
            n_top1 += truth == cands[0]
            n_top3 += truth in cands
            mark = "O" if truth in cands else "X"
        n_unc += res["uncertain"]
        print(f"{path.name[:23]:<24} "
              f"{p[0]['species'] + ' ' + format(p[0]['confidence'], '.2f'):<18} "
              f"{p[1]['species'] + ' ' + format(p[1]['confidence'], '.2f'):<14} "
              f"{p[2]['species'] + ' ' + format(p[2]['confidence'], '.2f'):<14} "
              f"{res['other_confidence']:>5.2f} "
              f"{('재촬영권유' if res['uncertain'] else '정상'):<9} {ms:>5.0f} {mark}")

    n = len(rows)
    print(f"\n[요약] {n}장 | uncertain {n_unc}장 ({n_unc / n:.0%}) | "
          f"평균 {sum(r[2] for r in rows) / n:.0f}ms")
    if n_truth:
        print(f"[정답 대조] 파일명에서 어종을 읽은 {n_truth}장 — "
              f"Top-1 {n_top1}/{n_truth} ({n_top1 / n_truth:.0%}) | "
              f"Top-3 {n_top3}/{n_truth} ({n_top3 / n_truth:.0%})")
        print("       참고: val 기준 Top-3 91%, uncertain 8.5% — 여기서 크게 낮으면 "
              "웹 사진과 실사용 사진의 도메인 갭이다")
    else:
        print("[안내] 파일명에 어종을 넣으면(`붕어_01.jpg`) 정답 대조까지 해준다")

    if args.html:
        write_html(rows, args.html)
        print(f"[OK] 리포트: {args.html.resolve()}")


def write_html(rows, out: Path) -> None:
    """사진을 눈으로 보면서 판단할 수 있게 썸네일과 함께 남긴다."""
    import base64

    out.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for path, res, ms in rows:
        b64 = base64.b64encode(path.read_bytes()).decode()
        cand = "".join(
            f"<li><b>{x['species']}</b> {x['confidence']:.0%}</li>" for x in res["predictions"])
        badge = ("<span class=u>재촬영 권유</span>" if res["uncertain"]
                 else "<span class=o>정상</span>")
        cards.append(f"""<div class=card>
  <img src="data:image/jpeg;base64,{b64}">
  <div><div class=name>{path.name}</div>{badge}
  <ol>{cand}</ol>
  <div class=meta>기타 {res['other_confidence']:.0%} · {ms:.0f}ms</div></div></div>""")
    out.write_text(f"""<!doctype html><meta charset=utf-8><title>실사용 사진 테스트</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#faf9f7;color:#1c1a17}}
.card{{display:flex;gap:16px;background:#fff;border:1px solid #e5e2dd;border-radius:10px;
padding:14px;margin-bottom:12px;align-items:center}}
img{{width:180px;height:180px;object-fit:cover;border-radius:8px}}
.name{{font-weight:600;margin-bottom:6px}}
ol{{margin:6px 0}} .meta{{color:#6b6560;font-size:13px}}
.u{{background:#fde8c8;color:#8a5a00;padding:2px 8px;border-radius:99px;font-size:12px}}
.o{{background:#d8f0d8;color:#1f5c1f;padding:2px 8px;border-radius:99px;font-size:12px}}
</style><h1>실사용 사진 테스트 ({len(rows)}장)</h1>{''.join(cards)}""", encoding="utf-8")


if __name__ == "__main__":
    main()
