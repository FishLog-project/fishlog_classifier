"""네이버 검색 API로 실사용 낚시 사진 수집 — 한국 낚시 사진의 주 소스.

왜 네이버인가 (2026-08-14 비교 실측):
  - Google  : icrawler 파서가 깨져 0장 (JS 렌더링 + 스크래퍼 차단)
  - Baidu   : 작동하지만 중국 사이트 결과 → 한국 낚시 사진이 아님
  - Bing    : 한국 결과를 주지만 **키워드당 20~60장이면 고갈**되고, 간헐적으로
              검색어와 무관한 이미지를 조용히 섞는다 (한 실행분 76%가 쓰레기)
  - 네이버  : 공식 API. **쿼리당 최대 1,000장**, 한국 낚시인이 블로그·카페에 올린
              실사용 사진이 그대로 나온다. 무료 25,000회/일.

준비: developers.naver.com 에서 '검색 > 이미지' 앱 등록 후 키를 환경변수로 둔다.
      프로젝트 루트 `.env` 에 적어두면 자동으로 읽는다 (커밋 금지 — .gitignore 처리됨):
          NAVER_CLIENT_ID=...
          NAVER_CLIENT_SECRET=...

사용 예:
    python -m scripts.crawl_naver --species all
    python -m scripts.crawl_naver --species 우럭 삼치 쏘가리 --limit 350
    python -m scripts.crawl_naver --species other        # '기타' 4버킷
    python -m scripts.crawl_naver --dry-run              # 쿼리 배분만 확인

파일명은 Bing 수집분과 같은 `web_<url해시>.jpg` 규칙이다 → dedup/prefilter/split이
소스를 구분하지 않고 동일하게 처리하고, 두 소스에 같은 사진이 있으면 자동으로 건너뛴다.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src import config
from scripts._common import (
    LicenseLog,
    RateLimiter,
    count_usable,
    download_from_pages,
    load_rejected,
    make_session,
    resolve_species,
)
from scripts.crawl_search import keywords_for, other_class_plan

API = "https://openapi.naver.com/v1/search/image.json"
MAX_DISPLAY = 100      # API 상한
MAX_START = 1000       # start + display - 1 <= 1000


def load_env_file(path: Path | None = None) -> None:
    """`.env` 의 KEY=VALUE 를 환경변수로 올린다 (python-dotenv 의존성 없이).

    이미 환경에 있는 값은 덮어쓰지 않는다.
    """
    path = path or (config.PROJECT_ROOT / ".env")
    if not path.exists():
        return
    # utf-8-sig: 메모장·PowerShell `Set-Content -Encoding utf8` 이 BOM을 붙인다.
    # BOM을 안 벗기면 첫 줄 키가 '﻿NAVER_CLIENT_ID' 가 되어 조용히 인식되지 않는다.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def get_credentials() -> tuple[str, str]:
    load_env_file()
    cid = os.environ.get("NAVER_CLIENT_ID", "")
    secret = os.environ.get("NAVER_CLIENT_SECRET", "")
    if not cid or not secret:
        raise SystemExit(
            "[warn] 네이버 API 키가 없다.\n"
            "  1) https://developers.naver.com/apps/#/register 에서 '검색 > 이미지' 앱 등록\n"
            "  2) 프로젝트 루트에 .env 파일을 만들고 아래 두 줄을 적는다:\n"
            "       NAVER_CLIENT_ID=발급받은_ID\n"
            "       NAVER_CLIENT_SECRET=발급받은_SECRET"
        )
    return cid, secret


def search_page(session, cid: str, secret: str, query: str, start: int,
                display: int, sort: str, limiter: RateLimiter) -> list[dict]:
    """이미지 검색 한 페이지. 실패하면 빈 리스트."""
    limiter.wait()
    try:
        r = session.get(
            API,
            params={"query": query, "display": display, "start": start, "sort": sort},
            headers={"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": secret},
            timeout=20,
        )
    except Exception as e:
        print(f"      [warn] 요청 실패: {type(e).__name__}")
        return []
    if r.status_code == 429:
        print("      [warn] 일일 호출 한도 초과(25,000회). 내일 다시 시도.")
        return []
    if r.status_code != 200:
        print(f"      [warn] HTTP {r.status_code}: {r.text[:120]}")
        return []
    return r.json().get("items", [])


def naver_pages(session, cid: str, secret: str, query: str, sort: str,
                limiter: RateLimiter):
    """(url, width, height) 리스트를 페이지 단위로 내놓는다. 최대 1,000위까지."""
    start = 1
    while start <= MAX_START:
        display = min(MAX_DISPLAY, MAX_START - start + 1)
        items = search_page(session, cid, secret, query, start, display, sort, limiter)
        if not items:
            return
        page = []
        for it in items:
            try:
                w, h = int(it.get("sizewidth") or 0), int(it.get("sizeheight") or 0)
            except ValueError:
                w = h = 0
            page.append((it.get("link", ""), w, h))
        yield page
        start += display


def main() -> None:
    ap = argparse.ArgumentParser(
        description="네이버 검색 API로 실사용 낚시 사진 수집 (+ licenses.csv 기록)")
    ap.add_argument("--species", nargs="*", default=None,
                    help="종명 또는 별칭(all/fish/other/confusable/easy/medium/hard/sea/fresh)")
    ap.add_argument("--limit", type=int, default=config.MVP_TARGET_IMAGES,
                    help=f"종당 목표 총 장수 (기본 MVP {config.MVP_TARGET_IMAGES})")
    ap.add_argument("--use-final-target", action="store_true",
                    help="난이도별 최종 목표량(easy 350/medium 600/hard 1000) 사용")
    ap.add_argument("--min-web", type=int, default=100,
                    help="클래스당 최소 검색 사진 수. iNat로 목표를 채운 클래스도 실사용 "
                         "사진을 이만큼 확보한다 (기본 100)")
    ap.add_argument("--extra-keywords", nargs="*", default=None,
                    help="지정한 종에 추가할 쿼리 (--species를 1개만 줄 때 유용)")
    ap.add_argument("--sort", default="sim", choices=("sim", "date"),
                    help="sim=정확도순(권장) / date=최신순")
    ap.add_argument("--min-side", type=int, default=320,
                    help="이미지 짧은 변 최소 픽셀 (기본 320)")
    ap.add_argument("--score-min", type=float, default=0.15,
                    help="다운로드 즉시 fish_prob가 이 값 미만이면 버린다. 0이면 검사 안 함")
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="API 호출 간 최소 간격(초). 한도는 25,000회/일 (기본 0.3)")
    ap.add_argument("--dry-run", action="store_true", help="쿼리 배분만 출력")
    args = ap.parse_args()

    cid, secret = ("", "") if args.dry_run else get_credentials()
    targets = resolve_species(args.species)
    rejected = load_rejected()

    # 물고기 판정기 — Bing만큼은 아니어도 검색 결과에 요리·풍경·인물이 섞인다.
    extra_check = None
    if args.score_min > 0 and not args.dry_run:
        from scripts.prefilter import make_scorer
        print("[env] 실시간 fish_prob 검사기 로딩 중…", flush=True)
        scorer = make_scorer()

        def extra_check(buf: bytes):  # noqa: F811  (download_image에 넘길 판정 함수)
            s = scorer(buf)
            return (s >= args.score_min, f"low_fish_{s:.3f}")

    print(f"[cfg] 대상 {len(targets)}클래스 | 종당 목표 "
          f"{'난이도별 최종' if args.use_final_target else args.limit} | "
          f"sort={args.sort} | score-min {args.score_min if extra_check else 'off'} | "
          f"거부목록 {len(rejected)}건")

    limiter = RateLimiter(args.sleep)
    session = make_session()
    total = 0

    with LicenseLog() as log:
        for name in targets:
            limit = config.SPECIES[name].target_images if args.use_final_target else args.limit
            have = count_usable(name, rejected)
            have_web = count_usable(name, rejected, prefix="web_")
            need = max(limit - have, args.min_web - have_web, 0)
            stop_at = have + need
            print(f"\n[data] {name}  보유 {have}(web {have_web}) / 목표 {limit}, "
                  f"web 최소 {args.min_web}  → 수집 {need}", flush=True)
            if need == 0:
                print("  이미 목표 달성 — 건너뜀")
                continue

            if name == config.OTHER_CLASS:
                plan = other_class_plan(limit)
            else:
                queries = keywords_for(name, args.extra_keywords)
                per_q = max(10, -(-need // len(queries)))
                plan = [(q, per_q) for q in queries]

            print("  쿼리 배분: " + ", ".join(f"{q}({n})" for q, n in plan))
            if args.dry_run:
                continue

            for query, want in plan:
                if count_usable(name, rejected) >= stop_at:
                    print("  목표 도달 — 남은 쿼리 생략")
                    break
                total += download_from_pages(
                    session, name, query, want,
                    naver_pages(session, cid, secret, query, args.sort, limiter),
                    source="naver", log=log, rejected=rejected,
                    # '기타'는 일부러 비물고기를 모으는 클래스 → 물고기 점수 검사 제외
                    extra_check=None if name == config.OTHER_CLASS else extra_check,
                    min_side=args.min_side)
            print(f"  [OK] {name}: 총 {count_usable(name, rejected)}장 "
                  f"(web {count_usable(name, rejected, prefix='web_')})")

    print(f"\n[done] 신규 저장 {total}장")
    print("[next] python -m scripts.dedup  →  python -m scripts.prefilter")


if __name__ == "__main__":
    main()
