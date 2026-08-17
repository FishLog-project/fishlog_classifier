"""DuckDuckGo 이미지 검색 수집 — 인증 없이 한국 낚시 사진을 가장 많이 주는 소스.

2026-08-14 소스 비교 실측:
  | 소스     | 요청당 결과 | 한국계 도메인 | 인증 | 비고                              |
  |----------|-------------|---------------|------|-----------------------------------|
  | Google   | 0장         | -             | -    | icrawler 파서 깨짐 (JS 렌더링)    |
  | Baidu    | 동작        | 없음          | 없음 | 중국 사이트 결과                  |
  | Bing     | 28~35장     | 중간          | 없음 | 키워드당 20~60장이면 고갈         |
  | 네이버   | -           | -             | 키   | 신규 앱에 '검색' API가 안 보임    |
  | **DDG**  | **100장**   | **55~65%**    | 없음 | next 토큰으로 계속 페이징 가능    |

동작 방식: `ddgs` 라이브러리를 쓴다. 내부 엔드포인트(i.js)를 직접 호출해봤으나
토큰(vqd)을 정상적으로 받아도 403이 잦았다 — 라이브러리가 백엔드 전환·토큰 처리를
대신 해준다. 공식 API가 아니므로 언젠가 깨질 수 있고, 그때는 `pip install -U ddgs`.

Bing 수집분과 파일명 규칙(`web_<url해시>.jpg`)이 같아서 dedup/prefilter/split이 소스를
구분하지 않고 처리하고, 두 소스가 같은 사진을 줘도 자동으로 한 장만 남는다.

사용 예:
    python -m scripts.crawl_ddg --species all
    python -m scripts.crawl_ddg --species 삼치 쏘가리 --limit 250
    python -m scripts.crawl_ddg --species other          # '기타' 4버킷
    python -m scripts.crawl_ddg --dry-run
"""

from __future__ import annotations

import argparse

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

REGION = "kr-kr"   # 한국어 결과를 우선한다

# 이미지 **다운로드**용 헤더. 한국 블로그 CDN 일부가 비브라우저 UA를 막는다.
# (검색 자체는 ddgs가 자기 세션으로 처리한다)
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.5",
}

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    raise SystemExit("ddgs가 없다 → pip install -r requirements.txt")


def _int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0     # 크기를 안 주는 결과가 있다 → 다운로드 후 검증한다


def ddg_pages(query: str, limiter: RateLimiter, max_results: int, verbose: bool):
    """(url, width, height) 리스트를 한 덩어리로 내놓는다.

    `download_from_pages`가 기대하는 '페이지 이터러블' 형태를 맞춘다.
    ddgs가 내부적으로 페이징하므로 여기서는 한 번에 max_results 개를 받는다.
    """
    limiter.wait()
    try:
        with DDGS() as ddgs:
            results = ddgs.images(query, region=REGION, max_results=max_results)
    except Exception as e:
        print(f"      [warn] 검색 실패: {type(e).__name__}: {str(e)[:100]}")
        return
    if verbose:
        print(f"      검색 결과 {len(results)}개")
    if results:
        yield [(r.get("image", ""), _int(r.get("width")), _int(r.get("height")))
               for r in results]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DuckDuckGo 이미지 검색으로 실사용 낚시 사진 수집 (인증 불필요)")
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
    ap.add_argument("--max-results", type=int, default=200,
                    help="쿼리당 최대 검색 결과 수. 목표보다 넉넉히 잡아야 점수 탈락분을 "
                         "메운다 (기본 200)")
    ap.add_argument("--min-side", type=int, default=320,
                    help="이미지 짧은 변 최소 픽셀 (기본 320)")
    ap.add_argument("--score-min", type=float, default=0.15,
                    help="다운로드 즉시 fish_prob가 이 값 미만이면 버린다. 0이면 검사 안 함. "
                         "검색 결과에 섞이는 요리·풍경·인물 사진을 여기서 막는다")
    ap.add_argument("--sleep", type=float, default=1.5,
                    help="요청 간 최소 간격(초). 너무 빠르면 차단된다 (기본 1.5)")
    ap.add_argument("--verbose", action="store_true", help="페이지별 결과 수 출력")
    ap.add_argument("--dry-run", action="store_true", help="쿼리 배분만 출력")
    args = ap.parse_args()

    targets = resolve_species(args.species)
    rejected = load_rejected()

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
          f"score-min {args.score_min if extra_check else 'off'} | "
          f"거부목록 {len(rejected)}건")

    limiter = RateLimiter(args.sleep)
    session = make_session()
    session.headers.update(BROWSER_HEADERS)
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
                    ddg_pages(query, limiter, args.max_results, args.verbose),
                    source="ddg", log=log, rejected=rejected,
                    # '기타'는 일부러 비물고기를 모으는 클래스 → 물고기 점수 검사 제외
                    extra_check=None if name == config.OTHER_CLASS else extra_check,
                    min_side=args.min_side)
            print(f"  [OK] {name}: 총 {count_usable(name, rejected)}장 "
                  f"(web {count_usable(name, rejected, prefix='web_')})")

    print(f"\n[done] 신규 저장 {total}장")
    print("[next] python -m scripts.dedup  →  python -m scripts.prefilter")


if __name__ == "__main__":
    main()
