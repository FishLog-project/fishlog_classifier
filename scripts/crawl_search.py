"""검색엔진(Bing) 이미지 크롤링 — 실사용 유사 사진 + `기타` 클래스 채우기.

왜 필요한가: iNaturalist 사진은 대부분 **수중·표본·도감** 사진이다. 앱 사용자는
**손에 든/바닥에 놓인/젖은/역광** 사진을 찍는다. 이 도메인 갭을 메우지 않으면
val 정확도는 높은데 실제 앱에서 틀리는 모델이 나온다.
반대로 라벨 오염이 심하다 → **사람 검수(5단계)가 필수**다.

파일명 규칙: `web_<url해시12>.jpg`
  - URL 해시라서 같은 사진을 다른 키워드로 다시 만나도 덮어쓰지 않고 건너뛴다(재실행 안전).
  - 내용이 같고 URL이 다른 중복은 dedup.py(phash)가 잡는다.

사용 예:
    python -m scripts.crawl_search                        # 25클래스 전부, MVP 목표까지
    python -m scripts.crawl_search --species 볼락 방어 --limit 250
    python -m scripts.crawl_search --species other        # '기타' 4버킷만
    python -m scripts.crawl_search --species 붕어 --extra-keywords "붕어 손" "붕어 조과"
    python -m scripts.crawl_search --dry-run              # 키워드·배분만 출력

Bing은 키워드당 최대 1,000장까지만 준다. 부족하면 키워드를 늘리는 게 정답이다.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time

from src import config
from scripts._common import (
    LicenseLog,
    LicenseRow,
    count_usable as _count,
    load_rejected,
    resolve_species,
    valid_image_bytes,
)

try:
    from icrawler import ImageDownloader
    from icrawler.builtin import BingImageCrawler
except ImportError:  # pragma: no cover
    raise SystemExit("icrawler가 없다 → pip install -r requirements.txt")

# 한국 낚시 사진을 원하므로 한국어 결과를 우선하게 만든다.
# (icrawler 기본 헤더는 Accept-Language가 zh-CN이라 중국어권 결과가 섞인다.)
KO_HEADERS = {
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.5,en;q=0.3",
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
}

# 실사용 분포에 가까운 사진을 끌어오는 접미사. 종 키워드에 붙여 쓴다.
# ("손", "조과" = 잡아서 들고 찍은 사진이 많이 걸린다)
REALWORLD_SUFFIXES = ("조과", "손", "잡은")


class RecordingDownloader(ImageDownloader):
    """URL 해시 파일명 + licenses.csv 기록 + 이미지 품질 검증을 붙인 다운로더."""

    # 크롤러 생성 후 주입한다 (icrawler가 ctor 인자를 그대로 넘겨주지만,
    # 속성 주입이 버전 변화에 덜 민감하다)
    license_log: LicenseLog | None = None
    species: str = ""
    keyword: str = ""
    scorer = None          # callable(bytes) -> fish_prob. None이면 점수 검사 안 함
    score_min: float = 0.0

    def get_filename(self, task, default_ext):
        h = hashlib.sha1(task["file_url"].encode("utf-8")).hexdigest()[:12]
        return f"web_{h}.jpg"

    def keep_file(self, task, response, min_size=None, max_size=None):
        # icrawler 기본 검사(min_size)보다 엄격하게: 바이트 수·디코딩·극단 비율까지 본다.
        ok, _why = valid_image_bytes(response.content)
        if not ok:
            return False
        if self.scorer is not None:
            # 물고기가 없는 이미지는 저장하지 않는다. False를 돌려주면 icrawler가
            # 할당량(max_num)도 소비하지 않으므로, 그 자리를 정상 사진으로 다시 채운다.
            if self.scorer(response.content) < self.score_min:
                return False
        return True

    def process_meta(self, task):
        if not task.get("success") or not task.get("filename") or self.license_log is None:
            return
        self.license_log.add(LicenseRow(
            filename=task["filename"],
            species=self.species,
            source="bing",
            source_id=self.keyword,      # 어떤 키워드로 걸렸는지 = 검수 단서
            license="unknown",           # 검색 크롤링분은 항상 unknown (decisions C-8)
            url=task["file_url"],
        ))


def crawl_keyword(keyword: str, species: str, want: int, *, log: LicenseLog,
                  threads: int, verbose: bool, rejected: set[tuple[str, str]],
                  offset: int = 0, scorer=None, score_min: float = 0.0) -> int:
    """키워드 하나로 `want`장 목표 수집. 저장 폴더는 data/raw/<species>.

    `offset`은 Bing 검색 결과의 시작 위치다. 같은 키워드를 다시 긁을 때 뒤쪽 결과를
    보게 해서 중복을 피한다 (Bing은 키워드당 1,000번째까지만 준다).
    """
    out_dir = config.RAW_DIR / species
    out_dir.mkdir(parents=True, exist_ok=True)
    before = _count(species, rejected)

    crawler = BingImageCrawler(
        downloader_cls=RecordingDownloader,
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=threads,
        storage={"root_dir": str(out_dir)},
        # 죽은 링크(404/403)가 흔해서 ERROR 레벨이면 로그가 그것만 가득 찬다.
        # 어차피 실패한 건 안 세니 --verbose 때만 보여준다.
        log_level=logging.INFO if verbose else logging.CRITICAL,
    )
    crawler.set_session(KO_HEADERS)
    crawler.downloader.license_log = log
    crawler.downloader.species = species
    crawler.downloader.keyword = keyword
    crawler.downloader.scorer = scorer
    crawler.downloader.score_min = score_min

    # max_num은 '성공 저장' 개수 기준이다(검증 탈락분은 세지 않는다) → 여유분을 더하면
    # 그만큼 초과 수집돼 키워드별 예산이 무너진다. want를 그대로 준다. (Bing 상한 1000)
    crawler.crawl(
        keyword=keyword,
        offset=offset,
        max_num=min(1000 - offset, want),
        min_size=(320, 320),        # 폰카 사진 기준. 너무 작으면 224 학습에 흐릿하다
        filters={"type": "photo"},  # 일러스트·클립아트 제외 (도감 그림이 많이 섞인다)
        overwrite=False,
        max_idle_time=45,
    )
    log.flush()
    got = _count(species, rejected) - before
    print(f"    '{keyword}'{f' @{offset}' if offset else ''} → +{got}장", flush=True)
    return got


def keywords_for(species: str, extra: list[str] | None) -> list[str]:
    """종별 검색 키워드 목록. config를 1차 소스로 쓰고 실사용 접미사를 덧붙인다."""
    sp = config.SPECIES[species]
    kws: list[str] = list(sp.keywords)
    # 이명은 한글만 쓴다 (학명 이명으로 검색하면 도감/논문 그림이 걸린다)
    kws += [a for a in sp.aliases if a and not a.isascii()]
    kws += [f"{species} {suf}" for suf in REALWORLD_SUFFIXES]
    kws += list(extra or [])

    seen, uniq = set(), []
    for k in kws:
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def bucket_of(keyword: str) -> str | None:
    for bucket, kws in config.OTHER_KEYWORDS.items():
        if keyword in kws:
            return bucket
    return None


def other_bucket_counts() -> dict[str, int]:
    """`기타`의 버킷별 현재 보유량.

    - iNat 사진(`inat_*`)은 전부 other_fish 버킷이다 (crawl_inat이 학명으로 긁은 것).
    - 검색 사진은 licenses.csv의 `source_id`(수집에 쓴 키워드)로 버킷을 되짚는다.
    """
    counts = {b: 0 for b in config.OTHER_BUCKET_RATIO}
    csv_path = config.DATA_DIR / "licenses.csv"
    if csv_path.exists():
        import csv as _csv
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in _csv.DictReader(f):
                if row.get("species") != config.OTHER_CLASS:
                    continue
                if row.get("source") == "inaturalist":
                    counts["other_fish"] += 1
                else:
                    b = bucket_of(row.get("source_id", ""))
                    if b:
                        counts[b] += 1
    return counts


def other_class_plan(limit: int) -> list[tuple[str, int]]:
    """`기타` 클래스: 4버킷 비중대로, **이미 채운 만큼을 빼고** 키워드별 목표량을 배분한다.

    iNat로 채운 몫(other_fish)을 빼지 않으면 그 버킷만 두 번 채워지고, 목표량에 먼저 도달해
    '일반 오촬영' 버킷이 0장으로 남는다 → 고양이 사진을 못 거르는 모델이 된다.
    또 버킷을 **번갈아(round-robin)** 배치해, 중간에 목표에 도달해도 모든 버킷이 섞이게 한다.
    """
    have = other_bucket_counts()
    per_bucket: dict[str, list[tuple[str, int]]] = {}
    for bucket, ratio in config.OTHER_BUCKET_RATIO.items():
        kws = config.OTHER_KEYWORDS[bucket]
        deficit = max(0, int(limit * ratio) - have.get(bucket, 0))
        if deficit == 0:
            continue
        per_kw = max(5, -(-deficit // len(kws)))  # 올림
        per_bucket[bucket] = [(k, per_kw) for k in kws]

    # 버킷을 번갈아 뽑는다
    plan: list[tuple[str, int]] = []
    for i in range(max((len(v) for v in per_bucket.values()), default=0)):
        for items in per_bucket.values():
            if i < len(items):
                plan.append(items[i])
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bing 이미지 크롤링으로 실사용 유사 사진 수집 (+ licenses.csv 기록)")
    ap.add_argument("--species", nargs="*", default=None,
                    help="종명 또는 별칭(all/fish/other/confusable/easy/medium/hard/sea/fresh)")
    ap.add_argument("--limit", type=int, default=config.MVP_TARGET_IMAGES,
                    help=f"종당 목표 총 장수(iNat 수집분 포함). 기본 {config.MVP_TARGET_IMAGES}")
    ap.add_argument("--use-final-target", action="store_true",
                    help="난이도별 최종 목표량 사용")
    ap.add_argument("--min-web", type=int, default=100,
                    help="클래스당 최소 검색 사진 수. iNat로 목표를 이미 채운 클래스도 "
                         "실사용 사진(손에 든/바닥/젖은)을 이만큼은 확보한다 (기본 100)")
    ap.add_argument("--extra-keywords", nargs="*", default=None,
                    help="지정한 종에 추가할 키워드 (--species를 1개만 줄 때 유용)")
    ap.add_argument("--max-passes", type=int, default=4,
                    help="목표 미달 시 offset을 밀며 키워드 목록을 다시 도는 최대 횟수 (기본 4)")
    ap.add_argument("--threads", type=int, default=4, help="다운로드 스레드 수 (기본 4)")
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="키워드 사이 대기 초 (기본 3). 예의상 두는 값이다 — 품질 방어는 "
                         "--score-min 이 한다")
    # Bing은 간헐적으로 검색어와 무관한 이미지를 뿌린다(에러 없이 조용히).
    # 2026-08-13 '우럭 조황' 검색에 필리핀 경찰 로고·선 드로잉·꽃 사진이 섞여 들어와
    # 그 실행분의 76%가 쓰레기였다. 스레드 수·대기 시간을 조절해도 막지 못했다.
    # → 막을 수 없으니 **받는 즉시 점수로 판정해 버린다**. 이게 유일하게 작동한 방어다.
    #    (실측: 정상 낚시 사진 0.56~0.94 / 무관 이미지 0.0003~0.08 → 사이가 텅 비어 있다)
    ap.add_argument("--score-min", type=float, default=0.15,
                    help="다운로드 즉시 fish_prob가 이 값 미만이면 버린다. "
                         "0이면 검사 안 함 (기본 0.15, prefilter와 같은 기준)")
    ap.add_argument("--verbose", action="store_true", help="icrawler 로그 전체 출력")
    ap.add_argument("--dry-run", action="store_true", help="키워드·배분만 출력")
    args = ap.parse_args()

    targets = resolve_species(args.species)
    rejected = load_rejected()

    # 실시간 품질 검사기. Bing이 간헐적으로 무관한 이미지를 뿌리므로 받는 즉시 판정한다.
    scorer = None
    if args.score_min > 0 and not args.dry_run:
        from scripts.prefilter import make_scorer
        print("[env] 실시간 fish_prob 검사기 로딩 중…", flush=True)
        scorer = make_scorer()

    print(f"[cfg] 대상 {len(targets)}클래스 | 종당 목표 "
          f"{'난이도별 최종' if args.use_final_target else args.limit} | "
          f"threads {args.threads} | 키워드 간 {args.sleep}초 | "
          f"score-min {args.score_min if scorer else 'off'} | 거부목록 {len(rejected)}건")

    total = 0
    with LicenseLog() as log:
        for name in targets:
            limit = config.SPECIES[name].target_images if args.use_final_target else args.limit
            have = _count(name, rejected)
            have_web = _count(name, rejected, prefix="web_")
            # iNat만으로 목표를 채운 클래스도 실사용 사진은 따로 확보해야 한다.
            # (iNat는 수중·표본 사진 위주라 앱 사진과 분포가 다르다 — 도메인 갭)
            need = max(limit - have, args.min_web - have_web, 0)
            stop_at = have + need   # 이 클래스에서 멈출 총 장수
            print(f"\n[data] {name}  보유 {have}(web {have_web}) / 목표 {limit}, "
                  f"web 최소 {args.min_web}  → 수집 {need}", flush=True)
            if need == 0:
                print("  이미 목표 달성 — 건너뜀")
                continue

            if name == config.OTHER_CLASS:
                plan = other_class_plan(limit)
            else:
                kws = keywords_for(name, args.extra_keywords)
                per_kw = max(10, -(-need // len(kws)))  # 올림 배분
                plan = [(k, per_kw) for k in kws]

            print("  키워드 배분: " + ", ".join(f"{k}({n})" for k, n in plan))
            if args.dry_run:
                continue

            # 키워드 목록을 한 번 돌아도 목표에 못 미치면, 검색 결과 뒤쪽(offset)을 보며
            # 다시 돈다. Bing은 같은 키워드로 앞쪽 결과만 반복해서 주므로 offset 없이
            # 재실행하면 전부 '이미 있는 파일'로 걸러져 한 장도 늘지 않는다.
            offset = 0
            for pass_i in range(args.max_passes):
                before_pass = _count(name, rejected)
                for keyword, want in plan:
                    if _count(name, rejected) >= stop_at:
                        break
                    # '기타'는 일부러 비물고기를 모으는 클래스 → 물고기 점수 검사 제외
                    total += crawl_keyword(
                        keyword, name, want, log=log,
                        threads=args.threads, verbose=args.verbose,
                        rejected=rejected, offset=offset,
                        scorer=None if name == config.OTHER_CLASS else scorer,
                        score_min=args.score_min)
                    time.sleep(args.sleep)   # Bing 쓰로틀링 회피
                gained = _count(name, rejected) - before_pass
                if _count(name, rejected) >= stop_at:
                    print("  목표 도달")
                    break
                if gained == 0:
                    print(f"  더 이상 새 결과 없음 (pass {pass_i + 1})")
                    break
                offset += 120
                if offset >= 1000:  # Bing 상한
                    print("  Bing 결과 상한(1000) 도달")
                    break
                print(f"  pass {pass_i + 1}: +{gained}장 → offset {offset} 로 재시도")
            print(f"  [OK] {name}: 총 {_count(name, rejected)}장 "
                  f"(web {_count(name, rejected, prefix='web_')})")

    # icrawler가 남기는 임시/부분 파일 정리
    for name in targets:
        for junk in (config.RAW_DIR / name).glob("*.part"):
            junk.unlink(missing_ok=True)

    print(f"\n[done] 신규 저장 {total}장")
    print("[warn] 검색 크롤링분은 라벨 오염이 있다. 반드시 dedup → prefilter → 사람 검수를 거칠 것")
    print("[next] python -m scripts.dedup")


if __name__ == "__main__":
    main()
