"""iNaturalist API 수집 — 학명 기반, 전문가 검증(research grade) 사진.

왜 여기부터인가: 라벨이 학명으로 붙어 있어 오염이 거의 없다. 검색 크롤링은
'붕어 낚시'로 검색해도 잉어 사진이 섞여 들어온다. 혼동 쌍(붕어/잉어,
우럭/볼락, 감성돔/벵에돔)은 반드시 이 소스로 먼저 채운다.

파일명 규칙: `inat_<observation_id>_<photo_id>.jpg`
  → split.py 가 obsid로 그룹핑해서 같은 개체 사진이 train/test로 쪼개지는 걸 막는다.
    (쪼개지면 test 정확도가 부풀려진다. data-pipeline.md "분할 규칙" 참조)

사용 예:
    python -m scripts.crawl_inat                      # 어종 24종, 종당 MVP 목표(250)까지
    python -m scripts.crawl_inat --species 붕어 잉어 --limit 400
    python -m scripts.crawl_inat --species confusable # 혼동 쌍만 집중 수집
    python -m scripts.crawl_inat --species other      # '기타' 클래스(24종 밖 어종)
    python -m scripts.crawl_inat --dry-run            # taxon 매칭만 확인 (다운로드 X)

재실행 안전: 이미 있는 파일은 건너뛴다. 중단되면 그냥 다시 돌리면 이어진다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from scripts._common import (
    LicenseLog,
    LicenseRow,
    RateLimiter,
    count_usable,
    download_image,
    load_rejected,
    make_session,
    resolve_species,
)

API = "https://api.inaturalist.org/v1"
PER_PAGE = 200  # API 최대값

# photo.url 은 정사각 썸네일(75px)을 준다. 파일명의 size 토큰만 갈아끼우면 원하는 크기.
# 'large'(1024px)는 용량이 크고 다운로드가 느리다 → 224px 학습엔 medium(500px)이 충분.
SIZE_TOKENS = ("square", "small", "medium", "large", "original")

# 학습에 쓸 수 있는 라이선스. all-rights-reserved(None)는 --any-license 없이는 제외.
OPEN_LICENSES = {"cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa",
                 "cc-by-nd", "cc-by-nc-nd"}


def photo_url(url: str, size: str = "medium") -> str:
    """썸네일 URL을 원하는 사이즈 URL로 바꾼다."""
    for token in SIZE_TOKENS:
        if f"/{token}." in url:
            return url.replace(f"/{token}.", f"/{size}.", 1)
    return url


def resolve_taxon(session, sci_name: str, limiter: RateLimiter) -> dict | None:
    """학명 → taxon 정보. 이름이 정확히 일치하는 것만 채택한다.

    부정확한 매칭을 그냥 쓰면 엉뚱한 종 사진을 수백 장 받는다 → 조용히 실패하지 말고
    None을 돌려 사람이 확인하게 한다.
    """
    limiter.wait()
    r = session.get(f"{API}/taxa", params={"q": sci_name, "per_page": 10}, timeout=20)
    if r.status_code != 200:
        print(f"  [warn] taxa 조회 실패 ({r.status_code}): {sci_name}")
        return None
    results = r.json().get("results", [])
    if not results:
        return None

    exact = [t for t in results if t.get("name", "").lower() == sci_name.lower()]
    if exact:
        return exact[0]
    # 동의어(synonym) 등으로 이름이 어긋난 경우 — 사람이 판단해야 한다.
    print(f"  [warn] 학명 정확 일치 없음: {sci_name} "
          f"→ 후보: {', '.join(t.get('name', '?') for t in results[:3])}")
    return None


def fetch_observations(session, taxon_id: int, limiter: RateLimiter, *,
                       need: int, max_per_obs: int, quality: str,
                       any_license: bool) -> list[dict]:
    """관측 목록을 페이지네이션으로 모은다. 사진 수가 `need`를 넘으면 멈춘다.

    order_by=id + id_below 방식(커서)을 쓴다. page 파라미터는 10,000건 이상에서
    막히고, 수집 중 새 관측이 올라오면 결과가 밀린다.
    """
    params = {
        "taxon_id": taxon_id,
        "quality_grade": quality,        # research = 커뮤니티 동의된 동정
        "photos": "true",
        "per_page": PER_PAGE,
        "order_by": "id",
        "order": "desc",
        "locale": "ko",
    }
    if not any_license:
        # 사진 라이선스 기준(관측 라이선스와 별개). CC 계열만.
        params["photo_license"] = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"

    collected: list[dict] = []
    photo_count = 0
    id_below: int | None = None

    while photo_count < need:
        q = dict(params)
        if id_below is not None:
            q["id_below"] = id_below
        limiter.wait()
        r = session.get(f"{API}/observations", params=q, timeout=30)
        if r.status_code != 200:
            print(f"  [warn] observations 조회 실패 ({r.status_code})")
            break
        results = r.json().get("results", [])
        if not results:
            break

        for obs in results:
            photos = (obs.get("photos") or [])[:max_per_obs]
            if not photos:
                continue
            collected.append(obs)
            photo_count += len(photos)

        id_below = results[-1]["id"]
        if len(results) < PER_PAGE:
            break  # 마지막 페이지

    return collected


def crawl_species(session, name: str, sci_name: str, *, limit: int, size: str,
                  max_per_obs: int, quality: str, any_license: bool,
                  limiter: RateLimiter, log: LicenseLog, dry_run: bool,
                  rejected: set[tuple[str, str]], label_species: str | None = None) -> int:
    """한 학명에 대해 수집. 저장 폴더는 `label_species`(기본 name)."""
    dest_species = label_species or name
    # 거부 목록(검수·프리필터에서 버린 것)은 세지 않는다 — 세면 "목표 달성"으로 오판해
    # 재수집이 아예 안 된다.
    have = count_usable(dest_species, rejected)
    need = max(0, limit - have)
    print(f"\n[data] {dest_species}  ({sci_name})  보유 {have} / 목표 {limit}", flush=True)
    if need == 0:
        print("  이미 목표 달성 — 건너뜀")
        return 0

    taxon = resolve_taxon(session, sci_name, limiter)
    if taxon is None:
        print(f"  [warn] taxon을 못 찾음 → 건너뜀. config.SPECIES['{name}'].scientific 확인 필요")
        return 0
    print(f"  taxon #{taxon['id']} {taxon.get('name')} "
          f"({taxon.get('preferred_common_name') or '-'}), "
          f"관측 {taxon.get('observations_count', '?')}건")
    if dry_run:
        return 0

    # 다운로드 실패(삭제된 사진 등)를 감안해 여유분을 더 긁는다.
    observations = fetch_observations(
        session, taxon["id"], limiter,
        need=int(need * 1.6) + 10, max_per_obs=max_per_obs,
        quality=quality, any_license=any_license,
    )
    print(f"  관측 {len(observations)}건 확보 → 다운로드 시작")

    out_dir = config.RAW_DIR / dest_species
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = skipped = failed = 0
    for obs in observations:
        if saved >= need:
            break
        for photo in (obs.get("photos") or [])[:max_per_obs]:
            if saved >= need:
                break
            lic = (photo.get("license_code") or "").lower()
            if not any_license and lic not in OPEN_LICENSES:
                skipped += 1
                continue

            url = photo_url(photo.get("url", ""), size)
            if not url:
                continue
            fname = f"inat_{obs['id']}_{photo['id']}.jpg"
            ok, why = download_image(session, url, out_dir / fname)
            if ok:
                saved += 1
                log.add(LicenseRow(
                    filename=fname,
                    species=dest_species,
                    source="inaturalist",
                    source_id=str(obs["id"]),
                    license=lic or "all-rights-reserved",
                    author=(photo.get("attribution") or "")[:200],
                    url=f"https://www.inaturalist.org/observations/{obs['id']}",
                ))
            elif why == "exists":
                skipped += 1
            else:
                failed += 1
            if saved and saved % 25 == 0:
                print(f"    {saved}/{need} …", flush=True)

    log.flush()
    print(f"  [OK] {dest_species}: +{saved}장 (건너뜀 {skipped}, 실패 {failed}) "
          f"→ 총 {count_usable(dest_species, rejected)}장")
    return saved


def crawl_other_class(session, *, limit: int, size: str, max_per_obs: int,
                      quality: str, any_license: bool, limiter: RateLimiter,
                      log: LicenseLog, dry_run: bool,
                      rejected: set[tuple[str, str]]) -> int:
    """`기타` 클래스의 'other_fish' 버킷을 iNaturalist에서 채운다.

    나머지 3버킷(조리·낚시장비·일반 오촬영)은 학명이 없으니 crawl_search.py 담당.
    """
    ratio = config.OTHER_BUCKET_RATIO["other_fish"]
    bucket_target = int(limit * ratio)
    taxa = config.OTHER_INAT_TAXA
    per_taxon = max(5, bucket_target // max(1, len(taxa)))
    print(f"\n[data] '{config.OTHER_CLASS}' other_fish 버킷: 목표 {bucket_target}장 "
          f"({len(taxa)}개 분류군 × {per_taxon}장)")

    total = 0
    out_dir = config.RAW_DIR / config.OTHER_CLASS
    out_dir.mkdir(parents=True, exist_ok=True)
    for sci in taxa:
        have_before = count_usable(config.OTHER_CLASS, rejected)
        # 클래스 전체 카운트를 쓰면 두 번째 분류군부터 목표 달성으로 오판된다
        # → 분류군별로 '현재 보유 + per_taxon'을 목표로 준다.
        total += crawl_species(
            session, sci, sci,
            limit=have_before + per_taxon, size=size, max_per_obs=max_per_obs,
            quality=quality, any_license=any_license, limiter=limiter, log=log,
            dry_run=dry_run, rejected=rejected, label_species=config.OTHER_CLASS,
        )
    return total


def main() -> None:
    ap = argparse.ArgumentParser(
        description="iNaturalist에서 어종 사진 수집 (+ data/licenses.csv 기록)")
    ap.add_argument("--species", nargs="*", default=None,
                    help="종명 또는 별칭(all/fish/other/confusable/easy/medium/hard/sea/fresh). "
                         "기본: 어종 24종")
    ap.add_argument("--limit", type=int, default=config.MVP_TARGET_IMAGES,
                    help=f"종당 목표 총 장수 (기본 MVP {config.MVP_TARGET_IMAGES})")
    ap.add_argument("--use-final-target", action="store_true",
                    help="난이도별 최종 목표량(easy 350/medium 600/hard 1000)을 쓴다")
    ap.add_argument("--size", default="medium", choices=("small", "medium", "large"),
                    help="다운로드 이미지 크기 (기본 medium=500px)")
    ap.add_argument("--max-per-obs", type=int, default=2,
                    help="관측 1건당 최대 사진 수. 크게 하면 유사 사진이 늘어난다 (기본 2)")
    ap.add_argument("--quality", default="research", choices=("research", "any"),
                    help="research=커뮤니티 동정 완료(권장). any=수집량 우선")
    ap.add_argument("--any-license", action="store_true",
                    help="라이선스 무관 수집. 공모전(비상업)엔 CC 계열만으로 충분하다")
    ap.add_argument("--sleep", type=float, default=1.1,
                    help="API 호출 간 최소 간격(초). iNat 권고 1초 이상 (기본 1.1)")
    ap.add_argument("--dry-run", action="store_true",
                    help="학명→taxon 매칭만 확인하고 다운로드하지 않는다")
    args = ap.parse_args()

    targets = resolve_species(args.species)
    limiter = RateLimiter(args.sleep)
    session = make_session()
    rejected = load_rejected()

    print(f"[cfg] 대상 {len(targets)}클래스 | size={args.size} | quality={args.quality} "
          f"| max-per-obs={args.max_per_obs} | "
          f"license={'any' if args.any_license else 'CC only'} | "
          f"거부목록 {len(rejected)}건")

    total = 0
    with LicenseLog() as log:
        for name in targets:
            sp = config.SPECIES[name]
            limit = sp.target_images if args.use_final_target else args.limit
            if name == config.OTHER_CLASS:
                total += crawl_other_class(
                    session, limit=limit, size=args.size,
                    max_per_obs=args.max_per_obs, quality=args.quality,
                    any_license=args.any_license, limiter=limiter, log=log,
                    dry_run=args.dry_run, rejected=rejected)
                continue
            total += crawl_species(
                session, name, sp.scientific,
                limit=limit, size=args.size, max_per_obs=args.max_per_obs,
                quality=args.quality, any_license=args.any_license,
                limiter=limiter, log=log, dry_run=args.dry_run, rejected=rejected)

            # 관측 수가 부족한 종은 근연 분류군으로 보충 (config.EXTRA_INAT_TAXA)
            for extra in config.EXTRA_INAT_TAXA.get(name, ()):
                if not args.dry_run and count_usable(name, rejected) >= limit:
                    break
                total += crawl_species(
                    session, extra, extra,
                    limit=limit, size=args.size, max_per_obs=args.max_per_obs,
                    quality=args.quality, any_license=args.any_license,
                    limiter=limiter, log=log, dry_run=args.dry_run,
                    rejected=rejected, label_species=name)

    print(f"\n[done] 신규 저장 {total}장  |  기록: {Path(config.DATA_DIR / 'licenses.csv')}")
    print("[next] python -m scripts.crawl_search   (실사용 유사 사진 + '기타' 나머지 버킷)")


if __name__ == "__main__":
    main()
