"""수집 스크립트 공용 유틸 — HTTP 세션, 이미지 검증, 라이선스 로그.

여기 있는 것들은 `scripts/*` 전용이다. 학습 코드(`src/`)는 이 모듈에 의존하지 않는다.
"""

from __future__ import annotations

import csv
import hashlib
import io
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests
from PIL import Image

from src import config

# iNaturalist는 UA 없는 요청을 차단할 수 있고, 크롤러 식별을 요구한다.
USER_AGENT = "fishilog-ai/0.1 (fish classifier for a non-commercial contest; contact via GitHub)"

# 너무 작은 이미지는 학습에 쓸모없다(썸네일·아이콘). 224px 입력이므로 최소 200px.
MIN_SIDE = 200
MIN_BYTES = 6 * 1024
MAX_BYTES = 12 * 1024 * 1024


def make_session(retries: int = 3) -> requests.Session:
    """재시도 붙인 세션. 수천 장 다운로드 중 일시적 5xx로 죽지 않게."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=retries,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def valid_image_bytes(buf: bytes, min_side: int = MIN_SIDE) -> tuple[bool, str]:
    """다운로드한 바이트가 학습에 쓸 만한 이미지인지 검사한다.

    HTML 에러 페이지나 잘린 파일이 .jpg로 저장되는 사고를 여기서 막는다.
    """
    if len(buf) < MIN_BYTES:
        return False, "too_small_bytes"
    if len(buf) > MAX_BYTES:
        return False, "too_large_bytes"
    try:
        img = Image.open(io.BytesIO(buf))
        img.verify()                      # 구조 검증 (verify 후에는 재사용 불가)
        img = Image.open(io.BytesIO(buf))  # 크기 확인용 재오픈
        w, h = img.size
    except Exception:
        return False, "decode_failed"
    if min(w, h) < min_side:
        return False, f"small_{w}x{h}"
    if max(w, h) / max(1, min(w, h)) > 4.0:
        return False, "extreme_aspect"    # 파노라마/배너 이미지
    return True, "ok"


def download_image(session: requests.Session, url: str, dest: Path,
                   timeout: float = 20.0, min_side: int = MIN_SIDE,
                   extra_check=None) -> tuple[bool, str]:
    """이미지를 내려받아 검증 후 저장한다. (성공여부, 사유) 반환.

    검증을 통과한 뒤에만 파일을 만든다 → 중간에 끊겨도 깨진 파일이 남지 않고
    `dest.exists()` 를 그대로 '이미 받음' 판정에 쓸 수 있다(재실행 안전).
    """
    if dest.exists():
        return False, "exists"
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code != 200:
            return False, f"http_{r.status_code}"
        buf = r.content
    except Exception as e:
        return False, f"error_{type(e).__name__}"

    ok, why = valid_image_bytes(buf, min_side=min_side)
    if not ok:
        return False, why

    # 추가 판정(예: 물고기 확률). 저장 전에 걸러야 쓰레기가 디스크에 남지 않는다.
    if extra_check is not None:
        ok2, why2 = extra_check(buf)
        if not ok2:
            return False, why2

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(buf)
    tmp.replace(dest)  # 원자적 교체
    return True, "ok"


# ---------------------------------------------------------------------------
# 라이선스 로그 (data/licenses.csv)
# ---------------------------------------------------------------------------
LICENSE_FIELDS = ["filename", "species", "source", "source_id",
                  "license", "author", "url", "collected_at"]


@dataclass
class LicenseRow:
    filename: str
    species: str
    source: str          # inaturalist | bing | google | manual
    source_id: str = ""
    license: str = "unknown"
    author: str = ""
    url: str = ""
    collected_at: str = ""

    def as_dict(self) -> dict[str, str]:
        d = self.__dict__.copy()
        d["collected_at"] = d["collected_at"] or date.today().isoformat()
        return d


class LicenseLog:
    """`data/licenses.csv` 추가 기록기.

    이미 기록된 filename은 건너뛴다(크롤러 재실행 시 중복 행 방지).
    상업 전환 시 CC0/CC-BY만 남기고 재필터링하려면 이 파일이 필수다 (decisions C-8).
    """

    def __init__(self, path: Path | None = None):
        self.path = path or (config.DATA_DIR / "licenses.csv")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._known = self._load_filenames()
        self._buffer: list[dict[str, str]] = []

    def _load_filenames(self) -> set[tuple[str, str]]:
        """이미 기록된 (종, 파일명) 쌍.

        파일명만으로 판정하면 **같은 사진이 두 클래스에 들어갔을 때 두 번째 기록이 사라진다**
        → 출처 추적이 끊기고, 클래스 간 라벨 충돌(같은 사진이 붕어·잉어 양쪽에 존재)을
        licenses.csv 로는 발견할 수 없게 된다.
        """
        if not self.path.exists():
            return set()
        with self.path.open("r", encoding="utf-8-sig", newline="") as f:
            return {(row.get("species", ""), row["filename"])
                    for row in csv.DictReader(f) if row.get("filename")}

    def add(self, row: LicenseRow) -> bool:
        key = (row.species, row.filename)
        if key in self._known:
            return False
        self._known.add(key)
        self._buffer.append(row.as_dict())
        if len(self._buffer) >= 50:
            self.flush()
        return True

    def flush(self) -> None:
        if not self._buffer:
            return
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        # utf-8-sig: 엑셀에서 한글 종명이 깨지지 않게 (제출 근거로 열어볼 파일)
        with self.path.open("a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LICENSE_FIELDS)
            if new_file:
                w.writeheader()
            w.writerows(self._buffer)
        self._buffer.clear()

    def __enter__(self) -> LicenseLog:
        return self

    def __exit__(self, *exc) -> None:
        self.flush()


# ---------------------------------------------------------------------------
# 거부 목록 (사람 검수 결과를 영구 보존)
# ---------------------------------------------------------------------------
# clean/ 은 raw/ 의 하드링크라서, 크롤링을 더 한 뒤 dedup을 다시 돌리면 검수로 지운
# 파일이 그대로 되살아난다. 그래서 '버린 파일'을 여기에 기록하고 dedup이 참조한다.
# (사람이 몇 시간 들인 검수 결과가 조용히 사라지는 걸 막는 유일한 장치다.)
REJECTED_TXT = config.DATA_DIR / "rejected.txt"
CLEAN_MANIFEST = config.DATA_DIR / "clean_manifest.csv"


def load_rejected() -> set[tuple[str, str]]:
    """{(종명, 파일명)} — 다시는 clean에 넣지 않을 파일들."""
    if not REJECTED_TXT.exists():
        return set()
    out: set[tuple[str, str]] = set()
    for line in REJECTED_TXT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "/" not in line:
            continue
        sp, _, fn = line.partition("/")
        out.add((sp, fn))
    return out


def append_rejected(pairs: list[tuple[str, str]]) -> int:
    """거부 목록에 추가(중복 무시). 추가된 개수를 돌려준다."""
    known = load_rejected()
    new = [p for p in pairs if p not in known]
    if not new:
        return 0
    REJECTED_TXT.parent.mkdir(parents=True, exist_ok=True)
    header = "" if REJECTED_TXT.exists() else "# 검수/프리필터에서 버린 파일: <종명>/<파일명>\n"
    with REJECTED_TXT.open("a", encoding="utf-8") as f:
        f.write(header)
        for sp, fn in new:
            f.write(f"{sp}/{fn}\n")
    return len(new)


def load_clean_manifest() -> set[tuple[str, str]]:
    """dedup이 마지막으로 clean에 넣은 파일 목록. 사람이 지운 것을 역산하는 기준."""
    if not CLEAN_MANIFEST.exists():
        return set()
    with CLEAN_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        return {(r["species"], r["filename"]) for r in csv.DictReader(f)
                if r.get("species") and r.get("filename")}


def write_clean_manifest(pairs: list[tuple[str, str]]) -> None:
    CLEAN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with CLEAN_MANIFEST.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "filename"])
        w.writerows(sorted(pairs))


# ---------------------------------------------------------------------------
# 클래스 인자 처리
# ---------------------------------------------------------------------------
def resolve_species(names: list[str] | None, *, fish_only: bool = False) -> list[str]:
    """CLI의 --species 인자를 클래스명 리스트로 바꾼다.

    별칭도 받는다: `all`(전체), `fish`(어종 24), `other`(기타),
    `easy|medium|hard`(난이도), `confusable`(혼동 그룹 전체), `sea|fresh`(서식).
    """
    if not names:
        return config.FISH_CLASSES if fish_only else config.CLASSES

    out: list[str] = []
    for raw in names:
        n = raw.strip()
        if n in config.SPECIES:
            out.append(n)
        elif n == "all":
            out.extend(config.CLASSES)
        elif n == "fish":
            out.extend(config.FISH_CLASSES)
        elif n == "other":
            out.append(config.OTHER_CLASS)
        elif n == "confusable":
            out.extend([s for g in config.CONFUSABLE_GROUPS for s in g])
        elif n in config.DIFFICULTY_TARGET:
            out.extend([k for k, v in config.SPECIES.items() if v.difficulty == n])
        elif n in ("sea", "fresh"):
            out.extend([k for k, v in config.SPECIES.items() if v.habitat == n])
        else:
            raise SystemExit(
                f"[warn] 알 수 없는 종/별칭: {raw}\n"
                f"  사용 가능: {', '.join(config.CLASSES)}\n"
                f"  별칭: all, fish, other, confusable, easy, medium, hard, sea, fresh"
            )

    seen, uniq = set(), []
    for n in out:  # config 순서 유지 + 중복 제거
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    if fish_only:
        uniq = [n for n in uniq if config.SPECIES[n].scientific]
    return uniq


def count_raw(species: str) -> int:
    d = config.RAW_DIR / species
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir()
               if p.suffix.lower() in config.VALID_IMAGE_EXTS and p.is_file())


def count_usable(species: str, rejected: set[tuple[str, str]], prefix: str = "") -> int:
    """`raw/<종>`에서 **쓸 수 있는** 이미지 수. 거부 목록에 있는 파일은 세지 않는다.

    거부된 쓰레기 사진을 세면 "이미 목표 달성"으로 오판해 재수집이 아예 안 된다.
    (거부 파일을 raw에서 지우지는 않는다 — 남겨두면 같은 URL을 다시 받지 않는다.)
    """
    d = config.RAW_DIR / species
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir()
               if p.name.startswith(prefix)
               and p.suffix.lower() in config.VALID_IMAGE_EXTS
               and (species, p.name) not in rejected)


def web_filename(url: str) -> str:
    """검색 수집분 파일명. **모든 검색엔진이 같은 규칙을 쓴다.**

    URL 해시라서 (1) 재실행해도 같은 사진을 다시 받지 않고,
    (2) Bing/DDG/네이버가 같은 사진을 줘도 자동으로 한 장만 남는다.
    `web_` 접두사는 prefilter·통계가 '검색 수집분'을 골라내는 기준이기도 하다.
    """
    return f"web_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}.jpg"


def download_from_pages(session: requests.Session, species: str, query: str, want: int,
                        pages, *, source: str, log: LicenseLog,
                        rejected: set[tuple[str, str]], extra_check=None,
                        min_side: int = MIN_SIDE) -> int:
    """검색 결과 페이지들을 소비해 `want`장 저장한다 (엔진 공통 드라이버).

    Args:
        pages: `(url, width, height)` 리스트를 페이지 단위로 내놓는 이터러블.
               폭·높이를 모르면 0을 넣으면 된다(다운로드 후 검증한다).
        source: licenses.csv 에 남길 출처 (bing | ddg | naver).
        extra_check: 저장 직전 판정 함수. 물고기 점수 필터가 여기로 들어온다.
    """
    out_dir = config.RAW_DIR / species
    saved = skipped = failed = 0

    for page in pages:
        if saved >= want:
            break
        for url, w, h in page:
            if saved >= want:
                break
            if not url:
                continue
            if w and h and min(w, h) < min_side:   # 검색엔진이 준 크기로 미리 거른다
                skipped += 1
                continue
            fname = web_filename(url)
            if (species, fname) in rejected:       # 전에 버린 사진 — 다시 받지 않는다
                skipped += 1
                continue

            ok, why = download_image(session, url, out_dir / fname,
                                     min_side=min_side, extra_check=extra_check)
            if ok:
                saved += 1
                log.add(LicenseRow(filename=fname, species=species, source=source,
                                   source_id=query,       # 어떤 쿼리로 걸렸는지 = 검수 단서
                                   license="unknown",     # 검색 수집분은 항상 unknown (C-8)
                                   url=url))
            elif why == "exists":
                skipped += 1
            else:
                failed += 1

    log.flush()
    print(f"    '{query}' → +{saved}장 (건너뜀 {skipped}, 탈락 {failed})", flush=True)
    return saved


class RateLimiter:
    """iNaturalist 권고(초당 1회 이하, 분당 60회 이하)를 지키는 최소 구현."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()
