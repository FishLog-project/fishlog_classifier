# 환경 세팅 · 명령어 모음

## 현재 머신 상태 (2026-08-12 확인)

| 항목 | 값 | 판단 |
|---|---|---|
| OS | Windows 11 | |
| Python | 3.14.6 (유일) | ❌ torch/albumentations 휠 없음 → 3.11 설치 필요 |
| GPU | GeForce MX450 2GB | ⚠️ VRAM 부족, batch 16 이하 |
| NVIDIA 드라이버 | 452.56 (2020) | ❌ 최신 CUDA torch는 ≥525 필요 |

→ **확정(A-12): 실제 학습은 Colab 무료 T4, 로컬은 코드 검증용.**

## 로컬 세팅

```powershell
# 1) Python 3.11 설치 (winget 또는 python.org)
winget install Python.Python.3.11

# 2) 가상환경
cd C:\fishilog_ai
py -3.11 -m venv .venv
.venv\Scripts\activate

# 3) 의존성
pip install -r requirements.txt

# 4) (GPU 쓸 경우, 드라이버 업데이트 후) CUDA 빌드로 torch 재설치
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

확인:
```powershell
python -c "import torch, timm, albumentations; print(torch.__version__, torch.cuda.is_available())"
python -m src.config --summary
```

## Colab에서 학습하기

### 0) 로컬에서 데이터 패키징

```powershell
python -m scripts.package_data          # → data/splits.zip (약 1.07GB, 6,999장)
```

이 zip을 Google Drive `MyDrive/fishilog/splits.zip` 에 업로드한다.
(`data/*.zip` 은 .gitignore 대상 — 레포에 커밋되지 않는다.)

### 1) Colab 셀 (런타임 → GPU/T4 선택 후)

```python
# 셀 1 — 코드 + 의존성
!git clone -b dev https://github.com/FishLog-project/fishlog_classifier.git /content/fishilog_ai
%cd /content/fishilog_ai
!pip install -q timm "albumentations>=2.0" opencv-python-headless

# 셀 2 — 데이터를 Drive에서 "로컬 디스크로 풀기" (직접 읽기 금지)
from google.colab import drive; drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/fishilog/splits.zip -d /content/fishilog_ai/data/
!python -m src.dataset --check          # 4904 / 1048 / 1047 = 6999 확인

# 셀 3 — 체크포인트는 Drive에 남겨 세션 끊김에 대비
!mkdir -p /content/drive/MyDrive/fishilog/models
!rm -rf models && ln -s /content/drive/MyDrive/fishilog/models models

# 셀 4 — 학습
!python -m src.train --batch-size 64 --num-workers 2
```

- 데이터는 **zip 1개로 올려 Colab 로컬 디스크에 풀 것**. Drive를 직접 읽으면 소파일 수천 개에서 매우 느리다.
- 반대로 `models/` 는 큰 파일 몇 개뿐이라 Drive 링크가 이득이다.
- 세션이 끊기면 셀 1~3을 다시 돌린 뒤 `!python -m src.train --resume models/last.pt`.
- 한글 폴더명이 깨져 보이면 `!python -m src.dataset --check` 의 **합계 숫자**로 판단할 것 (표시 문제일 뿐 학습에는 영향 없음).

## 자주 쓰는 명령

| 목적 | 명령 |
|---|---|
| 24종 목록·수집 현황 | `python -m src.config --summary` |
| 폴더·labels.json 재생성 | `python -m src.config --init` |
| **학명→taxon 매칭 확인** | `python -m scripts.crawl_inat --dry-run --species all` |
| iNat 수집 | `python -m scripts.crawl_inat --species all` |
| 검색 수집(주) | `python -m scripts.crawl_ddg --species all` |
| 검색 수집(보조) | `python -m scripts.crawl_search --species all` |
| 검수 폴더 생성 | `python -m scripts.review --make --species confusable` |
| **검수 결과 확정** | `python -m scripts.review --record` |
| 혼동 쌍 참고표 | `python -m scripts.refsheet` → `reports/ref_*.jpg` |
| 중복 제거(raw→clean) | `python -m scripts.dedup` |
| 비물고기 필터 리포트 | `python -m scripts.prefilter` → `reports/prefilter.csv` |
| 비물고기 실제 격리 | `python -m scripts.prefilter --apply` |
| **검수 결과 확정** | `python -m scripts.dedup --record-deleted` |
| 라벨 충돌 검사 | `python -m scripts.dedup --species confusable --report-cross` |
| 분할(clean→splits) | `python -m scripts.split` |
| split별 분포 점검 | `python -m src.dataset --check` |
| 증강 결과 눈으로 확인 | `python -m src.dataset --preview` → `reports/aug_preview.jpg` |
| 배치 1개 로드 테스트 | `python -m src.dataset --batch` |
| **Colab용 zip 패키징** | `python -m scripts.package_data` → `data/splits.zip` |
| 파이프라인 스모크 테스트 | `python -m src.train --smoke` |
| 기본 학습 | `python -m src.train` |
| 무거운 백본 | `python -m src.train --backbone convnext_tiny --batch-size 16` |
| 이어서 학습 | `python -m src.train --resume models/last.pt` |

## 트러블슈팅

| 증상 | 원인·해결 |
|---|---|
| `imread` 결과가 전부 None | 한글 경로. `dataset.imread_unicode` 를 쓸 것 (cv2.imread 직접 호출 금지) |
| DataLoader가 멈춤/에러 (Windows) | `--num-workers 0`. 또는 진입점에 `if __name__ == "__main__":` 확인 |
| CUDA out of memory | `--batch-size 16` → `8`, `--img-size 192` |
| `torch.cuda.is_available() == False` | CPU 휠 설치됨 또는 드라이버 구버전. 위 4단계 재설치 |
| albumentations 인자 에러 | 2.x API 기준으로 작성됨. `pip install -U "albumentations>=2.0"` |
| 한글이 콘솔에서 깨짐 | `src.config._force_utf8_console()` 이 자동 처리한다. 그래도 깨지면 `$env:PYTHONUTF8="1"` |
| `UnicodeEncodeError: 'cp949'` | 위와 동일. `src.config` 를 import하지 않는 새 스크립트를 만들었을 때만 발생 |
| `ModuleNotFoundError: src` | `python scripts/x.py` 로 실행함. `python -m scripts.x` 로 실행할 것 |
| crawl_inat이 특정 종만 0장 | 학명이 iNat에서 개정됨. `--dry-run` 으로 확인 후 `config.SPECIES` 수정 |
| crawl_search가 0장 | Bing HTML 구조 변경. `pip install -U icrawler` 후 `--verbose` 로 확인 |
| 검수로 지운 파일이 되살아남 | `dedup --record-deleted` 를 안 돌렸다. [data-pipeline.md](data-pipeline.md) 참조 |
