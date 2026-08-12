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

```python
# Colab 셀
!git clone <repo> fishilog_ai && cd fishilog_ai
!pip install -q timm albumentations opencv-python-headless
from google.colab import drive; drive.mount('/content/drive')
!ln -s /content/drive/MyDrive/fishilog_data/splits data/splits   # 데이터는 Drive에
!python -m src.train --batch-size 64 --num-workers 2
```

- 데이터는 **zip 1개로 올려 Colab 로컬 디스크에 풀 것**. Drive 직접 읽기는 소파일 수천 개에서 매우 느리다.
- 세션이 끊기면 `--resume models/last.pt` 로 이어서 학습.
- `models/`는 Drive에 심볼릭 링크해 체크포인트를 살려둔다.

## 자주 쓰는 명령

| 목적 | 명령 |
|---|---|
| 24종 목록·수집 현황 | `python -m src.config --summary` |
| 폴더·labels.json 재생성 | `python -m src.config --init` |
| split별 분포 점검 | `python -m src.dataset --check` |
| 증강 결과 눈으로 확인 | `python -m src.dataset --preview` → `reports/aug_preview.jpg` |
| 배치 1개 로드 테스트 | `python -m src.dataset --batch` |
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
| 한글이 콘솔에서 깨짐 | `$env:PYTHONUTF8="1"` |
