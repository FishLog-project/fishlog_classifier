# 코드 규약

## 절대 규칙

1. **`src/config.py`의 `SPECIES` 순서를 바꾸지 않는다.** 삽입 순서 = 모델 출력 인덱스.
   순서가 바뀌면 체크포인트·`labels.json`·ONNX의 라벨이 전부 어긋난다.
   현재 25클래스 = 어종 24(인덱스 0~23) + `기타`(인덱스 24).
   **`기타`는 항상 마지막**이므로, 종을 추가할 때는 `기타` **바로 앞**에 넣고 `OTHER_IDX`가 바뀐다는 점을 감안해 전체 재학습한다.
2. **종 목록을 다른 파일에 다시 적지 않는다.** 학명·키워드·목표량 모두 `config.SPECIES`에서 읽어 쓴다.
3. **`cv2.imread` 직접 호출 금지.** Windows에서 한글 경로를 못 읽는다 → `dataset.imread_unicode()` / `imwrite_unicode()` 사용.
4. **라벨 인덱스는 `config.CLASS_TO_IDX`로만 만든다.** 폴더 정렬 순서(`ImageFolder` 방식)에 의존 금지 — 한글 정렬은 로케일에 따라 달라진다.
5. **수집 이미지·체크포인트를 커밋하지 않는다.** `.gitignore`로 차단돼 있다.
6. **전처리는 학습·평가·서빙이 동일해야 한다.** 값은 체크포인트/`labels.json`에서 읽고 하드코딩하지 않는다.

## 파일 규칙

| 위치 | 용도 |
|---|---|
| `src/` | 학습·평가·변환. torch 의존 OK |
| `server/` | 추론 서버. **torch 의존 금지** (onnxruntime만) |
| `scripts/` | 1회성 데이터 수집·정제 스크립트 |
| `models/` | 체크포인트, `history.csv` (git 제외) |
| `reports/` | 평가 산출물, 미리보기 (git 제외) |

- 실행 가능한 모듈은 `python -m src.<name>` 형태로 argparse CLI를 갖는다.
- 진입점에는 반드시 `if __name__ == "__main__":` — Windows DataLoader가 spawn 방식이라 없으면 무한 재귀한다.

## 체크포인트 규약

`.pt`에는 가중치뿐 아니라 **재현에 필요한 메타데이터를 함께 저장한다**:
`model_state`, `backbone`, `classes`, `img_size`, `mean`, `std`, `epoch`, `metrics`, `config`.
→ 평가·ONNX 변환이 설정을 추측할 필요가 없다.

## 스타일

- 한글 주석 OK. 파일 인코딩은 UTF-8 고정.
- 타입 힌트 사용 (`from __future__ import annotations`).
- 주석은 "무엇을"이 아니라 **"왜"**를 적는다. (예: "한글 경로 때문에 imdecode 사용")
- 새 하이퍼파라미터는 `TrainConfig`에 추가하고 CLI 인자로 노출한다.
- 로그 접두사 통일: `[env] [cfg] [data] [loss] [resume] [OK] [warn] [done] [next]`

## 문서 갱신

- 작업을 끝내면 [roadmap.md](roadmap.md)의 체크박스와 "현재 위치"를 갱신한다.
- 결정이 나면 [decisions.md](decisions.md)의 B/C → A로 옮기고 결정 로그에 날짜를 남긴다.
- `CLAUDE.md`는 라우터다. 내용을 늘리지 말고 `docs/` 아래로 보낸다.
