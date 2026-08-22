# fishilog-ai

낚시 앱 어류 분류. 사진 → Top-3 후보 → 사용자 확정.
timm 전이학습 → ONNX → FastAPI 서빙. 25클래스 = 어종 24 + `기타`(OOD).

**새 세션은 `docs/roadmap.md`의 "현재 위치"부터 읽고 바로 이어서 작업한다.**
**불변**: `config.py`의 `SPECIES` 순서 = 출력 인덱스(변경 금지, `기타`는 항상 마지막). 수집 이미지 커밋 금지.

## 하려는 일 → `docs/` 문서

- `roadmap.md` — 진행 상황·다음 할 일 ← **여기부터**
- `decisions.md` — 확정된 결정·미결 논의
- `setup.md` — 환경·명령어
- `data-pipeline.md` — 수집·정제·분할
- `training.md` — 학습·튜닝
- `evaluation.md` — 평가·합격 기준
- `serving.md` — ONNX·서버·Docker
- `deploy.md` — EC2 배포 런북
- `integration.md` — 앱 연동
- `conventions.md` — 코드 규약
- `fish-model-plan.md` — 원본 계획서(아카이브, 현재 기준 아님)
