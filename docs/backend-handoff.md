# 백엔드 인계용 프롬프트

앱 백엔드(Spring Boot)는 **다른 레포**라 그쪽 Claude Code 세션은 이 레포의 문서를
읽지 못한다. 그래서 필요한 내용을 전부 담은 **자립형 프롬프트**를 여기 둔다.

아래 코드블록을 통째로 복사해 IntelliJ의 Claude Code에 붙여넣으면 된다.
내용의 원본은 [integration.md](integration.md)(계약·Spring 코드)와
[deploy.md](deploy.md)(배포 상태)다 — **모델 서버가 바뀌면 이 파일도 함께 갱신할 것.**
특히 사설 IP·`model_version`·임계값은 재배포 때 달라질 수 있다.

---

````text
낚시 앱 FishLog의 백엔드(Spring Boot)에 어류 분류 AI 서버를 연동하려고 해.
AI 모델 서버는 이미 만들어서 EC2에 배포까지 끝났고, 이 레포에서 할 일은
그 서버를 호출해서 결과를 클라이언트에 전달하는 것뿐이야. 모델 쪽은 건드릴 필요 없어.

## 구조
[앱] 사진 촬영 → [이 백엔드 EC2] → [모델 EC2 172.31.14.180:8000] → 응답 → [백엔드] → [앱]
모델 서버는 상태를 갖지 않아. 사용자·인증·스팟 매핑은 전부 백엔드 책임이야.

## 모델 서버 계약

POST http://172.31.14.180:8000/predict  (multipart/form-data, 필드명 `file`)

성공 응답:
{
  "success": true,
  "uncertain": false,
  "model_version": "b0-384-20260818",
  "predictions": [
    {"rank": 1, "species": "붕어", "confidence": 0.83},
    {"rank": 2, "species": "잉어", "confidence": 0.05},
    {"rank": 3, "species": "가물치", "confidence": 0.01}
  ],
  "other_confidence": 0.01,
  "top1_confidence": 0.83,
  "latency_ms": 81.2
}

실패 응답: {"success": false, "error": "<코드>", "detail": "..."}
| 상황 | HTTP | error |
| 빈 파일 | 400 | EMPTY_FILE |
| 손상·비이미지 | 400 | IMAGE_DECODE_FAILED |
| 미지원 포맷 | 415 | UNSUPPORTED_FORMAT |
| 10MB 초과 | 413 | FILE_TOO_LARGE |
| 화소 5천만 초과 | 413 | IMAGE_TOO_LARGE |
| 모델 미로드 | 503 | MODEL_NOT_LOADED |

GET /health → {"status":"ok","model_version":"...","num_classes":25,...} (모델 미로드 시 503)
GET /labels → 25종 전체 목록과 학명·서식지

## 반드시 지킬 것 두 가지

1. 이미지를 리사이즈·재인코딩하지 말고 **원본 바이트 그대로** 넘길 것.
   모델 서버는 학습과 비트 단위로 같은 전처리를 하도록 맞춰져 있어서,
   앞단에서 JPEG를 다시 구우면 정확도가 조용히 떨어져.
   크기 제한이 필요하면 리사이즈가 아니라 '거부'로 처리해.

2. RestClient를 빈으로 만들어 **재사용**할 것. 요청마다 새로 만들면
   처리량이 15건/초 → 3.4건/초로 떨어지는 걸 실측했어.

## 구현할 것

- application.yml: 모델 서버 URL, connect timeout 1s, read timeout 5s,
  multipart max-file-size 10MB
- FishClassifyClient: multipart 전송, 4xx는 재시도 금지(입력이 잘못된 거라 다시 보내도 같음),
  5xx·타임아웃·네트워크 오류만 1회 재시도, 실패 시 예외 대신 Optional.empty() 반환
- PredictResponse DTO: record + @JsonIgnoreProperties(ignoreUnknown=true)
  (snake_case 필드는 @JsonProperty로 매핑: model_version, other_confidence, top1_confidence)
- 컨트롤러: MultipartFile 받아서 그대로 전달, 실패 시 503 + fallback 안내
- 에러 케이스에 ErrorCode 추가

## 응답 처리 규칙

- 모델 응답은 거의 그대로 클라이언트에 넘겨도 돼. `기타`(비물고기/24종 밖)는
  이미 후보에서 빠져 있고 uncertain 판정도 서버가 끝내서 줘.
- uncertain: true 여도 **후보 3개는 그대로 보여준다.** "다시 찍어주세요"를 덧붙일 뿐이야.
- confidence는 25클래스 softmax 원값이라 predictions의 합이 1이 아니야.
  사용자에게 %로 그대로 노출할지는 신중히 (보정 전이라 과신 경향이 있음).

## 종명이 두 시스템의 조인 키야

모델이 주는 종명 문자열과 fishing_spots_all.json의 어종 표기가 정확히 일치해야 해.
연동 전에 대조해줘 — 우럭 vs 조피볼락, 광어 vs 넙치, 배스 vs 큰입배스 같은 표기 차이가
있으면 스팟 매핑이 조용히 실패해.

모델이 반환하는 24종:
감성돔, 농어, 돌돔, 벵에돔, 우럭, 참돔, 광어, 볼락, 갈치, 고등어, 삼치, 방어,
전갱이, 숭어, 붕어, 잉어, 쏘가리, 배스, 블루길, 가물치, 메기, 송어, 피라미, 동자개

## 알아둘 것

- 모델 정확도: Top-3 90.7%, Top-1 81%. Top-3 안에서 사용자가 고르는 구조라 이 수치로 충분해.
- 24종에 없는 어종(향어·학꽁치 등)은 Top-3에 정답이 아예 없어.
  "목록에서 직접 선택" 대안 경로가 반드시 필요해.
- 응답 지연은 평균 80ms. 타임아웃 5초는 순수 여유분이야.
- 모델 서버는 인증이 없어. 보안그룹으로 백엔드에서만 접근 가능하게 막혀 있으니
  이 주소를 외부에 노출하거나 클라이언트가 직접 부르게 만들면 안 돼.

우선 현재 백엔드 구조를 파악하고, 어디에 붙이는 게 맞을지 알려줘.
````

---

## 붙여넣기 전에

- `fishing_spots_all.json` 이 백엔드 레포에 없다면 "종명 조인 키" 문단은 빼도 된다.
- 모델 서버를 재배포해 사설 IP나 `model_version` 이 바뀌었다면 프롬프트의 값을
  먼저 고칠 것. 현재 값은 [deploy.md](deploy.md) 0절에 있다.

## 백엔드에서 연동 확인하는 명령

```bash
curl -s http://172.31.14.180:8000/health
curl -s -F "file=@fish.jpg" http://172.31.14.180:8000/predict
```

`/health` 가 200이면 통신은 끝난 것이다. 깨진 파일을 보내도 **4xx + JSON**이 오면
정상이다(500이 오면 안 된다).

## 아직 안 정해진 것

- **24종 밖 어종을 잡았을 때 앱 UX** ([decisions.md B-3](decisions.md)) —
  향어·학꽁치를 잡으면 Top-3에 정답이 아예 없다. "직접 선택" 화면이 앱에 있는지,
  없으면 누가 만드는지 정해야 한다.
- **confidence 를 사용자에게 % 로 노출할지** ([decisions.md C-2](decisions.md)) —
  노출한다면 temperature scaling 보정이 먼저다.
- **사용자 확정 결과 로깅** ([decisions.md C-5](decisions.md)) — 모델 Top-1과 사용자
  확정이 다른 사례가 재학습에 가장 값진 데이터다. 수집 동의·개인정보 처리방침이 필요하다.
