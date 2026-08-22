# 앱 백엔드 연동

## 역할 분담

```
[앱] 사진 촬영
  → [앱 백엔드] 업로드 수신·리사이즈
  → [모델 서버] POST /predict → Top-3 + confidence
  → [앱] 후보 3개 표시 → 사용자가 확정
  → [앱 백엔드] 어종 확정 + fishing_spots_all.json 스팟 매핑 + 기록 저장
```

**모델 서버는 상태를 갖지 않는다.** 사용자·스팟·인증 로직은 전부 앱 백엔드 책임.
모델 서버가 아는 것은 "이 이미지는 24종 중 무엇처럼 보이는가" 뿐이다.

## 계약 (앱 백엔드 ↔ 모델 서버)

| 항목 | 값 |
|---|---|
| 엔드포인트 | `POST /predict` (multipart, field `file`) |
| 타임아웃 | 5초 권장 |
| 재시도 | 1회 (네트워크 오류에 한함, 4xx는 재시도 금지) |
| 실패 시 폴백 | "직접 어종 선택" 화면으로 전환 |
| 종명 키 | `server/labels.json` 의 한글명 문자열 그대로 (24종. `기타`는 응답에 나오지 않음) |

**종명 문자열이 두 시스템의 조인 키다.** `fishing_spots_all.json` 의 어종 표기와 `labels.json` 의 24종 표기가 정확히 일치하는지 연동 전에 대조할 것. (예: `우럭` vs `조피볼락`, `광어` vs `넙치`, `배스` vs `큰입배스` — `config.SPECIES[*].aliases` 에 이명을 적어뒀으니 매핑 테이블로 쓸 수 있다.)

## 백엔드 구현 예시 (Spring Boot 3.x)

### ⚠️ 이미지를 다시 인코딩하지 말 것

백엔드가 편의로 리사이즈·재압축해서 넘기면 **모델이 보는 픽셀이 달라진다.**
모델 서버는 학습과 비트 단위로 같은 전처리를 하도록 맞춰져 있는데(serving.md),
그 앞단에서 JPEG를 다시 굽으면 그 노력이 무의미해진다.
**받은 바이트를 그대로 넘긴다.** 크기 제한이 필요하면 리사이즈가 아니라 **거부**로 처리한다.

### 설정

```yaml
# application.yml
fishlog:
  model:
    url: http://10.0.1.42:8000      # 모델 EC2의 사설 IP (deploy.md 3~7단계)
    connect-timeout: 1s
    read-timeout: 5s                 # 실측 80ms. 5초는 순수 여유분이다
spring:
  servlet:
    multipart:
      max-file-size: 10MB            # 모델 서버 상한과 맞춘다
```

### 클라이언트

```java
@Component
public class FishClassifyClient {

    private static final Logger log = LoggerFactory.getLogger(FishClassifyClient.class);
    private final RestClient restClient;

    public FishClassifyClient(RestClient.Builder builder,
                              @Value("${fishlog.model.url}") String baseUrl) {
        var settings = ClientHttpRequestFactorySettings.DEFAULTS
                .withConnectTimeout(Duration.ofSeconds(1))
                .withReadTimeout(Duration.ofSeconds(5));
        this.restClient = builder
                .baseUrl(baseUrl)
                .requestFactory(ClientHttpRequestFactories.get(settings))
                .build();
    }

    /** 실패하면 예외 대신 Optional.empty() — 호출부는 "직접 선택" 화면으로 폴백한다. */
    public Optional<PredictResponse> classify(byte[] image, String filename) {
        try {
            return Optional.of(post(image, filename));
        } catch (HttpClientErrorException e) {
            // 4xx = 입력이 잘못됐다. 다시 보내도 같은 답이므로 재시도하지 않는다.
            log.warn("모델 서버가 입력을 거부: {} {}", e.getStatusCode(), e.getResponseBodyAsString());
            return Optional.empty();
        } catch (Exception e) {
            // 5xx·타임아웃·네트워크 오류만 1회 재시도 (모델 서버 재시작 중일 수 있다)
            log.warn("모델 서버 호출 실패, 1회 재시도: {}", e.toString());
            try {
                return Optional.of(post(image, filename));
            } catch (Exception retry) {
                log.error("모델 서버 재시도 실패 — 직접 선택으로 폴백", retry);
                return Optional.empty();
            }
        }
    }

    private PredictResponse post(byte[] image, String filename) {
        var body = new LinkedMultiValueMap<String, Object>();
        body.add("file", new ByteArrayResource(image) {
            @Override public String getFilename() { return filename; }  // 없으면 서버가 파일로 인식하지 않는다
        });
        return restClient.post()
                .uri("/predict")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(PredictResponse.class);
    }

    /** 배포 직후·모델 교체 후 확인용. model_version 이 기대값인지 본다. */
    public Map<String, Object> health() {
        return restClient.get().uri("/health").retrieve().body(Map.class);
    }
}
```

### 응답 DTO

```java
@JsonIgnoreProperties(ignoreUnknown = true)   // 서버가 필드를 추가해도 깨지지 않게
public record PredictResponse(
        boolean success,
        boolean uncertain,
        @JsonProperty("model_version") String modelVersion,
        List<Prediction> predictions,
        @JsonProperty("other_confidence") double otherConfidence,
        @JsonProperty("top1_confidence") double top1Confidence
) {
    public record Prediction(int rank, String species, double confidence) {}
}
```

### 컨트롤러

```java
@PostMapping(value = "/api/fish/identify", consumes = MULTIPART_FORM_DATA_VALUE)
public ResponseEntity<?> identify(@RequestParam("image") MultipartFile image) throws IOException {
    if (image.isEmpty() || image.getSize() > 10 * 1024 * 1024) {
        return ResponseEntity.badRequest().body(Map.of("error", "INVALID_IMAGE"));
    }
    // 원본 바이트 그대로 — 리사이즈·재인코딩 금지(위 경고 참조)
    return client.classify(image.getBytes(), image.getOriginalFilename())
            .<ResponseEntity<?>>map(ResponseEntity::ok)
            .orElseGet(() -> ResponseEntity.status(503)
                    .body(Map.of("error", "CLASSIFIER_UNAVAILABLE", "fallback", "MANUAL_SELECT")));
}
```

### 커넥션을 재사용할 것

위 예시처럼 `RestClient` 를 **빈으로 한 번 만들어 재사용한다.** 요청마다 새로 만들면
매번 TCP 연결이 새로 열려 처리량이 **15건/초 → 3.4건/초로 떨어진다**(실측, deploy.md 2절).
추론 자체보다 연결 비용이 더 커지는 구간이다.

응답은 **그대로 클라이언트에 넘겨도 된다.** `기타`는 이미 후보에서 빠져 있고
`uncertain` 플래그도 서버가 계산해 준다 — 백엔드가 다시 판단할 필요가 없다.

## 앱 UX 요구사항

| 상황 | 앱 동작 |
|---|---|
| 정상 (`uncertain: false`) | Top-3 후보를 카드로 표시, 1순위 강조. 사용자가 탭해서 확정 |
| `uncertain: true` | "인식이 어려워요. 물고기가 화면에 크게 나오도록 다시 찍어주세요" + 그래도 후보는 보여줌 |
| 24종에 없는 어종 | **"목록에서 직접 선택"** 경로 필수 → [decisions.md B-3](decisions.md) |
| 서버 오류·타임아웃 | 직접 선택으로 폴백. 사진은 이미 저장해 두고 나중에 재시도 |

confidence를 사용자에게 % 로 노출할지는 신중히. 보정 전 softmax 확률은 실제 정확도보다 과신하는 경향이 있다 ([decisions.md C-2](decisions.md)).

## 데이터 플라이휠 (Phase 7 이후)

사용자가 확정한 (사진, 어종) 쌍은 **가장 가치 있는 학습 데이터**다. 실사용 분포 그 자체이기 때문.

- 저장 조건: 사용자 동의 필수 (개인정보 처리방침에 "AI 성능 개선 목적 이용" 명시)
- 특히 가치 있는 케이스: **모델 Top-1 ≠ 사용자 확정** — 모델이 틀린 실사례
- 분기별로 모아 재학습 → 정확도 개선 → `model_version` 올려 재배포
- 사용자 확정도 100% 정답은 아니다. 재학습 전 샘플 검수 필요

## 연동 테스트 시나리오

- [ ] 정상 사진 (24종 각 1장 이상)
- [ ] 손에 들고 찍은 사진 / 바닥에 놓인 사진 / 뜰채 안
- [ ] 역광·야간 플래시·물속 사진
- [ ] 여러 마리가 찍힌 사진
- [ ] 물고기가 아닌 사진 (사람·풍경·회 접시)
- [ ] 24종에 없는 어종 (향어·학꽁치 등)
- [ ] 세로로 찍은 폰 사진 (EXIF 회전)
- [ ] 10MB 초과 파일 / 손상 파일 / 빈 파일
- [ ] 모델 서버 다운 상태에서 앱 동작
