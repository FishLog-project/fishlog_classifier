# EC2 배포 런북

모델 서버를 **전용 EC2 1대**에 컨테이너로 올리고, 앱 백엔드 EC2가 중계한다.
(decisions.md B-2, 2026-08-20 확정)

```
[앱] 사진 촬영
  → [백엔드 EC2]  공인 IP, HTTPS, 인증·업로드 처리
      → 사설 IP 호출 POST /predict
  → [모델 EC2]    공인 접근 차단. Top-3 + confidence 반환
      → 응답
  → [백엔드 EC2]  결과를 그대로(또는 가공해) 클라이언트에 전달
```

**모델 서버에는 인증이 없다.** 상태 없는 내부 서비스로 설계했기 때문이다
(integration.md). 따라서 **보안그룹으로 막는 것이 유일한 방어선**이다 — 3단계를 건너뛰지 말 것.

---

## 1. 인스턴스

| | 모델 EC2 | 비고 |
|---|---|---|
| 타입 | **t3.small** (2 vCPU / 2GB) | 메모리 실측 132MB라 1GB로도 돌지만, 추론이 코어를 통째로 쓰므로 vCPU가 병목이다 |
| OS | Amazon Linux 2023 또는 Ubuntu 22.04+ | |
| 디스크 | 기본 8GB gp3 | 이미지 748MB + 여유 |
| AZ | **백엔드 EC2와 같은 VPC** | 사설 IP 호출이면 지연 1ms 미만 |
| 공인 IP | 붙이되 SG로 막는다 | 없으면 docker·apt 설치에 NAT Gateway가 필요해 비용이 더 든다 |

**t2/t3는 버스트 크레딧에 주의.** 추론이 CPU를 계속 쓰므로 지속 부하 시 크레딧이 소진되며
성능이 깎인다. 트래픽이 꾸준하면 `t3.small` + **unlimited 모드**를 켜거나 `c6i.large` 를 본다.
Graviton(`c6g`, `t4g`)은 더 싸지만 **arm64 이미지로 다시 빌드해야 한다**(6-C 참조).

## 2. 처리량 (2026-08-20 실측, 2워커 / 로컬 Docker)

| 동시 요청 | 처리량 | 지연 중앙값 | p95 |
|---|---|---|---|
| 1 | 15.1 건/초 | 62ms | 99ms |
| **2** | **18.1 건/초** | 110ms | 134ms |
| 4 | 15.9 건/초 | 248ms | 279ms |
| 8 | 14.0 건/초 | 504ms | 792ms |

**워커당 약 9건/초.** `-e WEB_CONCURRENCY=<vCPU 수>` 로 맞춘다(워커당 메모리 약 145MB).
t3.small(2 vCPU)이면 **약 18건/초**로, 공모전 시연 규모에는 충분하다.

동시 요청이 워커 수를 넘으면 **처리량은 안 늘고 지연만 선형으로 늘어난다**(동시 8 → 504ms).
CPU를 통째로 쓰는 작업이라 큐가 쌓일 뿐이기 때문이다. 늘리려면 워커(=vCPU)를 늘려야 한다.

⚠️ **백엔드가 커넥션을 재사용해야 한다.** 요청마다 새 연결을 열면 같은 서버에서
15건/초가 **3.4건/초로 떨어진다**(실측). Spring이면 `RestClient` 빈을 싱글턴으로 두고
재사용하면 된다 — 매 요청 `RestClient.create()` 를 호출하지 말 것.

## 3. 보안그룹 — 여기가 핵심

| SG | 방향 | 포트 | 소스 | 이유 |
|---|---|---|---|---|
| `model-sg` | 인바운드 | 8000 | **`backend-sg`(보안그룹 ID로 지정)** | 백엔드에서만 추론 호출 가능 |
| `model-sg` | 인바운드 | 22 | 내 사무실/집 IP `/32` | SSH |
| `model-sg` | 아웃바운드 | 전체 | 0.0.0.0/0 | 이미지·패키지 내려받기 |

소스를 IP가 아니라 **보안그룹 ID로 지정**한다. 백엔드 인스턴스가 재시작돼 사설 IP가 바뀌어도
규칙을 고칠 필요가 없다.

**8000 포트를 0.0.0.0/0 으로 열지 말 것.** 열면 누구나 이 모델로 무제한 추론을 돌릴 수 있다.

## 4. 모델 서버 초기 설정

```bash
ssh -i <키>.pem ec2-user@<모델서버 공인IP>

# Amazon Linux 2023
sudo dnf install -y docker
sudo systemctl enable --now docker        # enable 을 빠뜨리면 재부팅 후 안 뜬다
sudo usermod -aG docker ec2-user
exit && ssh ...                            # 그룹 반영을 위해 재접속
```

Ubuntu면 `sudo apt update && sudo apt install -y docker.io` 로 대체.

## 5. 이미지 전달

### A. scp (첫 배포에 권장 — AWS 설정이 필요 없다)

로컬에서 한 줄로 끝난다(압축 195MB):

```bash
docker save fishilog-ai:b0-384-20260818 \
  | gzip \
  | ssh -i <키>.pem ec2-user@<모델서버IP> 'gunzip | docker load'
```

### B. ECR (반복 배포에 권장)

```bash
aws ecr create-repository --repository-name fishilog-ai
aws ecr get-login-password --region ap-northeast-2 \
  | docker login --username AWS --password-stdin <계정ID>.dkr.ecr.ap-northeast-2.amazonaws.com
docker tag fishilog-ai:b0-384-20260818 <계정ID>.dkr.ecr.ap-northeast-2.amazonaws.com/fishilog-ai:b0-384-20260818
docker push <계정ID>.dkr.ecr.ap-northeast-2.amazonaws.com/fishilog-ai:b0-384-20260818
```

인스턴스에는 `AmazonEC2ContainerRegistryReadOnly` IAM 역할을 붙이고 `docker pull`.

### C. 인스턴스에서 직접 빌드 (Graviton/arm64를 쓸 때는 이 방법뿐)

`server/model.onnx` 는 git에 없다 — 레포를 클론한 뒤 **모델 파일만 따로 올려야** 한다.

```bash
git clone <레포> && cd fishilog_ai
scp -i <키>.pem server/model.onnx ec2-user@<IP>:~/fishilog_ai/server/   # 로컬에서
docker build -t fishilog-ai:b0-384-20260818 .
```

## 6. 실행

```bash
docker run -d --name fishilog \
  --restart unless-stopped \
  -p 8000:8000 \
  -e WEB_CONCURRENCY=2 \
  fishilog-ai:b0-384-20260818

docker logs -f fishilog          # "모델 로드 완료 ... 워밍업 ..." 이 뜨면 정상
curl -s localhost:8000/health
```

한 겹 더 잠그려면 사설 IP에만 바인딩한다(SG와 별개로 OS 레벨 차단):

```bash
-p $(hostname -I | awk '{print $1}'):8000:8000
```

**태그를 `latest` 가 아니라 `b0-384-20260818` 로 고정해서 띄운다.** 어떤 모델이 도는지
`/health` 의 `model_version` 과 태그가 함께 증언해야 한다(decisions.md C-6).

## 7. 확인 — 백엔드 EC2에서

```bash
curl -s http://<모델서버 사설IP>:8000/health
curl -s -F "file=@fish.jpg" http://<모델서버 사설IP>:8000/predict
```

`/health` 가 200이고 `model_version` 이 기대한 값이면 성공.
**밖(내 노트북)에서 같은 요청이 실패해야** 보안그룹이 제대로 걸린 것이다.

## 8. 백엔드가 지켜야 할 계약

integration.md의 계약을 그대로 옮긴다:

| 항목 | 값 |
|---|---|
| 엔드포인트 | `POST http://<사설IP>:8000/predict` (multipart, field `file`) |
| 타임아웃 | **5초** (실측 80ms이므로 넉넉하다) |
| 재시도 | 네트워크 오류·5xx만 1회. **4xx는 재시도 금지** (입력이 잘못된 것이라 다시 보내도 같다) |
| 실패 폴백 | "직접 어종 선택" 화면 |
| 업로드 상한 | 10MB. 백엔드에서 먼저 걸러 주면 더 좋다 |

- `uncertain: true` 여도 **후보는 그대로 보여준다.** "다시 찍어주세요"를 덧붙일 뿐이다.
- `predictions` 의 `species` 문자열이 **두 시스템의 조인 키**다. `fishing_spots_all.json` 의
  어종 표기와 대조할 것(`GET /labels` 로 25개 전체를 받을 수 있다).
- `confidence` 를 사용자에게 % 로 그대로 노출하는 것은 신중히 — 보정 전 softmax는
  과신하는 경향이 있다(decisions.md C-2).

## 9. 모델 교체 절차

```bash
# 1) 로컬: 새 체크포인트 → ONNX → 검증 → 이미지
python -m src.export_onnx --ckpt models/<새모델>.pt --version b0-384-<날짜>
python -m scripts.check_preprocess && python -m scripts.check_server
python -m scripts.tune_threshold --write        # 임계값은 모델마다 다시 잰다
docker build -t fishilog-ai:b0-384-<날짜> .

# 2) 전달 후 인스턴스에서 교체
docker stop fishilog && docker rm fishilog
docker run -d --name fishilog --restart unless-stopped -p 8000:8000 \
  -e WEB_CONCURRENCY=2 fishilog-ai:b0-384-<날짜>
curl -s localhost:8000/health      # model_version 이 바뀌었는지 확인
```

**임계값을 다시 재는 것을 건너뛰지 말 것.** 모델이 바뀌면 확률 분포가 바뀌어
같은 0.25가 전혀 다른 거부율을 뜻하게 된다.

이전 이미지는 지우지 말고 남겨둔다 — 문제가 생기면 `docker run` 한 줄로 되돌린다.

## 10. 운영

| 상황 | 확인 |
|---|---|
| 응답이 없다 | `docker ps`, `docker logs --tail 100 fishilog` |
| `/health` 503 | 모델 로드 실패. `detail` 필드에 원인이 들어 있다 |
| 느리다 | `docker stats` 로 CPU 100% 여부 → `WEB_CONCURRENCY` 상향 또는 인스턴스 확대 |
| 재부팅 후 안 뜸 | `sudo systemctl enable docker` 를 안 한 것 |

로그는 컨테이너 stdout으로만 남는다. 장기 보관이 필요하면 CloudWatch Logs 드라이버를 붙인다:
`--log-driver=awslogs --log-opt awslogs-group=/fishilog/model`

## 11. 아직 안 한 것

- **HTTPS 없음.** 백엔드↔모델은 VPC 내부 평문 HTTP다. 같은 VPC 안이라 수용 가능한 수준이지만,
  외부에 노출할 일이 생기면 ALB + ACM 인증서를 앞에 둔다.
- **인증 없음.** 위와 같은 이유. 보안그룹이 유일한 방어선이다.
- **오토스케일링 없음.** 1대 상시 구동. 공모전 규모에는 충분하다.
- **실사용 폰 사진 검증 없음.** val 사진은 웹 수집분이라 "손에 든/역광/바닥" 분포와 다르다
  → 배포 후 실제 사진 20장으로 확인할 것.
