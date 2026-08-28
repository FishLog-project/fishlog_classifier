# EC2 배포 런북

모델 서버를 **전용 EC2 1대**에 컨테이너로 올리고, 앱 백엔드 EC2가 중계한다.
(decisions.md B-2, 2026-08-22 확정)

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

## 0. 현재 배포 상태 (2026-08-28 기준)

| 항목 | 값 |
|---|---|
| OS | Ubuntu 26.04 LTS (docker.io 사전 설치돼 있었다) |
| 사양 | 2 vCPU / 1.9GB / 디스크 여유 26GB |
| **사설 IP** | **`172.31.14.180`** ← 백엔드가 호출할 주소 |
| 퍼블릭 IP | 탄력적 IP 연결됨 (콘솔에서 확인. SSH 전용) |
| 컨테이너 | `fishilog` / `fishilog-ai:b0-384-20260818` / `--restart unless-stopped` |
| 워커 | 2 (`WEB_CONCURRENCY=2`), 메모리 287MB |
| 빌드 컨텍스트 | EC2의 `~/fishilog/` 에 남아 있다 (모델 교체 시 재사용) |

검증 완료: `/health` 200 · 실사진 5장 **EC2 응답 = 로컬 추론 완전 일치** ·
외부에서 8000 차단 확인.

백엔드 설정값:
```
fishlog.model.url = http://172.31.14.180:8000
```

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

## 2. 처리량 (2026-08-22 실측, 2워커 / 로컬 Docker)

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

**먼저 이미 깔려 있는지 본다.** Ubuntu 26.04 AMI에는 docker.io가 들어 있었다:

```bash
docker --version && systemctl is-active docker && docker ps
```

세 줄이 다 통과하면 설치 단계는 건너뛴다. 아니면:

```bash
# Ubuntu
sudo apt update && sudo apt install -y docker.io
sudo systemctl enable --now docker        # enable 을 빠뜨리면 재부팅 후 안 뜬다
sudo usermod -aG docker ubuntu
exit && ssh ...                            # 그룹 반영을 위해 재접속

# Amazon Linux 2023 이면 dnf install -y docker, 사용자는 ec2-user
```

사용자명은 AMI마다 다르다 — **Ubuntu는 `ubuntu`, Amazon Linux는 `ec2-user`**.

## 5. 이미지 전달 — **EC2에서 빌드하는 쪽을 쓴다**

### A. EC2에서 직접 빌드 (권장, 2026-08-28 실제로 이렇게 했다)

로컬 Docker가 **필요 없고**, 올리는 양도 195MB가 아니라 **17MB**다.
pip 패키지는 EC2가 AWS 네트워크로 받으므로 오히려 빠르고, 인바운드 트래픽은 무료다.

```bash
# 1) 로컬 — 빌드에 필요한 것만 올린다 (server/ 와 Dockerfile 이면 충분)
KEY=<키>.pem; H=ubuntu@<공인IP>
ssh -i $KEY $H 'mkdir -p ~/fishilog/server'
scp -i $KEY Dockerfile $H:~/fishilog/
scp -i $KEY server/__init__.py server/inference.py server/main.py         server/labels.json server/requirements.txt server/model.onnx $H:~/fishilog/server/

# 2) 모델 파일이 온전히 갔는지 대조한다 (16MB짜리가 조용히 잘리면 서버가 안 뜬다)
md5sum server/model.onnx
ssh -i $KEY $H 'md5sum ~/fishilog/server/model.onnx'

# 3) EC2에서 빌드 (2~3분)
ssh -i $KEY $H 'cd ~/fishilog && docker build -t fishilog-ai:b0-384-20260818 -t fishilog-ai:latest .'
```

`.dockerignore` 없이도 컨텍스트가 17MB라 빠르다(`data/` 를 애초에 안 올리므로).

### B. docker save | ssh (로컬 이미지를 그대로 옮기고 싶을 때)

```bash
docker save fishilog-ai:b0-384-20260818 | gzip   | ssh -i <키>.pem ubuntu@<공인IP> 'gunzip | docker load'
```

⚠️ **이 파이프라인은 실패를 조용히 삼킨다.** 2026-08-28에 로컬 Docker Desktop이 내려간
줄 모르고 실행했더니, `docker save` 가 빈 스트림을 내보내고 원격에서
`unrecognized image format` 이 떴는데 **전체 종료 코드는 0** 이었다(파이프의 마지막
명령 기준). 성공한 줄 알고 넘어가기 쉽다.

쓸 거면 **전송 후 반드시 확인**한다:
```bash
ssh -i <키>.pem ubuntu@<공인IP> 'docker images fishilog-ai'
```

### C. ECR (배포가 잦아지면)

```bash
aws ecr create-repository --repository-name fishilog-ai
aws ecr get-login-password --region ap-northeast-2   | docker login --username AWS --password-stdin <계정ID>.dkr.ecr.ap-northeast-2.amazonaws.com
docker tag fishilog-ai:b0-384-20260818 <계정ID>.dkr.ecr.ap-northeast-2.amazonaws.com/fishilog-ai:b0-384-20260818
docker push <계정ID>.dkr.ecr.ap-northeast-2.amazonaws.com/fishilog-ai:b0-384-20260818
```

인스턴스에 `AmazonEC2ContainerRegistryReadOnly` IAM 역할을 붙이고 `docker pull`.

**Graviton(arm64)을 쓴다면 A 방식뿐이다** — 로컬(amd64)에서 만든 이미지는 실행되지 않는다.

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

빌드 컨텍스트가 EC2의 `~/fishilog/` 에 남아 있으므로 **바뀐 파일만 덮어쓰면 된다.**

```bash
# 1) 로컬: 새 체크포인트 → ONNX → 검증 → 임계값
python -m src.export_onnx --ckpt models/<새모델>.pt --version b0-384-<날짜>
python -m scripts.check_preprocess && python -m scripts.check_server
python -m scripts.tune_threshold --write        # 임계값은 모델마다 다시 잰다

# 2) 바뀐 파일만 전송 (labels.json 도 함께 — 임계값·버전이 여기 들어 있다)
scp -i <키>.pem server/model.onnx server/labels.json ubuntu@<공인IP>:~/fishilog/server/

# 3) EC2에서 재빌드·교체
ssh -i <키>.pem ubuntu@<공인IP>
cd ~/fishilog && docker build -t fishilog-ai:b0-384-<날짜> .
docker rm -f fishilog
docker run -d --name fishilog --restart unless-stopped -p 8000:8000 \
  -e WEB_CONCURRENCY=2 fishilog-ai:b0-384-<날짜>
curl -s localhost:8000/health      # model_version 이 바뀌었는지 확인
```

`labels.json` 을 빼먹으면 새 모델에 **옛 임계값·옛 버전 문자열**이 붙는다.
`/health` 의 `model_version` 이 안 바뀌었다면 이걸 의심한다.

**임계값을 다시 재는 것을 건너뛰지 말 것.** 모델이 바뀌면 확률 분포가 바뀌어
같은 0.25가 전혀 다른 거부율을 뜻하게 된다.

이전 이미지는 지우지 말고 남겨둔다 — 문제가 생기면 `docker run` 한 줄로 되돌린다.

## 10. 접속이 안 될 때 — 2026-08-28에 실제로 겪은 순서

배포에서 시간을 가장 많이 쓴 곳이 **네트워크**였다. 순서대로 의심할 것:

### ① 사설 IP로 SSH를 시도하고 있지 않은가

`172.31.x.x` 는 **VPC 내부 전용 주소**다. 내 노트북에서는 어떤 보안그룹 설정으로도
닿을 수 없다. 콘솔 세부 정보에 IP 칸이 여러 개 나란히 있어 **프라이빗 IPv4**를
복사하기 쉽다. SSH는 **퍼블릭 IPv4**로 한다.

| 주소 | 쓰는 곳 |
|---|---|
| 프라이빗 `172.31.14.180` | 백엔드 → 모델 (8000) |
| 퍼블릭 (EIP) | 내 노트북 → 모델 (22) |

### ② 퍼블릭 IP가 아예 없지 않은가

세부 정보의 **퍼블릭 IPv4 주소**가 `-` 면 인스턴스 생성 시 자동 할당이 꺼져 있었던
것이다. **탄력적 IP를 연결하면 재생성 없이 해결된다**(EC2 → 탄력적 IP → 할당 → 작업 →
연결). 기본 VPC라면 IGW가 이미 있으므로 EIP만으로 충분하다.

> 비용: 2024년 2월부터 AWS는 **모든 퍼블릭 IPv4에 동일하게 과금**한다(시간당 $0.005,
> 월 약 $3.6). 자동 할당이든 EIP든 요금이 같으므로 "EIP라서 더 비싸다"는 것은 옛 규칙이다.
> 다만 **인스턴스에 연결하지 않고 방치한 EIP도 과금**되니 프로젝트가 끝나면 릴리스한다.

### ③ 22번 소스를 백엔드 보안그룹으로 잡지 않았는가

실제로 이 실수를 했다. **22와 8000은 소스가 다르다:**

| 포트 | 소스 | 이유 |
|---|---|---|
| 22 | **내 IP `/32`** | 내가 관리하려고 |
| 8000 | **백엔드 SG** | 백엔드만 추론을 호출 |

22를 백엔드 SG로 잡으면 백엔드에서만 SSH가 되고 내 노트북은 막힌다.
집·카페 IP는 바뀔 수 있으니, 어느 날 갑자기 타임아웃이 나면 내 IP부터 다시 본다.

### 진단: TIMEOUT과 REFUSED를 구분한다

```bash
python -c "
import socket
for host,port in [('<공인IP>',22),('<공인IP>',8000),('github.com',22)]:
    s=socket.socket(); s.settimeout(6)
    try: s.connect((host,port)); print('OPEN   ',host,port)
    except socket.timeout: print('TIMEOUT',host,port)
    except Exception as e: print('REFUSED',host,port,type(e).__name__)
    finally: s.close()"
```

| 결과 | 뜻 |
|---|---|
| `TIMEOUT` | **방화벽이 조용히 버렸다** — 보안그룹·NACL·주소 문제 |
| `REFUSED` | 서버까지 도달했다. 포트에 아무것도 안 떠 있을 뿐 — **네트워크는 정상** |
| `OPEN` | 통과 |

`github.com:22` 를 함께 찔러 **내 네트워크가 22번을 막는지** 가른다. 그쪽이 OPEN인데
EC2만 TIMEOUT이면 원인은 AWS 쪽이다.

**22와 8000이 동시에 TIMEOUT이면** 규칙 하나를 빠뜨린 게 아니라, 보안그룹 인바운드가
비었거나 **인스턴스에 붙은 SG가 규칙을 넣은 SG와 다른 것**이다(인스턴스 → 보안 탭에서
`sg-` 를 대조한다).

### 정상 상태의 모습

- 내 노트북 → 22 : **OPEN**
- 내 노트북 → 8000 : **TIMEOUT** ← 막혀 있어야 맞다. 뚫리면 SG를 잘못 연 것이다
- 백엔드 → 8000 : **OPEN**

## 11. 운영

| 상황 | 확인 |
|---|---|
| 응답이 없다 | `docker ps`, `docker logs --tail 100 fishilog` |
| `/health` 503 | 모델 로드 실패. `detail` 필드에 원인이 들어 있다 |
| 느리다 | `docker stats` 로 CPU 100% 여부 → `WEB_CONCURRENCY` 상향 또는 인스턴스 확대 |
| 재부팅 후 안 뜸 | `sudo systemctl enable docker` 를 안 한 것 |

로그는 컨테이너 stdout으로만 남는다. 장기 보관이 필요하면 CloudWatch Logs 드라이버를 붙인다:
`--log-driver=awslogs --log-opt awslogs-group=/fishilog/model`

## 12. 아직 안 한 것

- **HTTPS 없음.** 백엔드↔모델은 VPC 내부 평문 HTTP다. 같은 VPC 안이라 수용 가능한 수준이지만,
  외부에 노출할 일이 생기면 ALB + ACM 인증서를 앞에 둔다.
- **인증 없음.** 위와 같은 이유. 보안그룹이 유일한 방어선이다.
- **오토스케일링 없음.** 1대 상시 구동. 공모전 규모에는 충분하다.
- **폰 사진 검증 없음.** 2026-08-28에 배포 서버로 17장을 확인했으나(어종 15/15,
  `기타` 2/2) 전부 **웹 사진**이었다(EXIF에 카메라 없음, 340~910px). 실제 앱이 받는
  3000px+ 폰 사진 — 손에 든/역광/젖은 바닥 — 은 아직 한 장도 넣어보지 않았다.
  → `python -m scripts.try_photos <폴더>` 로 확인할 것.
- ~~백엔드 → 모델 호출 확인~~ **완료(2026-08-28).** 백엔드 EC2에서 사설 IP로 요청이
  닿고 JSON 응답을 받는 것을 확인했다.
