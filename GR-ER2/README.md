# GR-ER2 · Gemini Robotics ER 2 실습

Gemini Robotics ER 2 API의 이미지 추론부터 Isaac Sim의 로봇 조작·Live 복구·멀티로봇 협업까지 실행하는 교육용 프로젝트입니다.

**[실행방법](./실행방법.md)**에서 설치와 API 등록부터 시작하세요.

| 할 일 | 매뉴얼 |
|---|---|
| GPU 없이 내 이미지로 API 사용 | [1.1~1.3 설치](./실행방법.md#step-1) → [2 API 등록·첫 요청](./실행방법.md#step-2) |
| 내 워크스테이션에서 로봇 실습 | [1.4~1.7 GPU·Docker·Isaac Sim](./실행방법.md#gpu-driver) |
| GPU 없는 PC에서 Brev로 로봇 실습 | [1.8 서버 생성](./실행방법.md#brev-setup) → [1.9 설치](./실행방법.md#brev-install) → [2 API](./실행방법.md#step-2) → [7.5 브라우저 GUI](./실행방법.md#brev-gui) |
| Brev 설치 후 다시 실행·결과 회수 | [7.5 터미널 A·B·C](./실행방법.md#brev-gui) → [9.1 회수·삭제](./실행방법.md#brev-cleanup) |
| 이미 설치된 환경에서 다시 실행 | [0.2 재시작](./실행방법.md#existing-install) |
| 이미지 비교·수동 조작·한 블록/세 블록 | [3 시작](./실행방법.md#step-3) → [4 비교](./실행방법.md#step-4) → [5 조작](./실행방법.md#step-5) → [6 모델 지시](./실행방법.md#step-6) |
| 영상 기반 방해 감지·복구 | [6.5~6.6 Live](./실행방법.md#live-recovery-experiment) |
| 서랍·Spot·두 Franka·자재 운송 | [6.7 이후 선택 실험](./실행방법.md#drawer-scenario) |
| 직접 모니터·원격 전체 3D 화면 | [7 화면 모드](./실행방법.md#step-7) |
| 결과 회수와 종료 | [8 결과](./실행방법.md#step-8) → [9 종료](./실행방법.md#step-9) |

## 실행 구조

- 호스트 Python 3.12 가상환경: 웹 GUI, Google API 요청, 입력·응답·평가 기록.
- Isaac Sim `6.0.0-dev2` 컨테이너: 카메라, 물리, 로봇 하위 제어. NVIDIA RTX 워크스테이션 또는 Brev L40S 서버에서 실행합니다.
- Brev 전체 GUI: noVNC + SSH 터널. 개인 PC에는 GPU가 필요하지 않고, 서버에 공개 포트를 열지 않습니다.
- Google API: 일반 ER 2 / Gemini 비교, ER 2 Streaming, 사후 영상 진행률 판단.
- `outputs/`: 실행별 입력·응답·영상·평가 결과.

기본 블록 실습은 고정 카메라 RGB를 판단 시점마다 전달하고 같은 관측의 깊이로 좌표를 변환합니다. Live 실습은 카메라 프레임을 계속 전송합니다. 모델과 제어기의 역할은 [입력·출력과 하위 제어의 연결](./docs/폐루프_구성과_모델_제어기_역할.md)에서 설명합니다.

## 코드와 자료

| 경로 | 내용 |
|---|---|
| `configs/default.json` | 모델 이름·API 제한·시뮬레이션 설정 |
| `scripts/` | 가상환경 설치·시뮬레이터·웹 실행 |
| `src/app.py`, `src/index.html` | 웹 GUI·이미지 비교·도구 호출 |
| `src/simulation.py` | 블록·RGB-D·Franka 제어 |
| `src/live_recovery.py`, `src/presentation.py` | Live 복구·입출력 통합 영상 |
| `src/scenario_*.py`, `src/drawer_motion.py` | 서랍·Spot 실험 |
| `src/coop_*.py`, `src/transport_*.py` | 협업·자재 운송·진행률 평가 |
| `tests/` | CPU 자동 테스트·문서 명령/링크 검사 |
| `data/samples/` | 시뮬레이터 시작 시 생성하는 샘플 |
| `runtime/` | 현재 카메라·대기열·로그, Git 제외 |

## 참고 자료

- 실험 결과: [블록 조작·복구](./docs/검증결과.md) · [스트리밍](./docs/스트리밍_시나리오_결과.md) · [서랍·Spot](./docs/새환경_실험결과.md) · [멀티로봇 협업](./docs/협업_실험결과.md)
- [Google ER 2 공식 문서](https://ai.google.dev/gemini-api/docs/robotics-overview)
- [NVIDIA Isaac Sim 요구사항](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)
