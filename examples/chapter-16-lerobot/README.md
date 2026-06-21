# 16장 LeRobot 실습 예제

이 폴더에는 『피지컬 AI 시스템 설계』 16장의 LeRobot 실습을 따라가기 위한 예제 코드와 환경 설정이 들어 있습니다.

## 실습 구성

1. LeRobot과 추가 의존성 설치
2. SmolVLA 정책을 LIBERO 환경에서 평가
3. 파이썬 코드에서 SmolVLA 정책 불러오기
4. MuJoCo 기반 Franka Panda 환경에서 시연 데이터 수집
5. 수집한 데이터로 ACT 정책 학습

## 폴더 구조

```text
chapter-16-lerobot/
├── README.md
├── docs/
│   ├── environment-setup.md
│   └── chapter-16-guide.md
├── configs/
│   └── gym_hil_record.json
└── scripts/
    ├── install_lerobot.sh
    ├── evaluate_smolvla.sh
    ├── record_sim_dataset.sh
    └── train_act.sh
```

## 빠른 시작

먼저 설치 스크립트를 실행합니다.

```bash
bash scripts/install_lerobot.sh
```

설치가 끝나면 LeRobot 가상환경을 활성화합니다.

```bash
conda activate lerobot
```

시뮬레이션 환경에서 데이터를 수집하려면 다음 명령을 실행합니다.

```bash
bash scripts/record_sim_dataset.sh
```

수집한 데이터로 ACT 정책을 학습하려면 다음 명령을 실행합니다.

```bash
bash scripts/train_act.sh
```

각 단계의 설명과 명령어 옵션은 [`docs/chapter-16-guide.md`](docs/chapter-16-guide.md)에서 확인할 수 있습니다.

> [!IMPORTANT]
> LeRobot은 지속적으로 업데이트되는 프로젝트이므로 설치 옵션이나 설정 형식이 달라질 수 있습니다. 명령이 정상적으로 실행되지 않으면 현재 설치한 LeRobot 버전의 도움말과 공식 문서를 함께 확인하세요.
