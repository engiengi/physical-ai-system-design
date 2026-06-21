# 16장 LeRobot 실습 가이드

## 1. LeRobot의 기본 파이프라인

LeRobot은 로봇 학습에 필요한 데이터셋, 정책 모델, 평가 환경, 실제 로봇 인터페이스를 하나의 흐름으로 연결합니다.

```text
관측 데이터
  ↓
전처리
  ↓
정책 모델
  ↓
행동 예측
  ↓
후처리
  ↓
시뮬레이터 또는 실제 로봇
```

관측 데이터에는 카메라 이미지, 관절 상태, 작업 지시 등이 포함될 수 있습니다. 정책 모델은 이 입력을 받아 로봇이 실행할 액션을 생성합니다.

---

## 2. SmolVLA를 LIBERO에서 평가하기

LIBERO 환경에 맞게 파인튜닝된 SmolVLA 체크포인트를 불러와 정책의 동작을 확인해보겠습니다.

```bash
export MUJOCO_GL=egl

lerobot-eval   --policy.path=HuggingFaceVLA/smolvla_libero   --env.type=libero   --env.task=libero_object   --eval.batch_size=1   --eval.n_episodes=3
```

`libero_object`는 여러 개의 세부 작업으로 구성된 평가 suite입니다. 환경 구성에 따라 `--eval.n_episodes=3`이 각 세부 작업에 반복 적용될 수 있으므로 예상보다 많은 에피소드가 실행될 수 있습니다.

여러 suite를 연속으로 평가하려면 다음과 같이 실행합니다.

```bash
lerobot-eval   --policy.path=HuggingFaceVLA/smolvla_libero   --env.type=libero   --env.task=libero_spatial,libero_object,libero_goal,libero_10   --eval.batch_size=1   --eval.n_episodes=10   --output_dir=./outputs/eval/smolvla_libero
```

평가가 끝나면 성공률과 함께 렌더링 영상도 확인합니다. 실패 장면을 보면 정책이 어느 단계에서 어려움을 겪는지 구분하기 쉽습니다.

- 목표 물체를 찾지 못하는 경우: 시각 인식 또는 작업 지시 이해 문제
- 물체에는 접근하지만 잡지 못하는 경우: 액션 정밀도 문제
- 물체를 잡은 뒤 놓치는 경우: 궤적 안정성 또는 그리퍼 제어 문제
- 행동 청크가 바뀌는 시점에 움직임이 튀는 경우: 청크 연결 또는 추론 지연 문제

---

## 3. 파이썬에서 SmolVLA 행동 예측하기

다음 코드는 정책, 전처리기, 후처리기, 데이터셋을 직접 연결하는 기본 흐름을 보여줍니다.

```python
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

model_id = "HuggingFaceVLA/smolvla_libero"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = SmolVLAPolicy.from_pretrained(model_id)
policy.to(device)
policy.eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={
        "device_processor": {
            "device": str(device),
        }
    },
)

dataset = LeRobotDataset("lerobot/libero", episodes=[0])

frame = dataset[0]

obs = preprocess(frame)

with torch.inference_mode():
    action = policy.select_action(obs)

action = postprocess(action)

print("예측된 행동값 형태:", action.shape)
# 예시 출력: torch.Size([1, 7])
# [배치 크기, 액션 차원]
# 액션 차원 7 = Δx, Δy, Δz (EEF 위치 델타 3) + Δroll, Δpitch, Δyaw (EEF 자세 델타 3) + gripper (1)
```

`LeRobotDataset`의 `episodes` 파라미터에 에피소드 번호 목록을 전달하면 해당 에피소드만 로드합니다. `dataset[0]`은 로드된 데이터셋의 첫 번째 프레임을 반환합니다.

`select_action`은 내부 큐를 관리하며 한 번에 1스텝의 행동을 반환합니다. 모델은 내부적으로 `chunk_size`만큼의 행동을 한 번에 예측한 뒤 큐에 쌓아두고, 호출할 때마다 `n_action_steps`만큼 꺼내 줍니다. `HuggingFaceVLA/smolvla_libero` 체크포인트는 `n_action_steps=1`로 설정되어 있어 출력 형태가 `[1, 7]`입니다.

청크 전체를 한꺼번에 받으려면 `predict_action_chunk`를 사용합니다.

```python
with torch.inference_mode():
    action_chunk = policy.predict_action_chunk(obs)
print("청크 전체 형태:", action_chunk.shape)
# 예시 출력: torch.Size([1, 50, 7])
# [배치 크기, 청크 길이, 액션 차원]
```

각 액션 차원의 의미는 데이터셋과 로봇 구성에 따라 달라집니다. LIBERO/Franka Panda 환경은 OSC_POSE 컨트롤러를 사용해 액션을 7차원으로 정의하지만, 다른 로봇이나 컨트롤러를 사용하는 환경에서는 차원 수와 의미가 달라집니다.

---

## 4. MuJoCo Franka Panda에서 데이터 수집하기

실제 로봇이 없어도 `gym_hil`의 MuJoCo 기반 Franka Panda 환경에서 키보드로 로봇을 조작하며 시연 데이터를 만들 수 있습니다.

### 4.1 설정 파일 확인

데이터 수집 설정은 [`configs/gym_hil_record.json`](../configs/gym_hil_record.json)에 들어 있습니다.

설정 파일에서 다음 값을 자신의 환경에 맞게 수정합니다.

```json
"repo_id": "username/sim_pick_cube"
```

`username`은 데이터셋을 구분할 수 있는 이름으로 바꿉니다. Hugging Face Hub에 업로드하려면 자신의 Hugging Face 사용자명을 사용하면 됩니다.

현재 설정에서는 다음 값이 사용됩니다.

```json
"push_to_hub": false
```

따라서 수집한 데이터는 Hugging Face Hub에 자동으로 업로드되지 않고 로컬에 저장됩니다.

### 4.2 데이터 수집 실행

LeRobot 저장소 루트에서 다음 명령을 실행합니다.

```bash
python -m lerobot.rl.gym_manipulator   --config_path /절대/경로/chapter-16-lerobot/configs/gym_hil_record.json
```

저장소에 포함된 실행 스크립트를 이용할 수도 있습니다.

```bash
bash scripts/record_sim_dataset.sh
```

명령을 실행하면 MuJoCo 기반의 Franka Panda 시뮬레이션 창이 열립니다. 가상 로봇의 말단부와 그리퍼를 조작해 큐브를 집는 동작을 수행합니다. 각 에피소드가 끝나면 조작 과정의 관측값과 액션이 데이터셋으로 저장됩니다.

### 4.3 키보드 조작

| 키 | 기능 |
|---|---|
| 방향키 `↑` `↓` `←` `→` | 로봇 말단부를 X-Y 평면에서 이동 |
| `Shift` / 오른쪽 `Shift` | Z축 방향 이동 |
| 왼쪽 `Ctrl` / 오른쪽 `Ctrl` | 그리퍼 열기 또는 닫기 |
| `Enter` | 현재 에피소드를 성공으로 종료 |
| `Backspace` | 현재 에피소드를 실패로 종료 |
| `Space` | 사용자 조작 시작 또는 중지 |
| `Esc` | 프로그램 종료 |

> [!NOTE]
> LeRobot 버전에 따라 키 매핑이 달라질 수 있습니다. 시뮬레이션 창이 열렸지만 조작이 되지 않는다면 실행 중인 터미널 메시지와 현재 설치한 버전의 `gym_hil` 안내를 확인하세요.

---

## 5. ACT 정책 학습하기

앞에서 수집한 데이터셋을 이용해 ACT 정책을 학습합니다.

```bash
lerobot-train   --dataset.repo_id=username/sim_pick_cube   --policy.type=act   --output_dir=outputs/train/act_sim_pick_cube   --job_name=act_sim_pick_cube   --policy.device=cuda   --policy.push_to_hub=false   --wandb.enable=false
```

각 옵션의 의미는 다음과 같습니다.

| 옵션 | 설명 |
|---|---|
| `--dataset.repo_id` | 학습에 사용할 데이터셋 |
| `--policy.type=act` | ACT 정책 사용 |
| `--output_dir` | 체크포인트와 로그 저장 위치 |
| `--job_name` | 실험 이름 |
| `--policy.device=cuda` | NVIDIA GPU에서 학습 |
| `--policy.push_to_hub=false` | 학습 결과를 Hugging Face Hub에 자동으로 업로드하지 않음 |
| `--wandb.enable=false` | Weights & Biases 기록 비활성화 |

학습 결과는 다음 폴더에 저장됩니다.

```text
outputs/train/act_sim_pick_cube/
```

학습이 끝난 뒤에는 저장된 체크포인트를 같은 시뮬레이션 환경에서 평가하거나 다른 LeRobot 예제와 연결할 수 있습니다.

> [!WARNING]
> LeRobot 버전에 따라 학습 옵션과 설정 형식이 달라질 수 있습니다. `DatasetConfig` 필드나 명령행 옵션과 관련된 오류가 나타나면 다음 명령으로 현재 버전이 지원하는 옵션을 확인하세요.
>
> ```bash
> lerobot-train --help
> ```

---

## 6. 다음 단계로 확장하기

이 실습을 완료하면 다음과 같은 방향으로 프로젝트를 확장할 수 있습니다.

1. 실제 로봇 또는 다른 시뮬레이션 환경 연결
2. 카메라, 관절, 그리퍼 데이터를 LeRobotDataset 형식으로 저장
3. ACT, Diffusion Policy, SmolVLA 등 작업에 맞는 정책 선택
4. 학습한 정책을 동일 환경에서 평가
5. 실패한 에피소드를 추가로 수집해 데이터셋 보강
6. 실제 로봇에 배포하기 전 안전 제한과 수동 정지 기능 검증

커스텀 로봇을 연결할 때는 LeRobot의 Robot 인터페이스를 구현하고, 관측 키와 액션 차원을 데이터셋 및 정책 규약에 맞춰야 합니다.
