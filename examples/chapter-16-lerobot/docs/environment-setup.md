# LeRobot 실습 환경 설정

## 1. 권장 환경

- Ubuntu 22.04
- NVIDIA GPU 권장
- Conda 또는 Miniconda
- Git
- Python 3.12
- `ffmpeg`

SmolVLA 평가와 ACT 학습은 연산량이 크므로 NVIDIA GPU 환경을 권장합니다. `gym_hil` 기반 시뮬레이션은 화면을 실시간으로 렌더링하므로 그래픽 환경도 필요합니다.

## 2. Conda 환경 만들기

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y ffmpeg=7.1.1 -c conda-forge
```

## 3. LeRobot 설치

작업할 폴더를 만들고 LeRobot 저장소를 내려받습니다.

```bash
mkdir -p ~/lerobot_test
cd ~/lerobot_test

git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

이어서 패키지를 설치합니다.

```bash
pip install --upgrade pip
pip install -e .
```

`pip install -e .`는 내려받은 LeRobot 소스 코드를 편집 가능한 상태로 설치하는 명령입니다. 설치 후에도 저장소의 코드 변경사항이 현재 환경에 바로 반영됩니다.

## 4. 추가 기능 설치

SmolVLA, LIBERO, HIL-SERL 시뮬레이션에 필요한 의존성을 설치합니다.

```bash
pip install -e ".[smolvla]"
pip install -e ".[libero]"
pip install -e ".[hilserl]"
```

> [!NOTE]
> 위 명령은 `pyproject.toml`이 있는 LeRobot 저장소 루트에서 실행해야 합니다. 다른 폴더에서 실행하면 `.[hilserl]`과 같은 추가 의존성 항목을 찾지 못할 수 있습니다.

## 5. 설치 확인

다음 명령으로 Python, PyTorch, CUDA, LeRobot 설치 상태를 확인합니다.

```bash
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import lerobot; print('LeRobot import OK')"
```

`CUDA: True`가 출력되면 PyTorch가 NVIDIA GPU를 인식한 상태입니다.

## 6. MuJoCo 렌더링 설정

디스플레이가 없는 서버나 원격 환경에서 평가할 때는 다음 환경변수를 사용합니다.

```bash
export MUJOCO_GL=egl
```

터미널을 새로 열 때마다 자동으로 적용하려면 `~/.bashrc`에 추가합니다.

```bash
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
source ~/.bashrc
```

로컬 데스크톱에서 MuJoCo 창을 직접 띄우는 경우에는 환경에 따라 `MUJOCO_GL=egl` 설정을 제거해야 할 수 있습니다.
