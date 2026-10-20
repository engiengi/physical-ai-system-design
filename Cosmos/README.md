# Cosmos 실험

설치부터 로컬·Brev 실행까지는 [실행방법.md](실행방법.md)를 참고하세요.

- `spark/src`, `spark/scripts`, `spark/configs`, `spark/tests`: 시뮬레이터 코드와 실험 도구
- `thor`: 추론 코드
- `ops`, `lab`: 배포와 원격 제어
- `spark/runs`, `spark/reports`, `spark/captures`, `spark/thor_results`: 로컬 실험 결과
- `spark/.venv`, `spark/.tools`, `spark/RoboLab`, `spark/external`: 로컬 실행 환경

모델·가상환경·외부 소스·녹화·결과는 Git에 포함되지 않습니다.
실험 결과는 실행 후 생성되며, 클라우드 서버를 삭제하기 전에 별도로 백업합니다.
