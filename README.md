# JAX to TensorFlow.js 데모

JAX와 Flax를 사용하여 MNIST 데이터셋으로 CNN 모델을 학습하고, TensorFlow.js로 웹에서 추론하는 예제 프로젝트입니다.

## 프로젝트 구조

```
jax-to-tfjs-demo/
├── src/
│   └── jax_to_tfjs_demo/
│       ├── models/
│       │   ├── jax_mnist_cnn.py      # JAX 기반 MNIST CNN 모델
│       │   └── flax_mnist_cnn.py     # Flax 기반 MNIST CNN 모델
│       ├── train_jax.py              # JAX 모델 학습 스크립트
│       ├── train_flax.py             # Flax 모델 학습 스크립트
│       ├── train_all.py              # 모든 모델 학습 스크립트
│       ├── convert_to_tfjs.py        # TensorFlow.js 변환 스크립트
│       ├── check_checkpoints.py      # 체크포인트 관리 스크립트
│       ├── examine_checkpoint.py     # 체크포인트 상세 분석 스크립트 
│       ├── verify_tfjs_model.py      # 변환된 모델 검증 스크립트
│       └── app.py                    # 통합 CLI 도구
└── checkpoints/
    ├── jax_mnist/                    # JAX 모델 체크포인트
    └── flax_mnist/                   # Flax 모델 체크포인트
```

## 설치 및 실행

### 필수 요구사항

- Python 3.8 이상
- [rye](https://rye-up.com/)

### 설치

프로젝트 의존성 설치:

```bash
rye sync
```

## 통합 CLI 도구 사용법

모든 기능을 통합한 CLI 도구로 학습, 변환, 검증을 한 곳에서 수행할 수 있습니다:

```bash
rye run app help
```

### 주요 명령어

#### 도움말 보기

```bash
rye run app help
```

#### 모델 학습

JAX 모델 학습:
```bash
rye run app train jax --epochs 10 --batch-size 32
```

Flax 모델 학습:
```bash
rye run app train flax --epochs 10 --batch-size 32
```

#### 모델 변환

모델을 TensorFlow.js로 변환:
```bash
rye run app convert --model-type jax_mnist --checkpoint-path checkpoints/jax_mnist/1
```

양자화를 적용하여 모델 변환:
```bash
rye run app convert --model-type jax_mnist --checkpoint-path checkpoints/jax_mnist/1 --quantization uint8
```

#### 모델 검증

변환된 모델 검증:
```bash
rye run app verify tfjs_models/jax_mnist
```

#### 체크포인트 관리

사용 가능한 체크포인트 확인:
```bash
rye run app check
```

체크포인트 상세 검사:
```bash
rye run app examine --checkpoint-path checkpoints/jax_mnist/1
```

#### 모든 모델 학습

모든 모델 유형 학습:
```bash
rye run app train-all
```

## 개별 스크립트 사용법

### 모델 학습

#### JAX 기반 모델 학습

기본 설정으로 학습:
```bash
rye run train-jax
```

사용자 정의 설정으로 학습:
```bash
rye run train-jax --epochs 10 --lr 0.0005
```

#### Flax 기반 모델 학습

기본 설정으로 학습:
```bash
rye run train-flax
```

사용자 정의 설정으로 학습:
```bash
rye run train-flax --epochs 10 --lr 0.0005
```

#### 모든 모델 학습

기본 설정으로 모든 모델 학습:
```bash
rye run train-all
```

특정 모델만 학습:
```bash
rye run train-all --model jax
rye run train-all --model flax
```

사용자 정의 설정으로 학습:
```bash
rye run train-all --epochs 10 --lr 0.0005 --model all
```

### TensorFlow.js로 변환

학습된 모델을 TensorFlow.js 형식으로 변환:

```bash
rye run convert
```

대화형 모드로 변환:
```bash
rye run convert-interactive
```

특정 모델과 체크포인트 지정:
```bash
rye run convert --model-type jax_mnist --checkpoint-path checkpoints/jax_mnist/1
```

양자화 적용:
```bash
rye run convert --model-type jax_mnist --checkpoint-path checkpoints/jax_mnist/1 --quantization uint8
```

### 체크포인트 관리

#### 체크포인트 디렉토리 준비

모델 학습을 시작하기 전에 체크포인트 디렉토리 구조를 생성합니다:

```bash
rye run prepare-ckpt
```

#### 사용 가능한 체크포인트 확인

학습된 모델의 체크포인트 목록을 확인합니다:

```bash
rye run check-ckpt
```

#### 체크포인트 상세 검사

특정 체크포인트의 구조와 내용을 자세히 검사합니다:

```bash
rye run examine-ckpt
```

특정 모델 타입과 스텝을 지정하여 검사:
```bash
rye run examine-ckpt --model-type jax_mnist --step 1
```

#### 변환된 모델 검증

TensorFlow.js로 변환된 모델을 검증합니다:

```bash
rye run verify-tfjs --model-path tfjs_models/jax_mnist
```

## 코드 세부 설명

### 모델 구현 (`models/`)

#### JAX 모델 (`jax_mnist_cnn.py`)

JAX 기반 CNN 모델 구현:
- 저수준 JAX API를 사용하여 CNN 모델 구현
- 파라미터 초기화 및 수동 관리
- 순전파 및 역전파 로직 구현

#### Flax 모델 (`flax_mnist_cnn.py`)

Flax 기반 CNN 모델 구현:
- `nn.Module`을 사용한 객체지향 구현
- 자동 파라미터 초기화 및 관리
- 구조화된 모델 정의

### 학습 스크립트

#### JAX 학습 (`train_jax.py`)

- JAX 기반 모델을 학습하기 위한 스크립트
- 데이터 로딩 및 전처리
- 학습 루프 및 손실 함수 구현
- Orbax를 사용한 체크포인트 저장

#### Flax 학습 (`train_flax.py`)

- Flax 기반 모델을 학습하기 위한 스크립트
- Flax의 `TrainState`를 사용한 학습 상태 관리
- 학습 루프 및 모델 업데이트 로직
- Orbax를 사용한 체크포인트 저장

#### 통합 학습 (`train_all.py`)

- 모든 모델 유형을 한 번에 학습하는 스크립트
- 명령행 인자 처리 및 학습 설정 관리
- 다양한 모델 유형에 대한 학습 실행

### 변환 및 검증 스크립트

#### TensorFlow.js 변환 (`convert_to_tfjs.py`)

- JAX/Flax 모델을 TensorFlow.js 형식으로 변환
- JAX 모델을 TensorFlow로 변환하는 로직
- TensorFlow 모델을 TensorFlow.js로 변환
- 양자화 옵션을 통한 모델 최적화
- 변환된 모델 검증

#### 체크포인트 관리 (`check_checkpoints.py`)

- 모델 체크포인트 관리 및 확인
- 체크포인트 디렉토리 생성 및 구조화
- 사용 가능한 체크포인트 나열
- 체크포인트 검사 및 관리

#### 체크포인트 검사 (`examine_checkpoint.py`)

- 체크포인트 파일 구조 상세 분석
- 체크포인트 메타데이터 검사
- 모델 파라미터 확인
- 차원 및 데이터 형식 분석

#### 모델 검증 (`verify_tfjs_model.py`)

- 변환된 TensorFlow.js 모델 검증
- 테스트 데이터로 모델 성능 검증
- 원본 모델과 변환된 모델 비교
- 양자화된 모델의 정확도 측정

### 통합 CLI 도구 (`app.py`)

- 모든 기능을 통합한 명령행 인터페이스
- 컬러 출력으로 가독성 향상
- 명령어 파싱 및 실행
- 자세한 도움말 및 예시 제공
- 학습, 변환, 검증을 위한 통합 인터페이스
- 명령 실행 결과 및 오류 관리

## 모델 설명

### JAX 기반 모델

순수 JAX를 사용한 CNN 구현:
- **컨볼루션 레이어 1**: 32개 필터 (3x3)
- **컨볼루션 레이어 2**: 64개 필터 (3x3)
- **완전 연결 레이어 1**: 128개 뉴런
- **출력 레이어**: 10개 뉴런 (MNIST 클래스 수)
- JAX의 저수준 API를 사용한 직접적인 구현
- 수동 파라미터 관리

### Flax 기반 모델

Flax를 사용한 고수준 CNN 구현:
- **컨볼루션 레이어 1**: 32개 필터 (3x3)
- **컨볼루션 레이어 2**: 64개 필터 (3x3)
- **완전 연결 레이어 1**: 128개 뉴런
- **출력 레이어**: 10개 뉴런 (MNIST 클래스 수)
- Flax의 `nn.Module`을 사용한 클래스 기반 구현
- 자동 파라미터 관리

## 공통 기능

두 구현 모두 다음 기능을 공유합니다:
- Adam 옵티마이저 사용
- 배치 크기 32
- 기본 학습률 0.001 (명령행 인자로 조정 가능)
- 기본 5 에포크 학습 (명령행 인자로 조정 가능)
- Orbax를 사용한 체크포인트 관리
- 최대 3개의 체크포인트 유지
- 각 에포크마다 체크포인트 저장

## 모델 양자화

TensorFlow.js로 변환 시 다음 양자화 옵션을 지원합니다:
- **없음 (None)**: 양자화 없이 원본 정밀도 유지
- **uint8**: 8비트 양자화 적용하여 모델 크기 대폭 감소 (약 4배)
- **uint16**: 16비트 양자화 적용하여 정밀도와 크기의 균형 (약 2배 감소)

양자화 적용 시 고려사항:
- 모델 크기와 로딩 시간 개선
- 일부 정확도 손실 가능성
- 추론 속도 향상
- 양자화 검증을 통한 품질 보장

## 체크포인트 구조

JAX/Flax 모델의 체크포인트는 다음 구조를 가집니다:
- `checkpoints/jax_mnist/1/`: 저장된 체크포인트 디렉토리
  - `_CHECKPOINT_METADATA`: 체크포인트 메타데이터
  - `model/`: 모델 파라미터 및 상태
    - `_METADATA`: 모델 메타데이터
    - `_sharding`: 샤딩 정보
    - `manifest.ocdbt`: 매니페스트 파일
    - 기타 모델 파라미터 디렉토리

## 의존성

- jax>=0.4.20
- flax>=0.8.1
- tensorflow>=2.15.0
- tensorflowjs>=4.15.0
- orbax-checkpoint>=0.4.1
- tensorflow-datasets>=4.9.4
- optax>=0.1.7
- inquirer>=3.1.3
- tqdm>=4.66.0

## 참고 자료

- [JAX 공식 문서](https://jax.readthedocs.io/)
- [Flax 공식 문서](https://flax.readthedocs.io/)
- [TensorFlow.js 공식 문서](https://www.tensorflow.org/js)
- [Orbax 체크포인트 문서](https://github.com/google/orbax)
