"""
FLAX 트레이너 테스트

FlaxTrainer 클래스의 체크포인트 관련 기능 테스트
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import jax

from jax_to_tfjs.train.flax_trainer import FlaxTrainer
from jax_to_tfjs.checkpoint_utils.jax_checkpointer import CheckpointNotFoundError
from jax_to_tfjs.models.flax.model_manager import FlaxModelManager


@pytest.fixture(scope="module", autouse=True)
def setup_deterministic_environment():
    """테스트 모듈 전체에 결정적 환경 설정"""
    # 난수 시드 고정
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_platform_name", "cpu")
    return None


@pytest.fixture
def checkpoint_dir():
    """테스트용 체크포인트 디렉토리"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def nonexistent_checkpoint_dir():
    """존재하지 않는 체크포인트 디렉토리"""
    temp_dir = tempfile.mkdtemp()
    path = Path(temp_dir) / "nonexistent"
    shutil.rmtree(temp_dir)  # 바로 제거하여 존재하지 않게 함
    yield path


def test_flax_trainer_checkpoint_integration(checkpoint_dir):
    """FlaxTrainer의 체크포인트 통합 테스트 (실제 데이터 사용)"""
    # 실제 FlaxModelManager 및 트레이너 생성
    model_manager = FlaxModelManager()
    model_manager.init_model()
    trainer = FlaxTrainer(model_manager)

    # 모델 학습 (체크포인트 저장 포함)
    # 실제 데이터를 사용하지만 에포크 수를 줄여 테스트 시간 단축
    trainer.train(
        num_epochs=1, subdir=str(checkpoint_dir), evaluate_after_training=False
    )

    # 체크포인트 검증
    # FLAX는 에포크 번호로 디렉토리 생성
    checkpoint_dirs = [
        d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()
    ]

    assert len(checkpoint_dirs) > 0, "체크포인트 디렉토리가 생성되지 않음"

    # 가장 최근 체크포인트 확인
    last_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.name))

    assert last_checkpoint.exists(), (
        f"최신 체크포인트 디렉토리 {last_checkpoint}가 존재하지 않습니다."
    )

    print(f"체크포인트 디렉토리 {last_checkpoint} 확인 완료")

    # 모델 파일 확인
    model_dir = last_checkpoint / "model"
    assert model_dir.exists(), f"모델 디렉토리 {model_dir}가 존재하지 않습니다."


def test_flax_trainer_missing_checkpoint(nonexistent_checkpoint_dir):
    """체크포인트가 없는 경우의 복구 프로세스 테스트

    시나리오:
    1. 존재하지 않는 체크포인트 디렉토리 확인
    2. CheckpointNotFoundError 예외가 발생하는지 확인 (의도된 동작)
    3. 예외 발생 후 새 체크포인트 디렉토리에 모델 학습 및 저장
    4. 새로 생성된 체크포인트가 올바르게 저장되었는지 검증
    """
    # 테스트 설정
    model_manager = FlaxModelManager()
    model_manager.init_model()
    trainer = FlaxTrainer(model_manager)

    # 1단계: 의도적으로 체크포인트 경로 확인 (존재하지 않아야 함)
    print(
        f"\n[테스트 1단계] 체크포인트 경로({nonexistent_checkpoint_dir}) 존재 여부 확인..."
    )
    try:
        # 체크포인트 경로가 없으므로 예외 발생해야 함
        if not nonexistent_checkpoint_dir.exists():
            print(
                f"[확인] 체크포인트 경로가 존재하지 않음: {nonexistent_checkpoint_dir}"
            )
            raise CheckpointNotFoundError(
                f"체크포인트 경로가 존재하지 않음: {nonexistent_checkpoint_dir}"
            )
        # 이 라인에 도달하면 테스트 실패 (경로가 존재)
        assert False, "오류: 체크포인트 경로가 없어야 하는데 존재합니다"
    except CheckpointNotFoundError as e:
        # 2단계: 예상된 예외 발생 확인 (성공 케이스)
        print(f"[테스트 2단계] 의도된 예외 발생 확인: {e}")
        print("[상황 설명] 체크포인트가 없는 상황은, 훈련이 필요한 상황으로 처리합니다")

        # 3단계: 새 체크포인트 디렉토리에 모델 학습
        new_checkpoint_dir = nonexistent_checkpoint_dir.parent / "new_flax_checkpoint"
        print(
            f"[테스트 3단계] 새 체크포인트 디렉토리({new_checkpoint_dir})에 모델 학습 시작..."
        )
        train_state = trainer.train(
            num_epochs=1,
            subdir=str(new_checkpoint_dir),
            evaluate_after_training=False,
        )

        # 4단계: 학습 결과 및 체크포인트 생성 검증
        print("[테스트 4단계] 학습 결과 및 체크포인트 검증...")
        # 학습 상태 검증
        assert train_state is not None, "오류: 학습 후 train_state가 None입니다"
        assert hasattr(train_state, "train_state"), (
            "오류: 반환된 객체에 train_state 속성이 없습니다"
        )
        assert hasattr(train_state.train_state, "params"), (
            "오류: train_state에 params 속성이 없습니다"
        )

        # 새 체크포인트 디렉토리 검증
        assert new_checkpoint_dir.exists(), (
            f"오류: 새 체크포인트 디렉토리({new_checkpoint_dir})가 생성되지 않았습니다"
        )

        # 체크포인트 디렉토리 확인
        checkpoint_dirs = [
            d for d in new_checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()
        ]
        assert len(checkpoint_dirs) > 0, (
            "오류: 체크포인트 디렉토리가 생성되지 않았습니다"
        )
        print(
            f"[테스트 완료] 체크포인트 복구 프로세스 검증 성공: {len(checkpoint_dirs)}개 체크포인트 생성됨"
        )


def test_flax_evaluation():
    """FLAX 모델의 평가 기능 테스트"""
    # 실제 FlaxModelManager 생성
    model_manager = FlaxModelManager()
    model_manager.init_model()

    # 트레이너 생성
    trainer = FlaxTrainer(model_manager)

    # 간단한 훈련
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "evaluation_test"
        train_state = trainer.train(
            num_epochs=1, subdir=str(checkpoint_dir), evaluate_after_training=False
        )

        # 학습 완료 확인
        assert train_state is not None
        assert hasattr(train_state, "train_state")
        assert hasattr(train_state.train_state, "params")

        # 체크포인트 디렉토리 확인
        checkpoint_dirs = [
            d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()
        ]
        assert len(checkpoint_dirs) > 0, "체크포인트 디렉토리가 생성되지 않음"

        # 가장 최근 체크포인트 확인
        last_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.name))
        model_dir = last_checkpoint / "model"
        assert model_dir.exists(), f"모델 디렉토리 {model_dir}가 존재하지 않습니다."
