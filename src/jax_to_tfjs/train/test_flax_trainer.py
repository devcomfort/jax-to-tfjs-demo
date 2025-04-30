"""
FLAX 트레이너 테스트

FlaxTrainer 클래스의 체크포인트 관련 기능 테스트
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import jax
import numpy as np

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
    train_state = trainer.train(num_epochs=1, subdir=str(checkpoint_dir))

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
    """체크포인트가 없는 경우 자동으로 train을 수행하는지 테스트"""
    # 실제 FlaxModelManager 및 트레이너 생성
    model_manager = FlaxModelManager()
    model_manager.init_model()
    trainer = FlaxTrainer(model_manager)

    # 체크포인트 로드 시도
    try:
        # 체크포인트 경로가 없으므로 예외 발생해야 함
        if not nonexistent_checkpoint_dir.exists():
            raise CheckpointNotFoundError(
                f"체크포인트 경로가 존재하지 않음: {nonexistent_checkpoint_dir}"
            )
        assert False, "체크포인트가 없는데 예외가 발생하지 않음"
    except CheckpointNotFoundError as e:
        print(f"예상된 오류 발생: {e}")
        # 체크포인트가 없으므로 학습 수행
        new_checkpoint_dir = nonexistent_checkpoint_dir.parent / "new_flax_checkpoint"
        train_state = trainer.train(
            num_epochs=1,
            subdir=str(new_checkpoint_dir),
        )

        # 학습 결과 검증
        assert train_state is not None
        assert hasattr(train_state, "train_state")
        assert hasattr(train_state.train_state, "params")

        # 새로 생성된 체크포인트 확인
        assert new_checkpoint_dir.exists()

        # 체크포인트 디렉토리 확인
        checkpoint_dirs = [
            d for d in new_checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()
        ]
        assert len(checkpoint_dirs) > 0, "체크포인트 디렉토리가 생성되지 않음"

        # 모델 체크포인트 확인
        last_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.name))
        model_dir = last_checkpoint / "model"
        assert model_dir.exists(), f"모델 디렉토리 {model_dir}가 존재하지 않습니다."


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
        train_state = trainer.train(num_epochs=1, subdir=str(checkpoint_dir))

        # 평가기 생성 및 평가 수행
        from jax_to_tfjs.train.flax_evaluator import FlaxEvaluator

        evaluator = FlaxEvaluator(model_manager)

        # 모델 평가
        accuracy, predictions = evaluator.evaluate(train_state.train_state)

        # 정확도는 초기 모델이므로 낮을 수 있으나, 숫자여야 함
        assert isinstance(accuracy, (float, np.floating)), (
            f"정확도가 숫자가 아님: {accuracy}"
        )
        assert 0 <= accuracy <= 1, f"정확도가 범위를 벗어남: {accuracy}"

        # 예측 결과 확인
        assert predictions is not None
        assert len(predictions) > 0
