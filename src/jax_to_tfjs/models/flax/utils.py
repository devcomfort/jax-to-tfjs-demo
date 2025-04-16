"""
FLAX 모델 관련 유틸리티 함수

FLAX 모델과 관련된 유틸리티 함수들을 제공합니다.
"""

import jax
from flax.training import train_state
from typing import Dict, Any

from .model_manager import FlaxModelManager


def create_train_state(
    rng: jax.random.PRNGKey, learning_rate: float = 0.001
) -> train_state.TrainState:
    """
    모델 학습 상태 생성 (이전 코드와의 호환성을 위한 함수)

    Args:
        rng: 난수 생성기 키
        learning_rate: 학습률 (기본값: 0.001)

    Returns:
        초기화된 학습 상태
    """
    model_manager = FlaxModelManager(rng)
    model_manager.init_model()
    return model_manager.create_train_state(learning_rate)


def load_checkpoint(checkpoint_dir: str = None, step: int = None) -> Dict[str, Any]:
    """
    체크포인트 로드 (호환성 함수)

    기존 코드와의 호환성을 위해 ModelManager.load_checkpoint를 호출합니다.

    Args:
        checkpoint_dir: 체크포인트 디렉토리 경로 (None이면 기본 경로 사용)
        step: 로드할 체크포인트 스텝 (None이면 가장 최근 체크포인트)

    Returns:
        로드된 학습 상태
    """
    return FlaxModelManager.load_checkpoint(checkpoint_dir, step)
