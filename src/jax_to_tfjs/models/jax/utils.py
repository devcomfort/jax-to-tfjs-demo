"""
JAX 모델 관련 유틸리티 함수

JAX 모델과 관련된 유틸리티 함수들을 제공합니다.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any

from .cnn_model import CNNModel
from .model_manager import JAXModelManager


def init_cnn_params(rng: jax.random.PRNGKey) -> Dict[str, Any]:
    """
    CNN 모델 파라미터 초기화 (호환성 함수)

    기존 코드와의 호환성을 위해 CNNModel.init_params를 호출합니다.

    Args:
        rng: JAX 난수 생성을 위한 키

    Returns:
        초기화된 모델 파라미터 딕셔너리
    """
    model = CNNModel(rng)
    return model.init_params()


def cnn_forward(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    """
    CNN 순전파 (호환성 함수)

    기존 코드와의 호환성을 위해 CNNModel.forward를 호출합니다.

    Args:
        params: 모델 파라미터
        x: 입력 이미지 배치

    Returns:
        모델 출력 (로짓)
    """
    model = CNNModel()
    return model.forward(params, x)


def load_checkpoint(checkpoint_dir: str = None, step: int = None) -> Dict[str, Any]:
    """
    체크포인트 로드 (호환성 함수)

    기존 코드와의 호환성을 위해 ModelManager.load_checkpoint를 호출합니다.

    Args:
        checkpoint_dir: 체크포인트 디렉토리 경로 (None이면 기본 경로 사용)
        step: 로드할 체크포인트 스텝 (None이면 가장 최근 체크포인트)

    Returns:
        로드된 파라미터와 옵티마이저 상태
    """
    return JAXModelManager.load_checkpoint(checkpoint_dir, step)
