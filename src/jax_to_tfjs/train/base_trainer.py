"""
모델 트레이너 추상 기본 클래스

모델 학습을 위한 인터페이스와 공통 기능을 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Optional, NamedTuple
import jax.numpy as jnp


class TrainingState(NamedTuple):
    """학습 상태를 저장하는 기본 NamedTuple"""

    step: int
    epoch: int
    loss: float


class BaseTrainer(ABC):
    """
    모델 학습 추상 기본 클래스

    모든 모델 트레이너는 이 클래스를 상속받아 구현해야 합니다.
    """

    def __init__(self, learning_rate: float = 0.001):
        """
        모델 학습 클래스 초기화

        Args:
            learning_rate: 학습률 (기본값: 0.001)
        """
        self.learning_rate = learning_rate

    @abstractmethod
    def _compute_loss(self, logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """
        손실 함수 계산

        Args:
            logits: 모델 출력
            labels: 정답 레이블

        Returns:
            계산된 손실값
        """
        pass

    @abstractmethod
    def _create_update_step(self):
        """
        학습 스텝 함수 생성

        Returns:
            학습 스텝 함수
        """
        pass

    @abstractmethod
    def train(self, num_epochs: int = 5, subdir: Optional[str] = None) -> TrainingState:
        """
        모델 학습 수행

        Args:
            num_epochs: 학습 에포크 수 (기본값: 5)
            subdir: 체크포인트 저장 하위 디렉토리 (기본값: None)

        Returns:
            최종 학습 상태
        """
        pass
