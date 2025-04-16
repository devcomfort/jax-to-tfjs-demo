"""
모델 평가자 추상 기본 클래스

모델 평가를 위한 인터페이스와 공통 기능을 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import jax.numpy as jnp


class BaseEvaluator(ABC):
    """
    모델 평가 추상 기본 클래스

    모든 모델 평가자는 이 클래스를 상속받아 구현해야 합니다.
    """

    @abstractmethod
    def evaluate(self, params: Any) -> Tuple[float, jnp.ndarray]:
        """
        테스트 세트에 대한 모델 평가

        Args:
            params: 모델 파라미터 또는 상태

        Returns:
            accuracy, predictions: 정확도와 예측값
        """
        pass

    def calculate_metrics(
        self, predictions: jnp.ndarray, labels: jnp.ndarray
    ) -> Dict[str, float]:
        """
        예측에 대한 다양한 메트릭 계산

        Args:
            predictions: 모델 예측값
            labels: 정답 레이블

        Returns:
            계산된 메트릭 사전
        """
        # 기본 구현은 정확도만 계산
        correct = jnp.sum(predictions == labels)
        accuracy = float(correct) / len(labels)

        return {"accuracy": accuracy, "correct": int(correct), "total": len(labels)}
