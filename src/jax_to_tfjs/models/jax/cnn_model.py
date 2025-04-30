"""
JAX CNN 모델 구현

JAX로 구현된 CNN 모델 클래스를 정의합니다.
"""

import jax
from typing import Dict, Any, Optional, Tuple

from ...conf.types import Array, PRNGKey, Params
from .utils import (
    init_cnn_params,
    create_cnn_forward,
)


class CNNModel:
    """
    JAX CNN 모델 클래스

    CNN 모델의 구조를 정의하고 파라미터 초기화, 모델 적용 등의 기능을 제공합니다.
    """

    def __init__(
        self,
        rng: Optional[PRNGKey] = None,
        input_shape: Tuple[int, int, int] = (28, 28, 1),
        num_classes: int = 10,
        num_filters: Tuple[int, ...] = (32, 64),
        dense_features: int = 128,
    ):
        """
        CNN 모델 클래스 초기화

        Args:
            rng: JAX 난수 생성을 위한 키 (기본값: None, 이 경우 자동 생성)
            input_shape: 입력 이미지 형태 (기본값: (28, 28, 1))
            num_classes: 출력 클래스 수 (기본값: 10)
            num_filters: 각 레이어의 필터 수 (기본값: (32, 64))
            dense_features: 완전 연결 레이어의 노드 수 (기본값: 128)
        """
        self.rng = rng if rng is not None else jax.random.PRNGKey(0)
        self.params = None
        self.opt_state = None
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.dense_features = dense_features

        # 모델 순전파 함수 생성
        self._forward_fn = create_cnn_forward(
            num_conv_layers=len(num_filters),
            num_dense_layers=2,  # 고정: 1개의 히든 레이어 + 1개의 출력 레이어
        )

    def init_params(self) -> Params:
        """
        CNN 모델 파라미터 초기화

        Returns:
            초기화된 모델 파라미터 딕셔너리
        """
        self.params = init_cnn_params(
            rng_key=self.rng,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            num_filters=self.num_filters,
            dense_features=self.dense_features,
        )
        return self.params

    def forward(self, params: Params, x: Array) -> Array:
        """
        CNN 모델 순전파

        Args:
            params: 모델 파라미터
            x: 입력 이미지 배치

        Returns:
            모델 출력 (로짓)
        """
        return self._forward_fn(params, x)

    def apply(self, params: Dict[str, Any], x: Array) -> Array:
        """
        모델 추론을 위한 함수

        Args:
            params: 모델 파라미터 (또는 {'params': ...} 형태의 딕셔너리)
            x: 입력 데이터

        Returns:
            모델 출력 (로짓)
        """
        # 파라미터가 {'params': ...} 형태로 전달될 수 있음
        if "params" in params and isinstance(params["params"], dict):
            params = params["params"]

        return self.forward(params, x)

    def init_optimizer(self, learning_rate: float = 0.001) -> None:
        """
        옵티마이저 초기화

        Args:
            learning_rate: 학습률
        """
        if self.params is None:
            self.init_params()

        # 간단한 옵티마이저 상태 초기화
        self.opt_state = {"learning_rate": learning_rate, "iteration": 0}
