"""
JAX CNN 모델 구현

JAX로 구현된 CNN 모델 클래스를 정의합니다.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any


class CNNModel:
    """
    JAX CNN 모델 클래스

    CNN 모델의 구조를 정의하고 파라미터 초기화, 모델 적용 등의 기능을 제공합니다.
    """

    def __init__(self, rng: jax.random.PRNGKey = None):
        """
        CNN 모델 클래스 초기화

        Args:
            rng: JAX 난수 생성을 위한 키 (기본값: None, 이 경우 자동 생성)
        """
        self.rng = rng if rng is not None else jax.random.PRNGKey(0)
        self.params = None

    def init_params(self) -> Dict[str, Any]:
        """
        CNN 모델 파라미터 초기화

        Returns:
            초기화된 모델 파라미터 딕셔너리
        """
        keys = jax.random.split(self.rng, 5)

        # 첫 번째 컨볼루션 레이어
        conv1 = {
            "w": jax.random.normal(keys[0], (3, 3, 1, 32)) * 0.1,
            "b": jnp.zeros(32),
        }

        # 두 번째 컨볼루션 레이어
        conv2 = {
            "w": jax.random.normal(keys[1], (3, 3, 32, 64)) * 0.1,
            "b": jnp.zeros(64),
        }

        # 첫 번째 완전 연결 레이어
        dense1 = {
            "w": jax.random.normal(keys[2], (7 * 7 * 64, 128)) * 0.1,
            "b": jnp.zeros(128),
        }

        # 출력 레이어
        dense2 = {"w": jax.random.normal(keys[3], (128, 10)) * 0.1, "b": jnp.zeros(10)}

        self.params = {
            "conv1": conv1,
            "conv2": conv2,
            "dense1": dense1,
            "dense2": dense2,
        }

        return self.params

    def forward(self, params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        """
        CNN 모델 순전파

        Args:
            params: 모델 파라미터
            x: 입력 이미지 배치

        Returns:
            모델 출력 (로짓)
        """
        # 입력 이미지 reshape (B, H, W) -> (B, H, W, C)
        if len(x.shape) == 3:
            x = x.reshape(*x.shape, 1)

        # 첫 번째 컨볼루션 레이어
        x = jax.lax.conv_general_dilated(
            x,
            params["conv1"]["w"],
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        x = x + params["conv1"]["b"][None, None, None, :]
        x = jax.nn.relu(x)
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "SAME"
        )

        # 두 번째 컨볼루션 레이어
        x = jax.lax.conv_general_dilated(
            x,
            params["conv2"]["w"],
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        x = x + params["conv2"]["b"][None, None, None, :]
        x = jax.nn.relu(x)
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "SAME"
        )

        # 완전 연결 레이어
        x = x.reshape((x.shape[0], -1))
        x = jnp.dot(x, params["dense1"]["w"]) + params["dense1"]["b"]
        x = jax.nn.relu(x)
        x = jnp.dot(x, params["dense2"]["w"]) + params["dense2"]["b"]

        return x

    def apply(self, params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        """
        모델 추론을 위한 함수

        Args:
            params: 모델 파라미터
            x: 입력 데이터

        Returns:
            모델 출력 (로짓)
        """
        # 파라미터가 {'params': ...} 형태로 전달될 수 있음
        if "params" in params and isinstance(params["params"], dict):
            params = params["params"]

        return self.forward(params, x)
