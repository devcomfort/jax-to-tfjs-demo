"""
FLAX CNN 모델 구현

FLAX로 구현된 CNN 모델 클래스를 정의합니다.
"""

import jax.numpy as jnp
import flax.linen as nn


class CNN(nn.Module):
    """
    FLAX CNN 모델 클래스

    Flax의 nn.Module을 상속하여 모델 구조를 정의합니다.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        모델 순전파

        Args:
            x: 입력 이미지 배치
            training: 학습 모드 여부

        Returns:
            모델 출력 (로짓)
        """
        # 채널 차원이 없으면 추가 (B, H, W) -> (B, H, W, 1)
        if len(x.shape) == 3:
            x = x.reshape(*x.shape, 1)

        # 첫 번째 컨볼루션 레이어
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # 두 번째 컨볼루션 레이어
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # 완전 연결 레이어
        x = x.reshape((x.shape[0], -1))  # 평탄화
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)

        return x
