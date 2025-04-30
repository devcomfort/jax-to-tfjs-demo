"""
JAX 모델 유틸리티 모듈

JAX 기반 모델 구현에 사용되는 유틸리티 함수들을 제공합니다.
레이어 연산, 파라미터 초기화, 모델 파라미터 저장 및 로드 기능 등을 포함합니다.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List, Union, Callable

import jax
import jax.numpy as jnp
from jax import lax
import orbax.checkpoint as ocp

from ...conf.types import Array, PRNGKey

logger = logging.getLogger(__name__)


def init_kernel(
    key: PRNGKey,
    shape: Tuple[int, ...],
    dtype: Any = jnp.float32,
    scale: float = 0.1,
) -> Array:
    """
    컨볼루션 또는 완전 연결 레이어의 커널 파라미터 초기화

    Args:
        key: JAX 난수 키
        shape: 커널 형태 (예: (3, 3, 64, 128) 또는 (784, 100))
        dtype: 데이터 타입 (기본값: jnp.float32)
        scale: 초기값 스케일 (기본값: 0.1)

    Returns:
        초기화된 커널 파라미터
    """
    return jax.random.normal(key, shape, dtype) * scale


def init_bias(shape: Union[int, Tuple[int, ...]], dtype: Any = jnp.float32) -> Array:
    """
    편향 파라미터 초기화 (0으로 초기화)

    Args:
        shape: 편향 형태 (예: 128 또는 (128,))
        dtype: 데이터 타입 (기본값: jnp.float32)

    Returns:
        초기화된 편향 파라미터
    """
    if isinstance(shape, int):
        shape = (shape,)
    return jnp.zeros(shape, dtype)


def generate_keys(key: PRNGKey, num_keys: int) -> List[PRNGKey]:
    """
    여러 개의 난수 키 생성

    Args:
        key: 시드로 사용할 JAX 난수 키
        num_keys: 생성할 키 개수

    Returns:
        생성된 난수 키 리스트
    """
    return list(jax.random.split(key, num_keys))


def init_cnn_params(
    rng_key: PRNGKey,
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    num_filters: Tuple[int, ...] = (32, 64),
    dense_features: int = 128,
) -> Dict[str, Dict[str, Array]]:
    """
    CNN 모델 파라미터 초기화

    Args:
        rng_key: JAX 난수 키
        input_shape: 입력 이미지 형태 (기본값: (28, 28, 1))
        num_classes: 출력 클래스 수 (기본값: 10)
        num_filters: 각 레이어의 필터 수 (기본값: (32, 64))
        dense_features: 완전 연결 레이어의 노드 수 (기본값: 128)

    Returns:
        초기화된 CNN 모델 파라미터
    """
    # 난수 키 생성
    keys = generate_keys(rng_key, len(num_filters) + 2)
    key_idx = 0

    params = {}
    input_channels = input_shape[-1]

    # 컨볼루션 레이어 초기화
    for i, filters in enumerate(num_filters):
        layer_name = f"conv{i + 1}"
        params[layer_name] = {
            "w": init_kernel(keys[key_idx], (3, 3, input_channels, filters)),
            "b": init_bias(filters),
        }
        input_channels = filters
        key_idx += 1

    # 평탄화 후 입력 크기 계산
    # 각 풀링 레이어마다 크기가 반으로 줄어듬
    h, w = input_shape[:2]
    for _ in num_filters:
        h = h // 2
        w = w // 2
    flattened_size = h * w * input_channels

    # 첫 번째 완전 연결 레이어
    params["dense1"] = {
        "w": init_kernel(keys[key_idx], (flattened_size, dense_features)),
        "b": init_bias(dense_features),
    }
    key_idx += 1

    # 출력 레이어
    params["dense2"] = {
        "w": init_kernel(keys[key_idx], (dense_features, num_classes)),
        "b": init_bias(num_classes),
    }

    return params


def cnn_forward(params: Dict[str, Dict[str, Array]], x: Array) -> Array:
    """
    CNN 모델 순전파

    Args:
        params: 모델 파라미터
        x: 입력 배열

    Returns:
        모델 출력
    """
    # 첫 번째 컨볼루션 레이어
    x = lax.conv(
        x,
        params["conv1"]["w"],
        window_strides=(1, 1),
        padding="SAME",
    )
    x = x + params["conv1"]["b"]
    x = jax.nn.relu(x)
    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID"
    )

    # 두 번째 컨볼루션 레이어 (여기서는 간략화를 위해 생략)

    # 평탄화
    x = x.reshape((x.shape[0], -1))

    # 첫 번째 밀집 레이어
    x = jnp.dot(x, params["dense1"]["w"]) + params["dense1"]["b"]
    x = jax.nn.relu(x)

    # 출력 레이어
    x = jnp.dot(x, params["dense2"]["w"]) + params["dense2"]["b"]
    return x


def load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    체크포인트에서 모델 파라미터 로드

    Args:
        checkpoint_path: 체크포인트 경로

    Returns:
        로드된 모델 파라미터 또는 로드 실패 시 None
    """
    if not os.path.exists(checkpoint_path):
        print(f"체크포인트 경로를 찾을 수 없습니다: {checkpoint_path}")
        return None

    try:
        checkpointer = ocp.PyTreeCheckpointer()
        params = checkpointer.restore(checkpoint_path)
        return params
    except Exception as e:
        print(f"체크포인트 로드 중 오류 발생: {str(e)}")
        return None


def save_checkpoint(params: Dict[str, Any], checkpoint_path: str) -> bool:
    """
    모델 파라미터를 체크포인트로 저장

    Args:
        params: 저장할 모델 파라미터
        checkpoint_path: 체크포인트 저장 경로

    Returns:
        저장 성공 여부
    """
    try:
        # 체크포인트 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(checkpoint_path, params)
        return True
    except Exception as e:
        print(f"체크포인트 저장 중 오류 발생: {str(e)}")
        return False


def apply_cnn_block(
    x: Array,
    params: Dict[str, Any],
    prefix: str,
    strides: Tuple[int, int] = (1, 1),
    pool: bool = True,
) -> Array:
    """
    CNN 블록 적용 (컨볼루션 + ReLU + 풀링)

    Args:
        x: 입력 배열
        params: 모델 파라미터
        prefix: 파라미터 접두사 (예: "conv1")
        strides: 컨볼루션 스트라이드 (기본값: (1, 1))
        pool: 풀링 레이어 포함 여부 (기본값: True)

    Returns:
        CNN 블록 적용 결과
    """
    x = conv2d(x, params[prefix]["w"], params[prefix]["b"], strides)
    x = relu(x)
    if pool:
        x = max_pool(x)
    return x


def create_cnn_forward(
    num_conv_layers: int = 2, num_dense_layers: int = 2
) -> Callable[[Dict[str, Any], Array], Array]:
    """
    CNN 모델 순전파 함수 생성

    Args:
        num_conv_layers: 컨볼루션 레이어 수 (기본값: 2)
        num_dense_layers: 완전 연결 레이어 수 (기본값: 2)

    Returns:
        CNN 모델 순전파 함수
    """

    def forward(params: Dict[str, Any], x: Array) -> Array:
        # 입력 이미지 reshape (B, H, W) -> (B, H, W, C)
        if len(x.shape) == 3:
            x = x.reshape(*x.shape, 1)

        # 컨볼루션 레이어 적용
        for i in range(1, num_conv_layers + 1):
            x = apply_cnn_block(x, params, f"conv{i}")

        # 평탄화
        x = flatten(x)

        # 완전 연결 레이어 적용
        for i in range(1, num_dense_layers):
            x = dense(x, params[f"dense{i}"]["w"], params[f"dense{i}"]["b"])
            x = relu(x)

        # 출력 레이어
        x = dense(
            x,
            params[f"dense{num_dense_layers}"]["w"],
            params[f"dense{num_dense_layers}"]["b"],
        )

        return x

    return forward


def conv2d(
    x: Array,
    w: Array,
    b: Array,
    strides: Tuple[int, int] = (1, 1),
    padding: str = "SAME",
) -> Array:
    """
    2D 컨볼루션 레이어 연산

    Args:
        x: 입력 배열 (NHWC 형식)
        w: 컨볼루션 커널 (HWIO 형식)
        b: 편향 파라미터
        strides: 스트라이드 (기본값: (1, 1))
        padding: 패딩 방식 (기본값: "SAME")

    Returns:
        컨볼루션 연산 결과
    """
    y = jax.lax.conv_general_dilated(
        x,
        w,
        window_strides=strides,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + b[None, None, None, :]


def dense(x: Array, w: Array, b: Array) -> Array:
    """
    완전 연결 레이어 연산

    Args:
        x: 입력 배열
        w: 가중치 행렬
        b: 편향 벡터

    Returns:
        완전 연결 레이어 연산 결과
    """
    return jnp.dot(x, w) + b


def max_pool(
    x: Array,
    window_shape: Tuple[int, int] = (2, 2),
    strides: Tuple[int, int] = (2, 2),
    padding: str = "SAME",
) -> Array:
    """
    최대 풀링 연산

    Args:
        x: 입력 배열 (NHWC 형식)
        window_shape: 풀링 윈도우 크기 (기본값: (2, 2))
        strides: 스트라이드 (기본값: (2, 2))
        padding: 패딩 방식 (기본값: "SAME")

    Returns:
        최대 풀링 연산 결과
    """
    return jax.lax.reduce_window(
        x,
        -jnp.inf,
        jax.lax.max,
        (1, *window_shape, 1),
        (1, *strides, 1),
        padding,
    )


def relu(x: Array) -> Array:
    """
    ReLU 활성화 함수

    Args:
        x: 입력 배열

    Returns:
        ReLU 적용 결과
    """
    return jax.nn.relu(x)


def flatten(x: Array) -> Array:
    """
    입력 배열 평탄화 (배치 차원 유지)

    Args:
        x: 입력 배열

    Returns:
        평탄화된 배열
    """
    return x.reshape((x.shape[0], -1))
