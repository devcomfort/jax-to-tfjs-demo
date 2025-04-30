"""
JAX 오토인코더 모델 구현

JAX로 구현된 오토인코더 모델 클래스를 정의합니다.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple, List, Callable

from ...conf.types import Array, PRNGKey, Params
from .utils import generate_keys, init_kernel, init_bias


class AutoencoderModel:
    """
    JAX 오토인코더 모델 클래스

    오토인코더 모델의 구조를 정의하고 파라미터 초기화, 모델 적용 등의 기능을 제공합니다.
    """

    def __init__(
        self,
        rng: Optional[PRNGKey] = None,
        input_shape: Tuple[int, int, int] = (28, 28, 1),
        latent_dim: int = 32,
        encoder_dims: List[int] = [128, 64],
        decoder_dims: Optional[List[int]] = None,
    ):
        """
        오토인코더 모델 클래스 초기화

        Args:
            rng: JAX 난수 생성을 위한 키 (기본값: None, 이 경우 자동 생성)
            input_shape: 입력 이미지 형태 (기본값: (28, 28, 1))
            latent_dim: 잠재 변수의 차원 (기본값: 32)
            encoder_dims: 인코더의 은닉층 차원들 (기본값: [128, 64])
            decoder_dims: 디코더의 은닉층 차원들 (기본값: encoder_dims를 역순으로 사용)
        """
        self.rng = rng if rng is not None else jax.random.PRNGKey(0)
        self.params = None
        self.opt_state = None
        self.input_shape = input_shape
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims
        self.decoder_dims = (
            decoder_dims if decoder_dims is not None else list(reversed(encoder_dims))
        )

        # 모델 순전파 함수 생성
        self._forward_fn = self._create_forward_fn()

    def init_params(self) -> Params:
        """
        오토인코더 모델 파라미터 초기화

        Returns:
            초기화된 모델 파라미터 딕셔너리
        """
        self.params = self._init_autoencoder_params()
        return self.params

    def _init_autoencoder_params(self) -> Dict[str, Dict[str, Array]]:
        """
        오토인코더 모델 파라미터 초기화

        Returns:
            초기화된 오토인코더 모델 파라미터
        """
        # 난수 키 생성
        total_layers = len(self.encoder_dims) + 1 + len(self.decoder_dims) + 1
        keys = generate_keys(self.rng, total_layers)
        key_idx = 0

        params = {"encoder": {}, "decoder": {}}

        # 인코더 레이어 초기화
        prev_dim = self.input_dim
        for i, dim in enumerate(self.encoder_dims):
            layer_name = f"dense{i + 1}"
            params["encoder"][layer_name] = {
                "w": init_kernel(keys[key_idx], (prev_dim, dim)),
                "b": init_bias(dim),
            }
            prev_dim = dim
            key_idx += 1

        # 잠재 변수 레이어
        params["encoder"]["latent"] = {
            "w": init_kernel(keys[key_idx], (prev_dim, self.latent_dim)),
            "b": init_bias(self.latent_dim),
        }
        key_idx += 1

        # 디코더 레이어 초기화
        prev_dim = self.latent_dim
        for i, dim in enumerate(self.decoder_dims):
            layer_name = f"dense{i + 1}"
            params["decoder"][layer_name] = {
                "w": init_kernel(keys[key_idx], (prev_dim, dim)),
                "b": init_bias(dim),
            }
            prev_dim = dim
            key_idx += 1

        # 출력 레이어
        params["decoder"]["output"] = {
            "w": init_kernel(keys[key_idx], (prev_dim, self.input_dim)),
            "b": init_bias(self.input_dim),
        }

        return params

    def _create_forward_fn(self) -> Callable:
        """
        오토인코더 모델 순전파 함수 생성

        Returns:
            오토인코더 모델 순전파 함수
        """

        def forward(params: Dict[str, Any], x: Array) -> Dict[str, Array]:
            # 입력을 평탄화
            batch_size = x.shape[0]
            x_flat = x.reshape(batch_size, -1)

            # 인코더 적용
            encoder_params = params["encoder"]
            h = x_flat
            for i in range(len(self.encoder_dims)):
                w = encoder_params[f"dense{i + 1}"]["w"]
                b = encoder_params[f"dense{i + 1}"]["b"]
                h = jnp.dot(h, w) + b
                h = jax.nn.relu(h)

            # 잠재 변수 생성
            z = (
                jnp.dot(h, encoder_params["latent"]["w"])
                + encoder_params["latent"]["b"]
            )

            # 디코더 적용
            decoder_params = params["decoder"]
            h = z
            for i in range(len(self.decoder_dims)):
                w = decoder_params[f"dense{i + 1}"]["w"]
                b = decoder_params[f"dense{i + 1}"]["b"]
                h = jnp.dot(h, w) + b
                h = jax.nn.relu(h)

            # 출력층 적용
            y = (
                jnp.dot(h, decoder_params["output"]["w"])
                + decoder_params["output"]["b"]
            )
            y = jax.nn.sigmoid(y)  # 이미지 픽셀 값은 0~1 범위로 정규화

            # 원래 이미지 형태로 재구성
            output_shape = (batch_size,) + self.input_shape
            y_reshaped = y.reshape(output_shape)

            return {"latent": z, "output": y_reshaped}

        return forward

    def encode(self, params: Params, x: Array) -> Array:
        """
        인코더 부분만 적용하여 잠재 변수 생성

        Args:
            params: 모델 파라미터
            x: 입력 이미지 배치

        Returns:
            잠재 변수
        """
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        # 인코더 적용
        encoder_params = params["encoder"]
        h = x_flat
        for i in range(len(self.encoder_dims)):
            w = encoder_params[f"dense{i + 1}"]["w"]
            b = encoder_params[f"dense{i + 1}"]["b"]
            h = jnp.dot(h, w) + b
            h = jax.nn.relu(h)

        # 잠재 변수 생성
        z = jnp.dot(h, encoder_params["latent"]["w"]) + encoder_params["latent"]["b"]
        return z

    def decode(self, params: Params, z: Array) -> Array:
        """
        디코더 부분만 적용하여 이미지 재구성

        Args:
            params: 모델 파라미터
            z: 잠재 변수

        Returns:
            재구성된 이미지
        """
        batch_size = z.shape[0]

        # 디코더 적용
        decoder_params = params["decoder"]
        h = z
        for i in range(len(self.decoder_dims)):
            w = decoder_params[f"dense{i + 1}"]["w"]
            b = decoder_params[f"dense{i + 1}"]["b"]
            h = jnp.dot(h, w) + b
            h = jax.nn.relu(h)

        # 출력층 적용
        y = jnp.dot(h, decoder_params["output"]["w"]) + decoder_params["output"]["b"]
        y = jax.nn.sigmoid(y)

        # 원래 이미지 형태로 재구성
        output_shape = (batch_size,) + self.input_shape
        return y.reshape(output_shape)

    def forward(self, params: Params, x: Array) -> Dict[str, Array]:
        """
        오토인코더 모델 순전파

        Args:
            params: 모델 파라미터
            x: 입력 이미지 배치

        Returns:
            모델 출력 (latent: 잠재 변수, output: 재구성된 이미지)
        """
        return self._forward_fn(params, x)

    def apply(self, params: Dict[str, Any], x: Array) -> Dict[str, Array]:
        """
        모델 추론을 위한 함수

        Args:
            params: 모델 파라미터 (또는 {'params': ...} 형태의 딕셔너리)
            x: 입력 데이터

        Returns:
            모델 출력 (latent: 잠재 변수, output: 재구성된 이미지)
        """
        # 파라미터가 {'params': ...} 형태로 전달될 수 있음
        if "params" in params and isinstance(params["params"], dict):
            params = params["params"]

        return self.forward(params, x)

    def compute_loss(self, params: Params, x: Array) -> Array:
        """
        재구성 손실 계산 (MSE)

        Args:
            params: 모델 파라미터
            x: 입력 이미지 배치

        Returns:
            재구성 손실 값
        """
        outputs = self.forward(params, x)
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = outputs["output"].reshape(x.shape[0], -1)

        # 평균 제곱 오차 계산
        return jnp.mean((x_flat - y_flat) ** 2)

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
