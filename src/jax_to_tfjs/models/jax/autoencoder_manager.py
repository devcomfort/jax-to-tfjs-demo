"""
JAX 오토인코더 모델 매니저 구현

JAX 오토인코더 모델의 초기화와 체크포인트 관리를 담당하는 클래스를 정의합니다.
"""

import os
import jax
import pickle
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, cast

from ...conf.types import PRNGKey, Params
from ...conf.paths import get_jax_checkpoint_path
from .autoencoder_model import AutoencoderModel


# 로깅 설정
logger = logging.getLogger(__name__)


class JAXAutoencoderManager:
    """
    JAX 오토인코더 모델 관리 클래스

    모델 초기화, 체크포인트 저장 및 로드 등의 기능을 제공합니다.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (28, 28, 1),
        latent_dim: int = 32,
        encoder_dims: List[int] = [128, 64],
        rng: Optional[PRNGKey] = None,
    ):
        """
        JAX 오토인코더 모델 매니저 초기화

        Args:
            input_shape: 입력 이미지 형태 (기본값: (28, 28, 1))
            latent_dim: 잠재 변수의 차원 (기본값: 32)
            encoder_dims: 인코더의 은닉층 차원들 (기본값: [128, 64])
            rng: JAX 난수 생성을 위한 키 (기본값: None, 이 경우 자동 생성)
        """
        self.rng = rng if rng is not None else jax.random.PRNGKey(0)
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims
        self.model: Optional[AutoencoderModel] = None
        self.params: Optional[Params] = None
        self.metadata = {
            "model_type": "jax_autoencoder",
            "input_shape": input_shape,
            "latent_dim": latent_dim,
            "encoder_dims": encoder_dims,
            "version": "1.0.0",
        }

    def initialize_model(self) -> AutoencoderModel:
        """
        오토인코더 모델 초기화

        Returns:
            초기화된 오토인코더 모델
        """
        if self.model is None:
            self.model = AutoencoderModel(
                rng=self.rng,
                input_shape=self.input_shape,
                latent_dim=self.latent_dim,
                encoder_dims=self.encoder_dims,
            )
        return self.model

    def initialize_params(self) -> Params:
        """
        모델 파라미터 초기화

        Returns:
            초기화된 모델 파라미터
        """
        if self.model is None:
            self.initialize_model()

        assert self.model is not None, "모델이 초기화되지 않았습니다."

        if self.params is None:
            self.params = self.model.init_params()

        return self.params

    def save_checkpoint(
        self,
        params: Optional[Params] = None,
        checkpoint_dir: Optional[str] = None,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        모델 체크포인트 저장

        Args:
            params: 저장할 모델 파라미터 (기본값: None, 이 경우 현재 파라미터 사용)
            checkpoint_dir: 체크포인트 저장 디렉토리 (기본값: None, 이 경우 기본 경로 사용)
            step: 현재 학습 단계 (기본값: 0)
            metadata: 추가 메타데이터 (기본값: None)

        Returns:
            저장된 체크포인트 파일 경로
        """
        # 파라미터 확인
        if params is None:
            if self.params is None:
                self.initialize_params()
            params = self.params

        assert params is not None, "파라미터가 초기화되지 않았습니다."

        # 체크포인트 디렉토리 확인
        if checkpoint_dir is None:
            checkpoint_dir = str(get_jax_checkpoint_path())

        # 디렉토리가 없으면 생성
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 메타데이터 병합
        save_metadata = self.metadata.copy()
        if metadata:
            save_metadata.update(metadata)
        save_metadata["step"] = step

        # 체크포인트 파일 경로 생성
        checkpoint_filename = f"autoencoder_step_{step}.pkl"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        # 모델 파라미터 및 메타데이터 저장
        with open(checkpoint_path, "wb") as f:
            pickle.dump(
                {
                    "params": params,
                    "metadata": save_metadata,
                },
                f,
            )

        logger.info(f"체크포인트가 저장되었습니다: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(
        self, checkpoint_path: str, initialize_model: bool = True
    ) -> Tuple[Params, Dict[str, Any]]:
        """
        모델 체크포인트 로드

        Args:
            checkpoint_path: 체크포인트 파일 경로
            initialize_model: 모델 초기화 여부 (기본값: True)

        Returns:
            (모델 파라미터, 메타데이터) 튜플
        """
        # 체크포인트 파일 존재 확인
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"체크포인트 파일이 존재하지 않습니다: {checkpoint_path}"
            )

        # 체크포인트 로드
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        # 파라미터 및 메타데이터 추출
        self.params = checkpoint["params"]
        metadata = checkpoint.get("metadata", {})

        # 메타데이터에서 모델 구성 정보 가져오기
        if "input_shape" in metadata:
            self.input_shape = metadata["input_shape"]
        if "latent_dim" in metadata:
            self.latent_dim = metadata["latent_dim"]
        if "encoder_dims" in metadata:
            self.encoder_dims = metadata["encoder_dims"]

        # 필요한 경우 모델 초기화
        if initialize_model:
            self.initialize_model()

        logger.info(f"체크포인트가 로드되었습니다: {checkpoint_path}")
        return self.params, metadata

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        이미지를 잠재 변수로 인코딩

        Args:
            x: 입력 이미지

        Returns:
            인코딩된 잠재 변수
        """
        if self.model is None:
            self.initialize_model()
        if self.params is None:
            self.initialize_params()

        assert self.model is not None, "모델이 초기화되지 않았습니다."
        assert self.params is not None, "파라미터가 초기화되지 않았습니다."

        return self.model.encode(self.params, x)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        잠재 변수에서 이미지 디코딩

        Args:
            z: 잠재 변수

        Returns:
            재구성된 이미지
        """
        if self.model is None:
            self.initialize_model()
        if self.params is None:
            self.initialize_params()

        assert self.model is not None, "모델이 초기화되지 않았습니다."
        assert self.params is not None, "파라미터가 초기화되지 않았습니다."

        return self.model.decode(self.params, z)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """
        이미지를 인코딩 후 디코딩하여 재구성

        Args:
            x: 입력 이미지

        Returns:
            재구성된 이미지
        """
        if self.model is None:
            self.initialize_model()
        if self.params is None:
            self.initialize_params()

        assert self.model is not None, "모델이 초기화되지 않았습니다."
        assert self.params is not None, "파라미터가 초기화되지 않았습니다."

        outputs = self.model.forward(self.params, x)
        return outputs["output"]
