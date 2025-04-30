"""
JAX 모델 관리자 모듈

체크포인트 저장, 로드 및 모델 관리 기능을 제공합니다.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, Tuple, cast

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from ...conf.types import Array, Params, OptState
from .cnn_model import CNNModel
from ..model_manager import BaseModelManager

# 로거 설정
logger = logging.getLogger(__name__)


class JAXModelManager(BaseModelManager):
    """
    JAX 모델 관리자 클래스

    JAX 모델의 체크포인트 저장, 로드 및 기타 관리 기능을 제공합니다.
    """

    def __init__(
        self,
        model_dir: str,
        checkpoint_dir: str,
        model: Optional[CNNModel] = None,
    ):
        """
        JAXModelManager 초기화

        Args:
            model_dir: 모델 디렉토리 경로
            checkpoint_dir: 체크포인트 디렉토리 경로
            model: CNNModel 인스턴스 (기본값: None)
        """
        super().__init__(model_dir, checkpoint_dir)

        self.model = model if model is not None else CNNModel()
        self._opt_state: OptState = None  # 옵티마이저 상태 저장을 위한 내부 속성

        if self.model.params is None:
            self.model.init_params()

        # 체크포인트 관리자 초기화
        self.checkpointer = ocp.PyTreeCheckpointer()

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.checkpoint_manager = ocp.CheckpointManager(
            directory=checkpoint_dir,
            checkpointers={"model": self.checkpointer},
            options=ocp.CheckpointManagerOptions(create=True),
        )

    @property
    def opt_state(self) -> OptState:
        """옵티마이저 상태 반환"""
        return self._opt_state

    @opt_state.setter
    def opt_state(self, value: OptState) -> None:
        """옵티마이저 상태 설정"""
        self._opt_state = value

    def save_checkpoint(
        self,
        step: int,
        path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        체크포인트 저장

        Args:
            step: 체크포인트 저장할 스텝
            path: 저장 경로 (기본값: None, 기본 경로 사용)
            metadata: 저장할 메타데이터 (기본값: None)
        """
        if path is None:
            path = self.checkpoint_dir

        logger.info(f"체크포인트 저장 중: {path}, 스텝: {step}")

        # 저장할 파라미터 사전 생성
        save_dict = {"params": self.model.params}

        # 옵티마이저 상태가 있으면 추가
        if self._opt_state is not None:
            save_dict["opt_state"] = self._opt_state

        try:
            # 체크포인트 디렉토리가 없으면 생성
            os.makedirs(os.path.join(path, str(step)), exist_ok=True)

            # Orbax API 호출
            self.checkpoint_manager.save(step, {"model": save_dict})
            logger.info(f"체크포인트 저장 완료: {path}/step_{step}")
        except Exception as e:
            logger.error(f"체크포인트 저장 중 오류 발생: {e}")

    def load_checkpoint(
        self,
        path: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Tuple[Params, OptState]:
        """
        체크포인트에서 모델 파라미터 로드

        Args:
            path: 체크포인트 경로 (기본값: None, 기본 경로 사용)
            step: 특정 스텝 (기본값: None, 가장 최근 스텝 사용)

        Returns:
            로드된 모델 파라미터와 옵티마이저 상태
        """
        if path is None:
            path = self.checkpoint_dir

        logger.info(f"체크포인트 로드 중: {path}")

        try:
            # 사용 가능한 스텝 확인
            available_steps = self.checkpoint_manager.all_steps()
            if not available_steps:
                logger.warning(f"디렉토리에 체크포인트가 없습니다: {path}")
                return {}, None

            # 스텝이 지정되지 않은 경우 가장 최근 스텝 사용
            if step is None:
                step = self.checkpoint_manager.latest_step()
                logger.info(f"최신 체크포인트 사용 중, 스텝: {step}")
            elif step not in available_steps:
                logger.warning(
                    f"스텝 {step}의 체크포인트를 찾을 수 없습니다. 최신 체크포인트 사용."
                )
                step = self.checkpoint_manager.latest_step()

            # 체크포인트 로드
            restored_state = self.checkpoint_manager.restore(step)

            if "model" in restored_state:
                restored_model = restored_state["model"]
                # 모델 파라미터 설정
                if "params" in restored_model:
                    self.model.params = restored_model["params"]
                else:
                    logger.warning(
                        "로드된 체크포인트에서 모델 파라미터를 찾을 수 없습니다."
                    )

                # 옵티마이저 상태 로드
                self._opt_state = restored_model.get("opt_state")

                # 반환 타입 명시적 캐스팅
                model_params = (
                    self.model.params if self.model.params is not None else {}
                )
                return cast(Params, model_params), self._opt_state
            else:
                logger.warning("로드된 체크포인트에서 모델 키를 찾을 수 없습니다.")
                return {}, None
        except Exception as e:
            logger.error(f"체크포인트 로드 중 오류 발생: {e}")
            return {}, None

    def restore(
        self,
        checkpoint_dir: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Params:
        """
        모델 상태 복원

        Args:
            checkpoint_dir: 체크포인트 디렉토리 (기본값: None, 기본 디렉토리 사용)
            step: 특정 스텝 (기본값: None, 가장 최근 스텝 사용)

        Returns:
            복원된 모델 파라미터
        """
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir

        logger.info(f"모델 상태 복원 중: {checkpoint_dir}")

        params, opt_state = self.load_checkpoint(checkpoint_dir, step)
        self.model.params = params  # 모델 파라미터 설정
        self._opt_state = opt_state  # 옵티마이저 상태 설정

        return params

    def predict(self, inputs: Union[Array, np.ndarray]) -> Array:
        """
        모델 예측 수행

        Args:
            inputs: 입력 데이터

        Returns:
            예측 결과
        """
        # 입력이 NumPy 배열인 경우 JAX 배열로 변환
        if isinstance(inputs, np.ndarray):
            inputs = jnp.array(inputs)

        if self.model.params is None:
            raise ValueError(
                "모델 파라미터가 초기화되지 않았습니다. 먼저 모델을 학습하거나 체크포인트를 로드하세요."
            )

        return self.model.apply(self.model.params, inputs)

    def export_to_tfjs(self, output_dir: str) -> None:
        """
        모델을 TensorFlow.js 형식으로 내보내기

        Args:
            output_dir: 출력 디렉토리 경로
        """
        if self.model.params is None:
            raise ValueError(
                "모델 파라미터가 초기화되지 않았습니다. 먼저 모델을 학습하거나 체크포인트를 로드하세요."
            )

        logger.info(f"TensorFlow.js 형식으로 내보내는 중: {output_dir}")

        # TODO: JAX 모델을 TensorFlow.js 형식으로 변환하는 로직 구현
        # 별도의 convert_to_tfjs.py 모듈에서 처리
