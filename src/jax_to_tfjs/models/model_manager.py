"""
모델 관리자 기본 클래스 모듈

모델 관리에 필요한 공통 인터페이스를 정의합니다.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BaseModelManager(ABC):
    """
    모델 관리자 기본 클래스

    모든 모델 관리자가 구현해야 하는 공통 인터페이스를 정의합니다.
    """

    def __init__(self, model_dir: str, checkpoint_dir: str):
        """
        BaseModelManager 초기화

        Args:
            model_dir: 모델 디렉토리 경로
            checkpoint_dir: 체크포인트 디렉토리 경로
        """
        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir

        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    @abstractmethod
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
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        path: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Any]:
        """
        체크포인트에서 모델 파라미터 로드

        Args:
            path: 체크포인트 경로 (기본값: None, 기본 경로 사용)
            step: 특정 스텝 (기본값: None, 가장 최근 스텝 사용)

        Returns:
            로드된 모델 파라미터와 옵티마이저 상태
        """
        pass

    @abstractmethod
    def restore(
        self,
        checkpoint_dir: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        모델 상태 복원

        Args:
            checkpoint_dir: 체크포인트 디렉토리 (기본값: None, 기본 디렉토리 사용)
            step: 특정 스텝 (기본값: None, 가장 최근 스텝 사용)

        Returns:
            복원된 모델 파라미터
        """
        pass

    @abstractmethod
    def predict(self, inputs: Union[Any, np.ndarray]) -> Any:
        """
        모델 예측 수행

        Args:
            inputs: 입력 데이터

        Returns:
            예측 결과
        """
        pass

    def export_to_tfjs(self, output_dir: str) -> None:
        """
        모델을 TensorFlow.js 형식으로 내보내기

        Args:
            output_dir: 출력 디렉토리 경로
        """
        raise NotImplementedError(
            "이 모델 관리자는 TensorFlow.js 내보내기를 지원하지 않습니다."
        )
