"""
타입 시스템 기반 모듈

JAX, NumPy, TensorFlow 간 타입 호환성을 위한 타입 정의와 헬퍼 함수를 제공합니다.
타입 검사 및 정적 분석을 지원하기 위한 공통 타입을 정의합니다.
"""

from .array_types import Array, ArrayLike, ensure_numpy
from .model_types import (
    PyTree,
    Params,
    OptState,
    ModelState,
    PRNGKey,
    ModelFn,
    LossFn,
    BatchData,
    ModelProtocol,
    TrainState,
    LogData,
)
from .utils import ensure_path, ensure_str_path, safe_cast

__all__ = [
    # 배열 관련 타입
    "Array",
    "ArrayLike",
    "ensure_numpy",
    # 모델 관련 타입
    "PyTree",
    "Params",
    "OptState",
    "ModelState",
    "PRNGKey",
    "ModelFn",
    "LossFn",
    "BatchData",
    "ModelProtocol",
    "TrainState",
    "LogData",
    # 유틸리티 함수
    "ensure_path",
    "ensure_str_path",
    "safe_cast",
]
