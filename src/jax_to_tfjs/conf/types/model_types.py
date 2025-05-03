"""
모델 관련 타입 모듈

모델 파라미터, 상태, 함수 시그니처 등의 타입 정의를 제공합니다.
"""

from typing import (
    Dict,
    Any,
    TypedDict,
    Callable,
    NamedTuple,
    Protocol,
    runtime_checkable,
)
import numpy as np
from .array_types import Array


# ----- 모델 데이터 타입 -----

# PyTree 타입 (중첩된 딕셔너리, 리스트, 튜플 등)
PyTree = Any

# Params 타입 (모델 파라미터를 위한 타입)
Params = PyTree

# 옵티마이저 상태 타입
OptState = Any  # 실제로는 optax.OptState가 되어야 함

# 모델 상태 타입
ModelState = Dict[str, Any]

# PRNGKey 타입
PRNGKey = Any  # 실제로는 jax.random.PRNGKey 반환 타입


# ----- 모델 함수 타입 -----

# 모델 적용 함수 타입
ModelFn = Callable[[Params, Array], Array]

# 손실 함수 타입
LossFn = Callable[[Array, Array], Array]


# ----- 데이터 타입 -----


# 배치 데이터 타입
class BatchData(TypedDict):
    """데이터 배치 타입"""

    image: Array
    label: Array


# ----- 모델 인터페이스 -----


@runtime_checkable
class ModelProtocol(Protocol):
    """모델 인터페이스 프로토콜"""

    def apply(self, params: Params, x: Array) -> Array: ...
    def init_params(self) -> Params: ...


# ----- 학습 상태 관리 타입 -----


class TrainState(NamedTuple):
    """학습 상태를 저장하는 클래스"""

    step: int
    epoch: int
    params: Params
    opt_state: OptState
    loss: float


# ----- 로깅 데이터 타입 -----


class LogData(TypedDict, total=False):
    """로깅 데이터 타입"""

    epoch: int
    step: int
    train_loss: float
    val_loss: float  # Optional
    accuracy: float  # Optional
    learning_rate: float
    checkpoint_path: str  # Optional
    elapsed_time: float  # Optional


__all__ = [
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
]
