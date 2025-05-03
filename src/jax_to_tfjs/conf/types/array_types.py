"""
배열 및 텐서 타입 관련 모듈

JAX, NumPy, TensorFlow 간 타입 호환성을 위한 타입 정의와 변환 함수를 제공합니다.
"""

from typing import Any, Union, Protocol, runtime_checkable, Tuple, cast
import numpy as np
import logging


@runtime_checkable
class ArrayLike(Protocol):
    """배열과 유사한 객체를 위한 프로토콜"""

    def __array__(self) -> np.ndarray: ...

    @property
    def shape(self) -> Tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...


# JAX 배열 타입 (실제 jax.Array 타입을 직접 가져오지 않고 타입 힌트만 사용)
# type: ignore 주석으로 타입 체커가 이 정의를 허용하도록 함
Array = Any  # type: ignore  # 실제로는 Union[jnp.ndarray, jax.Array]가 되어야 함


def ensure_numpy(array_like: Any) -> np.ndarray:
    """어떤 배열도 NumPy 배열로 변환

    JAX, TensorFlow, PyTorch 배열을 NumPy 배열로 안전하게 변환합니다.
    이미 NumPy 배열인 경우 그대로 반환합니다.

    Args:
        array_like: 변환할 배열 또는 배열과 유사한 객체

    Returns:
        NumPy 배열

    Raises:
        TypeError: 변환할 수 없는 객체가 전달된 경우
    """
    logger = logging.getLogger(__name__)

    # 입력 객체 정보 로깅
    logger.debug(f"Converting to numpy: {type(array_like).__name__}")

    # 이미 NumPy 배열인 경우
    if isinstance(array_like, np.ndarray):
        logger.debug("Already a numpy array, returning as is")
        return array_like

    # JAX 배열인 경우 (이름으로 확인)
    if hasattr(array_like, "__module__") and type(array_like).__module__.startswith(
        "jax."
    ):
        logger.debug("Detected JAX array, converting to numpy")
        # JAX 배열을 NumPy로 변환 (JAX 타입을 직접 가져오지 않으므로 isinstance 검사 불가)
        return np.array(array_like)

    # TensorFlow 텐서인 경우 (이름으로 확인)
    if hasattr(array_like, "__module__") and type(array_like).__module__.startswith(
        "tensorflow."
    ):
        logger.debug("Detected TensorFlow tensor, converting to numpy")
        # TensorFlow 텐서를 NumPy로 변환
        return array_like.numpy()  # type: ignore

    # PyTorch 텐서인 경우 (이름으로 확인)
    if hasattr(array_like, "__module__") and type(array_like).__module__.startswith(
        "torch."
    ):
        logger.debug("Detected PyTorch tensor, converting to numpy")
        # PyTorch 텐서를 NumPy로 변환
        return array_like.detach().cpu().numpy()  # type: ignore

    # ArrayLike 프로토콜 구현 확인
    if isinstance(array_like, ArrayLike):
        logger.debug("Object implements ArrayLike protocol, using __array__ method")
        return np.array(array_like)

    # 그 외 배열과 유사한 객체 (np.array로 변환 시도)
    try:
        logger.debug("Attempting generic numpy array conversion")
        array = np.array(array_like)
        logger.debug(f"Conversion successful: shape={array.shape}, dtype={array.dtype}")
        return array
    except Exception as e:
        # 상세 오류 정보 수집
        error_msg = f"Cannot convert {type(array_like)} to numpy array"

        # 객체 정보 수집 시도
        obj_attrs = []
        if hasattr(array_like, "__dict__"):
            obj_attrs = [
                f"{attr}={getattr(array_like, attr)}"
                for attr in dir(array_like)
                if not attr.startswith("__") and not callable(getattr(array_like, attr))
            ]

        attrs_info = ", ".join(obj_attrs) if obj_attrs else "no attributes found"
        logger.error(f"{error_msg}. Object attributes: {attrs_info}. Error: {e}")

        raise TypeError(f"{error_msg}: {e}")


__all__ = ["ArrayLike", "Array", "ensure_numpy"]
