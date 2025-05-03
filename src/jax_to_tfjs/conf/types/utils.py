"""
타입 변환 유틸리티 모듈

타입 변환 헬퍼 함수와 경로 관련 유틸리티 함수를 제공합니다.
"""

from typing import Any, Union, TypeVar, Type, cast
from pathlib import Path


def ensure_path(path: Union[str, Path]) -> Path:
    """문자열 또는 Path 객체를 Path 객체로 변환

    Args:
        path: 경로 문자열 또는 Path 객체

    Returns:
        Path 객체
    """
    if isinstance(path, str):
        return Path(path)
    return path


def ensure_str_path(path: Union[str, Path]) -> str:
    """문자열 또는 Path 객체를 문자열로 변환

    Args:
        path: 경로 문자열 또는 Path 객체

    Returns:
        경로 문자열
    """
    if isinstance(path, Path):
        return str(path)
    return path


T = TypeVar("T")


def safe_cast(obj: Any, target_type: Type[T]) -> T:
    """안전한 타입 캐스팅

    런타임에 타입 검사를 수행하고 캐스팅합니다.

    Args:
        obj: 캐스팅할 객체
        target_type: 대상 타입

    Returns:
        캐스팅된 객체

    Raises:
        TypeError: 타입 캐스팅이 불가능한 경우
    """
    if isinstance(obj, target_type):
        return obj
    try:
        result = target_type(obj)  # type: ignore
        return result
    except (ValueError, TypeError) as e:
        raise TypeError(f"Cannot cast {type(obj)} to {target_type}: {e}")


__all__ = ["ensure_path", "ensure_str_path", "safe_cast"]
