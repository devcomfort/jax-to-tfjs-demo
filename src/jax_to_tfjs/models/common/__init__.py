"""
공통 유틸리티 모듈

JAX 및 FLAX 모델에서 공통으로 사용하는 유틸리티 함수들을 제공합니다.
"""

from .train_utils import train_and_evaluate

__all__ = [
    "train_and_evaluate",
]
