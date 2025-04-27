"""
체크포인트 및 모델 파일 경로 헬퍼 함수 모듈

이 모듈은 jax_to_tfjs.conf.path.Path 클래스를 활용하여
자주 사용되는 체크포인트 경로를 가져오는 유틸리티 함수를 제공합니다.
"""

from typing import Optional
from pathlib import Path as Path_
from .path import Path


def get_jax_checkpoint_path(subdir: Optional[str] = None) -> Path_:
    """
    JAX 모델 체크포인트 경로를 반환합니다.

    인자:
        subdir (Optional[str]): 추가 하위 디렉토리 (기본값: None)

    반환:
        Path: JAX 체크포인트 경로
    """
    path_manager = Path()
    checkpoint_dir = path_manager.jax_checkpoint_dir
    if subdir:
        return checkpoint_dir / subdir
    return checkpoint_dir


def get_flax_checkpoint_path(subdir: Optional[str] = None) -> Path_:
    """
    Flax 모델 체크포인트 경로를 반환합니다.

    인자:
        subdir (Optional[str]): 추가 하위 디렉토리 (기본값: None)

    반환:
        Path: Flax 체크포인트 경로
    """
    path_manager = Path()
    checkpoint_dir = path_manager.flax_checkpoint_dir
    if subdir:
        return checkpoint_dir / subdir
    return checkpoint_dir


def get_onnx_path() -> Path_:
    """
    ONNX 모델 저장 디렉토리 경로를 반환합니다.

    반환:
        Path: ONNX 모델 저장 디렉토리 경로
    """
    path_manager = Path()
    return path_manager.onnx_dir
