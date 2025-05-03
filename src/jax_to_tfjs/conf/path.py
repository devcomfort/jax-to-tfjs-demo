"""
경로 설정 모듈

프로젝트 관련 경로 설정 및 유틸리티 함수를 제공합니다.
"""

from pathlib import Path as PathLib
from os.path import dirname
from typing import Optional

PROJECT_ROOT = PathLib(dirname(__file__)).parent.parent.parent


class Path:
    def __init__(self):
        self.project_root = PROJECT_ROOT

        # 기본 체크포인트 디렉토리
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.jax_checkpoint_dir = self.checkpoint_dir / "jax"
        self.flax_checkpoint_dir = self.checkpoint_dir / "flax"

        # 추가된 디렉토리
        self.onnx_dir = self.project_root / "onnx"
        self.tensorflow_dir = self.project_root / "tensorflow"
        self.tfjs_dir = self.project_root / "tfjs"

    def get_jax_checkpoint_path(self, subdir: Optional[str] = None) -> PathLib:
        """
        JAX 모델 체크포인트 경로를 반환합니다.

        인자:
            subdir (Optional[str]): 추가 하위 디렉토리 (기본값: None)

        반환:
            PathLib: JAX 체크포인트 경로
        """
        if subdir:
            return self.jax_checkpoint_dir / subdir
        return self.jax_checkpoint_dir

    def get_flax_checkpoint_path(self, subdir: Optional[str] = None) -> PathLib:
        """
        Flax 모델 체크포인트 경로를 반환합니다.

        인자:
            subdir (Optional[str]): 추가 하위 디렉토리 (기본값: None)

        반환:
            PathLib: Flax 체크포인트 경로
        """
        if subdir:
            return self.flax_checkpoint_dir / subdir
        return self.flax_checkpoint_dir

    def get_onnx_path(self, model_name: Optional[str] = None) -> PathLib:
        """
        ONNX 모델 저장 경로를 반환합니다.

        인자:
            model_name (Optional[str]): 모델 이름 (기본값: None)

        반환:
            PathLib: ONNX 모델 저장 경로
        """
        if model_name:
            return self.onnx_dir / f"{model_name}.onnx"
        return self.onnx_dir

    def get_tensorflow_path(self, model_name: Optional[str] = None) -> PathLib:
        """
        TensorFlow 모델 저장 경로를 반환합니다.

        인자:
            model_name (Optional[str]): 모델 이름 (기본값: None)

        반환:
            PathLib: TensorFlow 모델 저장 경로
        """
        if model_name:
            return self.tensorflow_dir / model_name
        return self.tensorflow_dir

    def get_tfjs_path(self, model_name: Optional[str] = None) -> PathLib:
        """
        TensorFlow.js 모델 저장 경로를 반환합니다.

        인자:
            model_name (Optional[str]): 모델 이름 (기본값: None)

        반환:
            PathLib: TensorFlow.js 모델 저장 경로
        """
        if model_name:
            return self.tfjs_dir / model_name
        return self.tfjs_dir


# 싱글톤 인스턴스
_path_instance = None


def get_path() -> Path:
    """
    경로 관리자 싱글톤 인스턴스를 반환합니다.

    반환:
        Path: 경로 관리자 인스턴스
    """
    global _path_instance
    if _path_instance is None:
        _path_instance = Path()
    return _path_instance
