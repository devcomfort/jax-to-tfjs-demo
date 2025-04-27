"""
경로 설정 모듈

프로젝트 관련 경로 설정을 제공합니다.
"""

from pathlib import Path as Path_
from os.path import dirname

PROJECT_ROOT = Path_(dirname(__file__)).parent.parent.parent


class Path:
    def __init__(self):
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints"
        self.jax_checkpoint_dir = self.checkpoint_dir / "jax"
        self.flax_checkpoint_dir = self.checkpoint_dir / "flax"
        self.onnx_dir = PROJECT_ROOT / "onnx"
