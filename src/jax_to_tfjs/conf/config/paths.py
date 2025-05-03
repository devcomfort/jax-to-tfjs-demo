"""
경로 관련 설정 모듈

파일 경로와 관련된 설정을 dataclass 형태로 제공합니다.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class PathConfig:
    """경로 관련 설정"""

    checkpoint_dir: str = "checkpoints"
    jax_checkpoint_dir: str = "jax_mnist"
    flax_checkpoint_dir: str = "flax_mnist"
    results_dir: str = "results"
    web_model_dir: str = "model"
    data_cache_dir: str = "data_cache"
    log_dir: str = "logs"

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PathConfig":
        """딕셔너리에서 설정 객체 생성"""
        return cls(**config_dict)
