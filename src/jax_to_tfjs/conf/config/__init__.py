"""
설정 모듈 패키지

dataclass 기반 설정 관리 시스템을 제공합니다.
"""

import os
from .training import TrainingConfig, ModelType
from .data import DataConfig
from .paths import PathConfig
from .conversion import ConversionConfig
from .loader import Config


# 싱글톤 인스턴스
_config = None


def get_config() -> Config:
    """전역 설정 객체 가져오기"""
    global _config
    if _config is None:
        _config = Config()

        # 설정 파일 확인
        config_file = os.environ.get("JAX_TFJS_CONFIG_FILE", "config.json")
        if os.path.exists(config_file):
            _config.load_from_file(config_file)

    return _config


def update_config(config_dict):
    """전역 설정 객체 업데이트"""
    config = get_config()
    config.update_from_dict(config_dict)
    return config


__all__ = [
    "TrainingConfig",
    "DataConfig",
    "PathConfig",
    "ConversionConfig",
    "ModelType",
    "Config",
    "get_config",
    "update_config",
]
