"""
설정 로더 모듈

환경 변수, 설정 파일, CLI 인자를 통합 관리하는 시스템입니다.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .training import TrainingConfig, ModelType
from .data import DataConfig
from .paths import PathConfig
from .conversion import ConversionConfig

logger = logging.getLogger(__name__)
ENV_PREFIX = "JAX_TFJS_"


@dataclass
class Config:
    """통합 설정 클래스"""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    conversion: ConversionConfig = field(default_factory=ConversionConfig)
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """초기화 후 처리: 환경 변수에서 설정 로드"""
        self._load_from_env()

    def _load_from_env(self):
        """환경 변수에서 설정 로드"""
        # 학습 설정
        batch_size_env = os.environ.get(f"{ENV_PREFIX}BATCH_SIZE")
        if batch_size_env:
            self.training.batch_size = int(batch_size_env)

        epochs_env = os.environ.get(f"{ENV_PREFIX}EPOCHS")
        if epochs_env:
            self.training.epochs = int(epochs_env)

        lr_env = os.environ.get(f"{ENV_PREFIX}LEARNING_RATE")
        if lr_env:
            self.training.learning_rate = float(lr_env)

        model_type_env = os.environ.get(f"{ENV_PREFIX}MODEL_TYPE")
        if model_type_env:
            if model_type_env.lower() == "jax":
                self.training.model_type = ModelType.JAX
            elif model_type_env.lower() == "flax":
                self.training.model_type = ModelType.FLAX

        # 데이터 설정
        cache_data_env = os.environ.get(f"{ENV_PREFIX}CACHE_DATA")
        if cache_data_env:
            self.data.cache_data = cache_data_env.lower() in ("true", "1", "yes")

        # 경로 설정
        checkpoint_dir_env = os.environ.get(f"{ENV_PREFIX}CHECKPOINT_DIR")
        if checkpoint_dir_env:
            self.paths.checkpoint_dir = checkpoint_dir_env

        # 변환 설정
        quantize_env = os.environ.get(f"{ENV_PREFIX}QUANTIZE")
        if quantize_env:
            self.conversion.quantize = quantize_env.lower() in ("true", "1", "yes")

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """딕셔너리에서 설정 업데이트"""
        # 학습 설정
        if "training" in config_dict:
            training_dict = config_dict["training"]
            self.training = TrainingConfig.from_dict(training_dict)

        # 데이터 설정
        if "data" in config_dict:
            data_dict = config_dict["data"]
            self.data = DataConfig.from_dict(data_dict)

        # 경로 설정
        if "paths" in config_dict:
            paths_dict = config_dict["paths"]
            self.paths = PathConfig.from_dict(paths_dict)

        # 변환 설정
        if "conversion" in config_dict:
            conversion_dict = config_dict["conversion"]
            self.conversion = ConversionConfig.from_dict(conversion_dict)

        # 사용자 정의 설정
        if "custom_settings" in config_dict:
            self.custom_settings.update(config_dict["custom_settings"])

    def load_from_file(self, config_file: Union[str, Path]):
        """설정 파일에서 설정 로드"""
        try:
            with open(config_file, "r") as f:
                config_dict = json.load(f)
                self.update_from_dict(config_dict)
            logger.info(f"설정을 파일 {config_file}에서 로드했습니다.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"설정 파일 {config_file} 로드 실패: {str(e)}")

    def save_to_file(self, config_file: Union[str, Path]):
        """설정을 파일로 저장"""
        config_dict = self.to_dict()
        try:
            with open(config_file, "w") as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"설정을 파일 {config_file}에 저장했습니다.")
        except Exception as e:
            logger.error(f"설정 파일 {config_file} 저장 실패: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "paths": self.paths.to_dict(),
            "conversion": self.conversion.to_dict(),
            "custom_settings": self.custom_settings,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """딕셔너리에서 설정 객체 생성"""
        config = cls()
        config.update_from_dict(config_dict)
        return config
