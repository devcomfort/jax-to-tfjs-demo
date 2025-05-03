"""
학습 관련 설정 모듈

모델 학습에 관련된 설정을 dataclass 형태로 제공합니다.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any


class ModelType(Enum):
    """모델 유형 열거형"""

    JAX = "jax_mnist"
    FLAX = "flax_mnist"


@dataclass
class TrainingConfig:
    """학습 관련 설정"""

    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 0.001
    evaluate_after_training: bool = True
    validation_split: float = 0.1
    early_stopping: bool = False
    early_stopping_patience: int = 3
    model_type: ModelType = ModelType.JAX

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        data = asdict(self)
        # Enum 값 처리
        if self.model_type:
            data["model_type"] = self.model_type.value
        return data

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """딕셔너리에서 설정 객체 생성"""
        # Enum 값 처리
        if "model_type" in config_dict:
            model_type = config_dict["model_type"]
            if isinstance(model_type, str):
                if model_type.lower() in ["jax", "jax_mnist"]:
                    config_dict["model_type"] = ModelType.JAX
                elif model_type.lower() in ["flax", "flax_mnist"]:
                    config_dict["model_type"] = ModelType.FLAX

        return cls(**config_dict)
