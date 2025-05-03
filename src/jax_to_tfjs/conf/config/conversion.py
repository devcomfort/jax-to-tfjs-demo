"""
모델 변환 관련 설정 모듈

모델 변환과 관련된 설정을 dataclass 형태로 제공합니다.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class ConversionConfig:
    """모델 변환 관련 설정"""

    quantize: bool = False
    quantization_type: str = "uint8"  # 'uint8', 'uint16'
    web_model_name: str = "model.json"
    include_optimizer: bool = False
    save_format: str = "tfjs"  # 'tfjs', 'keras_saved_model'

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConversionConfig":
        """딕셔너리에서 설정 객체 생성"""
        return cls(**config_dict)
