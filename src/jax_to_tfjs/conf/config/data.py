"""
데이터 관련 설정 모듈

데이터 처리에 관련된 설정을 dataclass 형태로 제공합니다.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List


@dataclass
class DataConfig:
    """데이터 관련 설정"""

    cache_data: bool = True
    shuffle_buffer_size: int = 10000
    prefetch_buffer_size: int = 4
    num_parallel_calls: int = -1  # -1은 tf.data.AUTOTUNE을 의미
    normalize_method: str = "divide_by_255"  # 'standard', 'divide_by_255', 'min_max'

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataConfig":
        """딕셔너리에서 설정 객체 생성"""
        return cls(**config_dict)
