"""
메트릭 타입 모듈

모델 평가 메트릭 관련 타입 정의를 제공합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TypedDict


# 메트릭 결과를 위한 TypedDict 정의
class MetricsResult(TypedDict, total=False):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion_matrix: Optional[List[List[int]]]
    class_metrics: Dict[str, Dict[str, float]]
    mse: Optional[float]
    rmse: Optional[float]
    mae: Optional[float]
    r2: Optional[float]
    error: Optional[str]


@dataclass
class ModelMetrics:
    """
    모델 평가 메트릭 데이터 클래스
    """

    # 기본 분류 메트릭
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # 추가 분류 메트릭 (옵션)
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 회귀 메트릭 (옵션)
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None

    # 에러 정보
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, metrics_dict: Dict[str, Any]) -> "ModelMetrics":
        """
        딕셔너리로부터 ModelMetrics 객체를 생성합니다.

        Args:
            metrics_dict: 메트릭 정보가 담긴 딕셔너리

        Returns:
            ModelMetrics 객체
        """
        # 필수 필드는 기본값으로 초기화하고, 딕셔너리에 있는 값으로 업데이트
        return cls(
            accuracy=metrics_dict.get("accuracy", 0.0),
            precision=metrics_dict.get("precision", 0.0),
            recall=metrics_dict.get("recall", 0.0),
            f1=metrics_dict.get("f1", 0.0),
            roc_auc=metrics_dict.get("roc_auc"),
            confusion_matrix=metrics_dict.get("confusion_matrix"),
            class_metrics=metrics_dict.get("class_metrics", {}),
            mse=metrics_dict.get("mse"),
            rmse=metrics_dict.get("rmse"),
            mae=metrics_dict.get("mae"),
            r2=metrics_dict.get("r2"),
            error=metrics_dict.get("error"),
        )

    def to_dict(self) -> MetricsResult:
        """
        ModelMetrics 객체를 딕셔너리로 변환합니다.

        Returns:
            메트릭 정보가 담긴 딕셔너리
        """
        # TypedDict을 사용하여 타입 안전성 확보
        result: MetricsResult = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

        # None이 아닌 옵션 필드만 추가
        if self.roc_auc is not None:
            result["roc_auc"] = self.roc_auc
        if self.confusion_matrix is not None:
            result["confusion_matrix"] = self.confusion_matrix
        if self.class_metrics:
            result["class_metrics"] = self.class_metrics
        if self.mse is not None:
            result["mse"] = self.mse
        if self.rmse is not None:
            result["rmse"] = self.rmse
        if self.mae is not None:
            result["mae"] = self.mae
        if self.r2 is not None:
            result["r2"] = self.r2
        if self.error is not None:
            result["error"] = self.error

        return result
