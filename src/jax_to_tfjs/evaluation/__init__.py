"""
모델 평가 유틸리티 모듈

JAX/Flax 모델의 평가, 시각화, 메트릭 계산에 관련된 기능을 제공합니다.
"""

from .data import load_mnist_test
from .evaluator import evaluate_jax_model, evaluate_flax_model
from .visualization import (
    visualize_image_predictions,
    visualize_confusion_matrix,
    visualize_roc_curve,
    visualize_precision_recall_curve,
    visualize_metrics_by_class
)
from .metrics import (
    calculate_confusion_matrix,
    calculate_metrics,
    save_metrics_to_json,
    print_metrics_table
)

# 편의를 위해 모든 함수를 직접 노출
__all__ = [
    'load_mnist_test',
    'evaluate_jax_model',
    'evaluate_flax_model',
    'visualize_image_predictions',
    'visualize_confusion_matrix',
    'visualize_roc_curve',
    'visualize_precision_recall_curve',
    'visualize_metrics_by_class',
    'calculate_confusion_matrix',
    'calculate_metrics',
    'save_metrics_to_json',
    'print_metrics_table'
] 