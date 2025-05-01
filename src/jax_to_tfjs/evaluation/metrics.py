"""
메트릭 계산 유틸리티

모델 평가를 위한 다양한 메트릭을 계산하고 저장하는 기능을 제공합니다.
"""

import numpy as np
import pandas as pd
import json
import logging
import os
from typing import Dict, List, Tuple, Optional, Any, Union, Literal, cast, TypedDict
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from .types.metrics import ModelMetrics

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Literal 타입 정의
AveragePrecisionRecallF1 = Literal[
    "binary", "micro", "macro", "weighted", "samples", None
]
AverageROCAUC = Literal["macro", "weighted", "micro", None]


def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[Literal["true", "pred", "all"]] = None,
) -> np.ndarray:
    """
    혼동 행렬을 계산합니다.

    인자:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        normalize: 정규화 방법 ('true', 'pred', 'all', None)

    반환값:
        혼동 행렬
    """
    # 입력 데이터 검증
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("비어있는 레이블이 있습니다. 빈 혼동 행렬을 반환합니다.")
        # 빈 혼동 행렬 반환 (1x1)
        return np.zeros((1, 1), dtype=np.int32)

    # 길이 불일치 확인
    if len(y_true) != len(y_pred):
        logger.warning(
            f"레이블 길이 불일치: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
        # 더 작은 길이로 자르기
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    cm = confusion_matrix(y_true, y_pred)

    # 정규화 적용
    if normalize is not None:
        if normalize == "true":
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == "pred":
            cm = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :]
        elif normalize == "all":
            cm = cm.astype("float") / cm.sum()
        else:
            raise ValueError(f"정규화 방법이 올바르지 않습니다: {normalize}")

        # NaN을 0으로 대체
        cm = np.nan_to_num(cm)

    return cm


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    task_type: str = "classification",
    average: AveragePrecisionRecallF1 = "macro",
    classes: Optional[List[str]] = None,
) -> ModelMetrics:
    """
    여러 성능 지표를 계산합니다.

    Args:
        y_true: 실제 레이블
        y_pred: 예측된 레이블
        y_prob: (옵션) 예측 확률
        task_type: 작업 유형 ("classification" 또는 "regression")
        average: 평균 방법 (분류 메트릭스에서 사용됨)
        classes: 클래스 이름 목록

    Returns:
        계산된 지표의 ModelMetrics 객체
    """
    if y_true is None or y_pred is None:
        logger.error("y_true와 y_pred는 None이 될 수 없습니다.")
        return ModelMetrics(error="입력 데이터가 없습니다.")

    if len(y_true) == 0 or len(y_pred) == 0:
        logger.error("y_true와 y_pred는 비어 있을 수 없습니다.")
        return ModelMetrics(error="입력 데이터가 비어 있습니다.")

    # 길이가 일치하는지 확인
    if len(y_true) != len(y_pred):
        logger.warning(
            f"y_true와 y_pred의 길이가 일치하지 않습니다: {len(y_true)} vs {len(y_pred)}"
        )
        # 더 짧은 길이에 맞춰 자름
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        if y_prob is not None and len(y_prob) > min_len:
            y_prob = y_prob[:min_len]

    metrics = ModelMetrics()

    try:
        if task_type.lower() == "classification":
            # 분류 메트릭 계산
            metrics.accuracy = float(accuracy_score(y_true, y_pred))

            # 다중 클래스 분류인지 확인
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))

            # 이진 분류이거나 멀티클래스로 평균을 계산할 수 있는 경우
            if len(unique_labels) <= 2 or average is not None:
                try:
                    metrics.precision = float(
                        precision_score(
                            y_true, y_pred, average=average, zero_division=0
                        )
                    )
                    metrics.recall = float(
                        recall_score(y_true, y_pred, average=average, zero_division=0)
                    )
                    metrics.f1 = float(
                        f1_score(y_true, y_pred, average=average, zero_division=0)
                    )
                except Exception as e:
                    logger.warning(f"분류 메트릭 계산 중 오류 발생: {e}")

            # 예측 확률이 제공된 경우 ROC AUC 계산
            if y_prob is not None:
                try:
                    # 이진 분류
                    if len(unique_labels) <= 2:
                        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                            # 두 번째 클래스의 확률 사용
                            prob = y_prob[:, 1]
                        else:
                            prob = y_prob.ravel()

                        metrics.roc_auc = float(roc_auc_score(y_true, prob))
                    # 다중 클래스
                    elif y_prob.shape[1] > 1:
                        # average 인자가 "binary"인 경우 다중 클래스에서는 사용할 수 없으므로 조정
                        actual_average: AverageROCAUC = (
                            "macro"
                            if average == "binary"
                            else cast(AverageROCAUC, average)
                        )
                        metrics.roc_auc = float(
                            roc_auc_score(
                                y_true,
                                y_prob,
                                multi_class="ovr",
                                average=actual_average,
                            )
                        )
                except Exception as e:
                    logger.warning(f"ROC AUC 계산 중 오류 발생: {e}")

            # 혼동 행렬 계산
            try:
                metrics.confusion_matrix = confusion_matrix(y_true, y_pred).tolist()
            except Exception as e:
                logger.warning(f"혼동 행렬 계산 중 오류 발생: {e}")

            # 분류 보고서 계산
            try:
                class_report = classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    target_names=classes,
                    zero_division=0,
                )

                if isinstance(class_report, dict):
                    # accuracy, macro avg, weighted avg 키 건너뛰기
                    metrics.class_metrics = {}
                    for class_name, values in class_report.items():
                        if class_name not in ["accuracy", "macro avg", "weighted avg"]:
                            if isinstance(values, dict):
                                metrics.class_metrics[class_name] = values
                else:
                    logger.warning("분류 보고서가 딕셔너리 형식이 아닙니다.")
            except Exception as e:
                logger.warning(f"분류 보고서 계산 중 오류 발생: {e}")

        elif task_type.lower() == "regression":
            # 회귀 메트릭 계산
            metrics.mse = float(mean_squared_error(y_true, y_pred))
            metrics.rmse = float(np.sqrt(metrics.mse))
            metrics.mae = float(mean_absolute_error(y_true, y_pred))
            try:
                metrics.r2 = float(r2_score(y_true, y_pred))
            except Exception as e:
                logger.warning(f"R2 점수 계산 중 오류 발생: {e}")
                metrics.r2 = float(0)  # 기본값으로 0 설정

    except Exception as e:
        logger.error(f"메트릭 계산 중 오류 발생: {e}")
        metrics.error = str(e)

    return metrics


def save_metrics_to_json(
    metrics: Union[ModelMetrics, Dict[str, Any]],
    output_path: Union[str, Path],
    filename: str = "evaluation_metrics.json",
) -> str:
    """
    계산된 메트릭을 JSON 파일로 저장합니다.

    인자:
        metrics: 저장할 메트릭 (ModelMetrics 객체 또는 딕셔너리)
        output_path: 저장할 디렉토리 경로
        filename: 저장할 파일 이름

    반환값:
        저장된 파일의 전체 경로
    """
    # 디렉토리가 없으면 생성
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 파일 경로 생성
    file_path = output_path / filename

    # ModelMetrics 객체인 경우 딕셔너리로 변환
    if isinstance(metrics, ModelMetrics):
        metrics_dict = metrics.to_dict()
    else:
        metrics_dict = metrics

    # JSON으로 저장
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"메트릭을 JSON으로 저장했습니다: {file_path}")
    return str(file_path)


def print_metrics_table(metrics: Union[ModelMetrics, Dict[str, Any]]) -> None:
    """
    계산된 메트릭을 테이블 형식으로 출력합니다.

    인자:
        metrics: 출력할 메트릭 (ModelMetrics 객체 또는 딕셔너리)
    """
    # ModelMetrics 객체인 경우 딕셔너리로 변환
    if isinstance(metrics, ModelMetrics):
        metrics_dict = metrics.to_dict()
    else:
        metrics_dict = metrics

    # 기본 메트릭 출력
    print("\n===== 모델 평가 결과 =====")

    if "accuracy" in metrics_dict:
        print(f"정확도 (Accuracy): {metrics_dict['accuracy']:.4f}")
    if "precision" in metrics_dict:
        print(f"정밀도 (Precision): {metrics_dict['precision']:.4f}")
    if "recall" in metrics_dict:
        print(f"재현율 (Recall): {metrics_dict['recall']:.4f}")
    if "f1" in metrics_dict:
        print(f"F1 점수: {metrics_dict['f1']:.4f}")

    # 회귀 메트릭 출력
    if "mse" in metrics_dict:
        print(f"평균 제곱 오차 (MSE): {metrics_dict['mse']:.4f}")
    if "rmse" in metrics_dict:
        print(f"평균 제곱근 오차 (RMSE): {metrics_dict['rmse']:.4f}")
    if "mae" in metrics_dict:
        print(f"평균 절대 오차 (MAE): {metrics_dict['mae']:.4f}")
    if "r2" in metrics_dict:
        print(f"결정 계수 (R²): {metrics_dict['r2']:.4f}")

    # ROC AUC 출력
    if "roc_auc" in metrics_dict:
        # 단일 값
        if isinstance(metrics_dict["roc_auc"], (int, float)):
            print(f"ROC AUC: {metrics_dict['roc_auc']:.4f}")
        # 클래스별 값
        elif isinstance(metrics_dict["roc_auc"], dict):
            print("\nROC AUC:")
            for class_name, auc_value in metrics_dict["roc_auc"].items():
                print(f"  - {class_name}: {auc_value:.4f}")

    # 클래스별 메트릭 출력
    if "class_metrics" in metrics_dict:
        print("\n클래스별 메트릭:")
        try:
            # 판다스 데이터프레임으로 변환하여 출력
            class_metrics_df = pd.DataFrame.from_dict(
                {k: v for k, v in metrics_dict["class_metrics"].items()}, orient="index"
            )
            # 소수점 4자리까지 반올림하여 출력
            print(class_metrics_df.round(4))
        except Exception as e:
            print(f"클래스별 메트릭 출력 중 오류 발생: {e}")


def load_metrics_from_json(json_path: str) -> ModelMetrics:
    """
    JSON 파일에서 메트릭을 로드합니다.

    인자:
        json_path: JSON 파일 경로

    반환값:
        로드된 메트릭 객체
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            metrics_dict = json.load(f)
        return ModelMetrics.from_dict(metrics_dict)
    except Exception as e:
        logger.error(f"메트릭 로드 중 오류 발생: {e}")
        return ModelMetrics(error=f"JSON 로드 오류: {str(e)}")
