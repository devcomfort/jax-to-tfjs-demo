"""
메트릭 계산 유틸리티

모델 평가를 위한 다양한 메트릭을 계산하고 저장하는 기능을 제공합니다.
"""
import numpy as np
import pandas as pd
import json
import logging
import os
from typing import Dict, List, Tuple, Optional, Any, Union
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
    classification_report
)

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None
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
    cm = confusion_matrix(y_true, y_pred)
    
    # 정규화 적용
    if normalize is not None:
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()
        else:
            raise ValueError(f"정규화 방법이 올바르지 않습니다: {normalize}")
        
        # NaN을 0으로 대체
        cm = np.nan_to_num(cm)
        
    return cm

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, Any]:
    """
    다양한 성능 메트릭을 계산합니다.
    
    인자:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        y_proba: 예측 확률 (다중 클래스인 경우 클래스별 확률)
        average: 평균 방법 ('macro', 'micro', 'weighted', 'samples')
        
    반환값:
        계산된 메트릭을 포함하는 딕셔너리
    """
    metrics = {}
    
    # 기본 메트릭 계산
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    # 클래스별 메트릭 계산
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['class_metrics'] = {}
    
    for class_label, class_metrics in class_report.items():
        if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics['class_metrics'][class_label] = {
                'precision': class_metrics['precision'],
                'recall': class_metrics['recall'],
                'f1-score': class_metrics['f1-score'],
                'support': class_metrics['support']
            }
    
    # 혼동 행렬 계산
    metrics['confusion_matrix'] = calculate_confusion_matrix(y_true, y_pred).tolist()
    metrics['confusion_matrix_normalized'] = calculate_confusion_matrix(
        y_true, y_pred, normalize='true'
    ).tolist()
    
    # ROC 곡선과 AUC (가능한 경우)
    if y_proba is not None:
        metrics['roc_auc'] = {}
        metrics['pr_auc'] = {}
        
        # 레이블 종류 확인
        classes = np.unique(y_true)
        n_classes = len(classes)
        
        if n_classes == 2:  # 이진 분류의 경우
            # 클래스 1의 확률만 사용
            if y_proba.shape[1] == 2:
                y_prob = y_proba[:, 1]
            else:
                y_prob = y_proba
                
            # ROC 곡선과 AUC
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            metrics['roc_auc']['micro'] = float(auc(fpr, tpr))
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
            
            # 정밀도-재현율 곡선
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics['pr_auc']['micro'] = float(average_precision_score(y_true, y_prob))
            metrics['pr_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist()
            }
        else:  # 다중 분류의 경우
            if y_proba.shape[1] == n_classes:
                # 클래스별 ROC AUC 계산
                for i, class_label in enumerate(classes):
                    fpr, tpr, _ = roc_curve((y_true == class_label).astype(int), y_proba[:, i])
                    metrics['roc_auc'][str(class_label)] = float(auc(fpr, tpr))
                
                # 클래스별 PR AUC 계산
                for i, class_label in enumerate(classes):
                    precision, recall, _ = precision_recall_curve((y_true == class_label).astype(int), y_proba[:, i])
                    metrics['pr_auc'][str(class_label)] = float(average_precision_score(
                        (y_true == class_label).astype(int), y_proba[:, i]
                    ))
    
    logger.info(f"메트릭 계산 완료: 정확도={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    return metrics

def save_metrics_to_json(
    metrics: Dict[str, Any],
    output_path: Union[str, Path],
    filename: str = 'evaluation_metrics.json'
) -> str:
    """
    계산된 메트릭을 JSON 파일로 저장합니다.
    
    인자:
        metrics: 저장할 메트릭 딕셔너리
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
    
    # JSON으로 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"메트릭을 JSON으로 저장했습니다: {file_path}")
    return str(file_path)

def print_metrics_table(metrics: Dict[str, Any]) -> None:
    """
    계산된 메트릭을 테이블 형식으로 출력합니다.
    
    인자:
        metrics: 출력할 메트릭 딕셔너리
    """
    # 기본 메트릭 출력
    print("\n===== 모델 평가 결과 =====")
    print(f"정확도 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"정밀도 (Precision): {metrics['precision']:.4f}")
    print(f"재현율 (Recall): {metrics['recall']:.4f}")
    print(f"F1 점수: {metrics['f1']:.4f}")
    
    # ROC AUC 출력 (있는 경우)
    if 'roc_auc' in metrics:
        print("\n----- ROC AUC -----")
        for class_name, auc_value in metrics['roc_auc'].items():
            print(f"클래스 {class_name}: {auc_value:.4f}")
    
    # 클래스별 메트릭 출력 (있는 경우)
    if 'class_metrics' in metrics:
        print("\n----- 클래스별 메트릭 -----")
        
        # 데이터프레임으로 변환하여 테이블 형식으로 출력
        class_metrics_df = pd.DataFrame.from_dict(
            {k: v for k, v in metrics['class_metrics'].items()},
            orient='index'
        )
        
        print(class_metrics_df.round(4))
    
    print("\n============================")

def load_metrics_from_json(json_path: str) -> Dict[str, Any]:
    """
    JSON 파일에서 메트릭을 로드합니다.
    
    인자:
        json_path: 로드할 JSON 파일 경로
        
    반환값:
        Dict[str, Any]: 로드된 메트릭 딕셔너리
    """
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics 