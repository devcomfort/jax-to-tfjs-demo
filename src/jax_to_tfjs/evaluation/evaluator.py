"""
모델 평가 유틸리티

JAX 및 Flax 모델의 성능을 평가하는 기능을 제공합니다.
"""
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import logging
from jax_to_tfjs.models.jax_mnist_cnn import cnn_forward as jax_forward
from jax_to_tfjs.evaluation.metrics import calculate_metrics

# 경고 비활성화
logging.getLogger('jax').setLevel(logging.ERROR)

def evaluate_jax_model(checkpoint, test_images, test_labels, with_probs=False):
    """
    JAX 모델 평가
    
    인자:
        checkpoint: JAX 모델 체크포인트 (params 또는 체크포인트 객체)
        test_images: 테스트 이미지 데이터
        test_labels: 테스트 레이블 데이터
        with_probs: 메트릭 계산 및 반환 여부
    
    반환:
        with_probs가 False인 경우:
            tuple: (accuracy, predictions, logits) - 정확도, 예측, 로짓 값
        with_probs가 True인 경우:
            tuple: (metrics, predictions, probabilities) - 메트릭 딕셔너리, 예측, 확률 값
    """
    # 체크포인트에서 params 추출
    if isinstance(checkpoint, dict):
        # checkpoint가 딕셔너리인 경우
        if 'params' in checkpoint:
            params = checkpoint['params']
        elif 'model' in checkpoint and isinstance(checkpoint['model'], dict) and 'params' in checkpoint['model']:
            params = checkpoint['model']['params']
        else:
            # 직접 params인 경우
            params = checkpoint
    else:
        # 다른 유형의 체크포인트인 경우
        params = checkpoint
    
    # JAX 배열로 변환
    test_images_jax = jnp.array(test_images)
    test_labels_jax = jnp.array(test_labels)
    
    # 배치 크기
    batch_size = 100
    num_samples = test_images_jax.shape[0]
    num_batches = num_samples // batch_size
    
    correct = 0
    all_predictions = []
    all_logits = []
    
    for i in tqdm(range(num_batches), desc="JAX 모델 평가 중"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_images = test_images_jax[start_idx:end_idx]
        batch_labels = test_labels_jax[start_idx:end_idx]
        
        try:
            # 예측 시도
            logits = jax_forward(params, batch_images)
            predictions = jnp.argmax(logits, axis=1)
            
            # 정확도 계산
            correct += jnp.sum(predictions == batch_labels)
            all_predictions.append(predictions)
            all_logits.append(logits)
        except Exception as e:
            # 오류 발생 시 체크포인트 구조 로깅
            logging.error(f"예측 중 오류 발생: {e}")
            logging.error(f"체크포인트 키: {list(params.keys()) if isinstance(params, dict) else 'Not a dict'}")
            raise
    
    accuracy = correct / num_samples
    predictions = jnp.concatenate(all_predictions)
    logits = jnp.concatenate(all_logits)
    
    if with_probs:
        # 확률 및 메트릭 계산
        probs = jax.nn.softmax(logits, axis=-1)
        metrics = calculate_metrics(test_labels_jax, predictions, probs)
        return metrics, predictions, probs
    else:
        return accuracy, predictions, logits

def evaluate_flax_model(state, test_images, test_labels, with_probs=False):
    """
    Flax 모델 평가
    
    인자:
        state: Flax 모델 상태 (TrainState)
        test_images: 테스트 이미지 데이터
        test_labels: 테스트 레이블 데이터
        with_probs: 메트릭 계산 및 반환 여부
    
    반환:
        with_probs가 False인 경우:
            tuple: (accuracy, predictions, logits) - 정확도, 예측, 로짓 값
        with_probs가 True인 경우:
            tuple: (metrics, predictions, probabilities) - 메트릭 딕셔너리, 예측, 확률 값
    """
    # JAX 배열로 변환
    test_images_jax = jnp.array(test_images)
    test_labels_jax = jnp.array(test_labels)
    
    # 배치 크기
    batch_size = 100
    num_samples = test_images_jax.shape[0]
    num_batches = num_samples // batch_size
    
    correct = 0
    all_predictions = []
    all_logits = []
    
    for i in tqdm(range(num_batches), desc="Flax 모델 평가 중"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_images = test_images_jax[start_idx:end_idx]
        batch_labels = test_labels_jax[start_idx:end_idx]
        
        # 예측
        logits = state.apply_fn({'params': state.params}, batch_images)
        predictions = jnp.argmax(logits, axis=1)
        
        # 정확도 계산
        correct += jnp.sum(predictions == batch_labels)
        all_predictions.append(predictions)
        all_logits.append(logits)
    
    accuracy = correct / num_samples
    predictions = jnp.concatenate(all_predictions)
    logits = jnp.concatenate(all_logits)
    
    if with_probs:
        # 확률 및 메트릭 계산
        probs = jax.nn.softmax(logits, axis=-1)
        metrics = calculate_metrics(test_labels_jax, predictions, probs)
        return metrics, predictions, probs
    else:
        return accuracy, predictions, logits 