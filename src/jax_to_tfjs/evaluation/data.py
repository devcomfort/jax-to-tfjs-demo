"""
데이터 로딩 유틸리티

모델 평가에 필요한 데이터셋을 로드하고 전처리하는 기능을 제공합니다.
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_mnist_test(normalize: bool = True, 
                   as_numpy: bool = True,
                   num_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    MNIST 테스트 데이터셋을 로드합니다.
    
    인자:
        normalize: 이미지 정규화 여부 ([0, 1] 범위로 변환)
        as_numpy: numpy 배열로 변환 여부
        num_samples: 로드할 샘플 수 (None인 경우 전체 데이터셋 로드)
        
    반환값:
        튜플: (이미지, 레이블)
    """
    logger.info("MNIST 테스트 데이터셋을 로드합니다...")
    
    # MNIST 테스트 데이터셋 로드
    ds = tfds.load('mnist', split='test', as_supervised=True)
    
    # 데이터셋 전처리
    def preprocess(image, label):
        # 이미지를 float32로 변환
        image = tf.cast(image, tf.float32)
        
        if normalize:
            # [0, 1] 범위로 정규화
            image = image / 255.0
            
        return image, label
    
    # 전처리 적용
    ds = ds.map(preprocess)
    
    # 전체 데이터를 메모리에 로드
    if as_numpy:
        images = []
        labels = []
        
        # 데이터셋 제한 (선택 사항)
        if num_samples is not None:
            ds = ds.take(num_samples)
        
        # 데이터셋 반복
        for image, label in tfds.as_numpy(ds):
            images.append(image)
            labels.append(label)
            
        # numpy 배열로 변환
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"MNIST 테스트 데이터 로드 완료: {len(images)} 샘플")
        return images, labels
    else:
        # TensorFlow 데이터셋 반환
        if num_samples is not None:
            ds = ds.take(num_samples)
        logger.info("MNIST 테스트 데이터셋 로드 완료")
        return ds

def load_fashion_mnist_test(normalize: bool = True,
                          as_numpy: bool = True,
                          num_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fashion MNIST 테스트 데이터셋을 로드합니다.
    
    인자:
        normalize: 이미지 정규화 여부 ([0, 1] 범위로 변환)
        as_numpy: numpy 배열로 변환 여부
        num_samples: 로드할 샘플 수 (None인 경우 전체 데이터셋 로드)
        
    반환값:
        튜플: (이미지, 레이블)
    """
    logger.info("Fashion MNIST 테스트 데이터셋을 로드합니다...")
    
    # Fashion MNIST 테스트 데이터셋 로드
    ds = tfds.load('fashion_mnist', split='test', as_supervised=True)
    
    # 데이터셋 전처리
    def preprocess(image, label):
        # 이미지를 float32로 변환
        image = tf.cast(image, tf.float32)
        
        if normalize:
            # [0, 1] 범위로 정규화
            image = image / 255.0
            
        return image, label
    
    # 전처리 적용
    ds = ds.map(preprocess)
    
    # 전체 데이터를 메모리에 로드
    if as_numpy:
        images = []
        labels = []
        
        # 데이터셋 제한 (선택 사항)
        if num_samples is not None:
            ds = ds.take(num_samples)
        
        # 데이터셋 반복
        for image, label in tfds.as_numpy(ds):
            images.append(image)
            labels.append(label)
            
        # numpy 배열로 변환
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"Fashion MNIST 테스트 데이터 로드 완료: {len(images)} 샘플")
        return images, labels
    else:
        # TensorFlow 데이터셋 반환
        if num_samples is not None:
            ds = ds.take(num_samples)
        logger.info("Fashion MNIST 테스트 데이터셋 로드 완료")
        return ds

def load_cifar10_test(normalize: bool = True,
                     as_numpy: bool = True,
                     num_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    CIFAR-10 테스트 데이터셋을 로드합니다.
    
    인자:
        normalize: 이미지 정규화 여부 ([0, 1] 범위로 변환)
        as_numpy: numpy 배열로 변환 여부
        num_samples: 로드할 샘플 수 (None인 경우 전체 데이터셋 로드)
        
    반환값:
        튜플: (이미지, 레이블)
    """
    logger.info("CIFAR-10 테스트 데이터셋을 로드합니다...")
    
    # CIFAR-10 테스트 데이터셋 로드
    ds = tfds.load('cifar10', split='test', as_supervised=True)
    
    # 데이터셋 전처리
    def preprocess(image, label):
        # 이미지를 float32로 변환
        image = tf.cast(image, tf.float32)
        
        if normalize:
            # [0, 1] 범위로 정규화
            image = image / 255.0
            
        return image, label
    
    # 전처리 적용
    ds = ds.map(preprocess)
    
    # 전체 데이터를 메모리에 로드
    if as_numpy:
        images = []
        labels = []
        
        # 데이터셋 제한 (선택 사항)
        if num_samples is not None:
            ds = ds.take(num_samples)
        
        # 데이터셋 반복
        for image, label in tfds.as_numpy(ds):
            images.append(image)
            labels.append(label)
            
        # numpy 배열로 변환
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"CIFAR-10 테스트 데이터 로드 완료: {len(images)} 샘플")
        return images, labels
    else:
        # TensorFlow 데이터셋 반환
        if num_samples is not None:
            ds = ds.take(num_samples)
        logger.info("CIFAR-10 테스트 데이터셋 로드 완료")
        return ds

def preprocess_for_model(images: np.ndarray, 
                        model_type: str,
                        target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    이미지를 모델에 맞게 전처리합니다.
    
    인자:
        images: 이미지 배열
        model_type: 모델 타입 ('jax', 'flax', 'tensorflow', 'pytorch')
        target_size: 목표 이미지 크기 (높이, 너비)
        
    반환값:
        전처리된 이미지 배열
    """
    preprocessed_images = images.copy()
    
    # 타겟 사이즈가 지정된 경우 리사이즈
    if target_size is not None:
        if len(images.shape) == 4:
            # 배치 이미지인 경우
            original_shape = images.shape
            
            # TensorFlow 사용하여 리사이징
            temp_images = []
            for img in images:
                resized = tf.image.resize(img, target_size).numpy()
                temp_images.append(resized)
            
            preprocessed_images = np.array(temp_images)
            logger.info(f"이미지 크기 변경: {original_shape} -> {preprocessed_images.shape}")
        else:
            # 단일 이미지인 경우
            preprocessed_images = tf.image.resize(images, target_size).numpy()
    
    # 모델 타입에 따른 추가 전처리
    if model_type.lower() in ['jax', 'flax']:
        # JAX/Flax 모델은 일반적으로 [0, 1] 범위의 float32 예상
        if preprocessed_images.dtype != np.float32:
            preprocessed_images = preprocessed_images.astype(np.float32)
        
        # 이미 정규화된 경우를 확인
        if preprocessed_images.max() > 1.0:
            preprocessed_images = preprocessed_images / 255.0
            
    elif model_type.lower() == 'tensorflow':
        # TensorFlow 모델은 일반적으로 [-1, 1] 범위 예상
        if preprocessed_images.dtype != np.float32:
            preprocessed_images = preprocessed_images.astype(np.float32)
            
        # [0, 1] -> [-1, 1]로 변환 (이미 [0, 1] 범위인 경우)
        if preprocessed_images.max() <= 1.0 and preprocessed_images.min() >= 0:
            preprocessed_images = preprocessed_images * 2.0 - 1.0
        elif preprocessed_images.max() > 1.0:  # [0, 255] -> [-1, 1]
            preprocessed_images = preprocessed_images / 127.5 - 1.0
    
    elif model_type.lower() == 'pytorch':
        # PyTorch 모델은 일반적으로 채널 우선 순서 (C, H, W) 예상
        if len(preprocessed_images.shape) == 4:
            # NHWC -> NCHW
            preprocessed_images = np.transpose(preprocessed_images, (0, 3, 1, 2))
            logger.info(f"PyTorch 포맷으로 변환: {preprocessed_images.shape} (NCHW)")
    
    logger.info(f"데이터 전처리 완료: {model_type} 모델용")
    return preprocessed_images 