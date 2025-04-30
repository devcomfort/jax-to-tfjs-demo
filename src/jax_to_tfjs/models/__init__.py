"""
MNIST 모델 모듈

이 패키지는 JAX 및 FLAX를 사용하여 구현된 MNIST 모델 관련 클래스와 함수를 제공합니다.
"""

# 모듈 가져오기
from . import jax, flax, common

__all__ = [
    # 모듈
    "jax",  # Jax 모듈
    "flax",  # Flax 모듈
    "common",  # 공통 모듈
    # Jax
    "JaxCNNModel",  # Jax CNN 모델
    "JaxMNISTDataLoader",  # Jax MNIST 데이터 로더
    "JaxModelTrainer",  # Jax 모델 트레이너
    "JaxModelEvaluator",  # Jax 모델 평가기
    "jax_train_and_evaluate",  # Jax 모델 학습 및 평가
    # Flax
    "FlaxCNN",  # Flax CNN 모델
    "FlaxModelManager",  # Flax 모델 매니저
    "FlaxModelTrainer",  # Flax 모델 트레이너
    "FlaxModelEvaluator",  # Flax 모델 평가기
    "flax_train_and_evaluate",  # Flax 모델 학습 및 평가
    "flax_load_checkpoint",  # Flax 모델 체크포인트 로드
    "jax_load_checkpoint",  # Jax 모델 체크포인트 로드
    "init_cnn_params",  # CNN 파라미터 초기화
    "cnn_forward",  # CNN 순방향 전파
]

# train 모듈에서 클래스들 가져오기
from ..train.data_loader import MNISTDataLoader as JaxMNISTDataLoader
from ..train.jax_trainer import JAXTrainer as JaxModelTrainer
from ..train.jax_evaluator import JAXEvaluator as JaxModelEvaluator
from ..train.flax_trainer import FlaxTrainer as FlaxModelTrainer
from ..train.flax_evaluator import FlaxEvaluator as FlaxModelEvaluator

# JAX 모듈 인터페이스
from .jax.cnn_model import CNNModel as JaxCNNModel
from .jax.utils import cnn_forward, init_cnn_params
from .common.train_utils import train_and_evaluate as jax_train_and_evaluate

# Flax 모듈 인터페이스
from .flax.cnn_model import CNN as FlaxCNN
from .flax.model_manager import FlaxModelManager
from .flax.utils import create_train_state
from .common.train_utils import train_and_evaluate as flax_train_and_evaluate

# 이전 코드와의 호환성을 위해 expose하는 함수들
from .flax.utils import load_checkpoint as flax_load_checkpoint
from .jax.utils import load_checkpoint as jax_load_checkpoint

from .jax import CNNModel, JAXModelManager
from .flax import CNN, FlaxModelManager

__all__ = [
    # JAX 모델들
    "CNNModel",
    "JAXModelManager",
    # FLAX 모델들
    "CNN",
    "FlaxModelManager",
]
