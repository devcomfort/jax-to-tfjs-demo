"""
모델 학습 관련 공통 유틸리티 함수

JAX 및 FLAX 모델을 학습하기 위한 공통 유틸리티 함수를 제공합니다.
"""

from typing import Dict, Any, Optional, Union
from flax.training import train_state

from ...train import JAXTrainer, JAXEvaluator, FlaxTrainer, FlaxEvaluator


def train_and_evaluate_jax(
    model,
    num_epochs: int = 5,
    learning_rate: float = 0.001,
    evaluate_model: bool = True,
    subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    JAX 모델 학습 및 평가 실행 함수

    Args:
        model: 학습할 JAX 모델 인스턴스
        num_epochs: 학습 에포크 수 (기본값: 5)
        learning_rate: 학습률 (기본값: 0.001)
        evaluate_model: 학습 후 평가 수행 여부 (기본값: True)
        subdir: 체크포인트 저장 하위 디렉토리 (기본값: None)

    Returns:
        학습된 모델 파라미터
    """
    # 모델 초기화
    if model.params is None:
        model.init_params()

    # 학습 수행
    trainer = JAXTrainer(model, learning_rate)
    training_state = trainer.train(num_epochs, subdir)

    # 학습 완료 후 전체 평가 수행
    if evaluate_model:
        print("\n" + "=" * 50)
        print("모델 최종 평가 중...")
        evaluator = JAXEvaluator(model)
        evaluator.evaluate(training_state.params)

    return training_state.params


def train_and_evaluate_flax(
    model_manager,
    num_epochs: int = 5,
    learning_rate: float = 0.001,
    evaluate_model: bool = True,
    subdir: Optional[str] = None,
) -> train_state.TrainState:
    """
    FLAX 모델 학습 및 평가 실행 함수

    Args:
        model_manager: 모델 매니저 인스턴스
        num_epochs: 학습 에포크 수 (기본값: 5)
        learning_rate: 학습률 (기본값: 0.001)
        evaluate_model: 학습 후 평가 수행 여부 (기본값: True)
        subdir: 체크포인트 저장 하위 디렉토리 (기본값: None)

    Returns:
        학습된 모델의 학습 상태
    """
    # 모델 초기화
    if model_manager.variables is None:
        model_manager.init_model()

    # 학습 수행
    trainer = FlaxTrainer(model_manager, learning_rate)
    training_state = trainer.train(num_epochs, subdir)

    # 학습 완료 후 전체 평가 수행
    if evaluate_model:
        print("\n" + "=" * 50)
        print("모델 최종 평가 중...")
        evaluator = FlaxEvaluator(model_manager)
        evaluator.evaluate(training_state.train_state)

    return training_state.train_state


def train_and_evaluate(
    model_or_manager,
    num_epochs: int = 5,
    learning_rate: float = 0.001,
    evaluate_model: bool = True,
    subdir: Optional[str] = None,
) -> Union[Dict[str, Any], train_state.TrainState]:
    """
    모델 학습 및 평가 실행 함수

    모델 타입에 따라 적절한 학습 함수를 호출합니다.

    Args:
        model_or_manager: JAX 모델 또는 FLAX 모델 매니저 인스턴스
        num_epochs: 학습 에포크 수 (기본값: 5)
        learning_rate: 학습률 (기본값: 0.001)
        evaluate_model: 학습 후 평가 수행 여부 (기본값: True)
        subdir: 체크포인트 저장 하위 디렉토리 (기본값: None)

    Returns:
        학습된 모델 파라미터 또는 학습 상태
    """
    # 모델 타입 확인
    from ...models.jax.cnn_model import CNNModel
    from ...models.flax.model_manager import FlaxModelManager

    if isinstance(model_or_manager, CNNModel):
        return train_and_evaluate_jax(
            model_or_manager, num_epochs, learning_rate, evaluate_model, subdir
        )
    elif isinstance(model_or_manager, FlaxModelManager):
        return train_and_evaluate_flax(
            model_or_manager, num_epochs, learning_rate, evaluate_model, subdir
        )
    else:
        raise TypeError(
            f"지원되지 않는 모델 타입: {type(model_or_manager)}. CNNModel 또는 FlaxModelManager 인스턴스가 필요합니다."
        )
