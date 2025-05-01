"""
공통 학습 유틸리티 모듈

JAX 및 FLAX 모델에서 공통으로 사용하는 학습 및 평가 유틸리티 함수들을 제공합니다.
"""

from typing import Any, Optional, Dict, Union

from ...train.jax_trainer import JAXTrainer
from ...train.flax_trainer import FlaxTrainer
# 순환 임포트 문제 해결을 위해 제거
# from ...evaluation.models.jax_evaluator import evaluate_jax_model
# from ...evaluation.models.flax_evaluator import evaluate_flax_model


def train_and_evaluate(
    model_or_manager: Any,
    num_epochs: int = 5,
    learning_rate: float = 0.001,
    evaluate_model: bool = True,
    subdir: Optional[str] = None,
) -> Union[Dict[str, Any], Any]:
    """
    공통 학습 및 평가 함수

    JAX 또는 FLAX 모델을 학습하고 평가하는 공통 함수입니다.
    model_or_manager 인자에 따라 적절한 트레이너와 평가기를 선택합니다.

    Args:
        model_or_manager: JAX 모델 또는 FLAX 모델 매니저
        num_epochs: 학습 에포크 수 (기본값: 5)
        learning_rate: 학습률 (기본값: 0.001)
        evaluate_model: 학습 후 평가 수행 여부 (기본값: True)
        subdir: 체크포인트 저장 하위 디렉토리 (기본값: None)

    Returns:
        JAX 모델의 경우 학습된 파라미터, FLAX 모델의 경우 학습 상태
    """
    # 모델 타입 확인 (JAX 또는 FLAX)
    is_jax = hasattr(model_or_manager, "forward")

    # 적절한 트레이너 선택
    if is_jax:
        # JAX 모델 트레이너
        trainer = JAXTrainer(model_or_manager, learning_rate=learning_rate)

        # 학습 수행
        training_state = trainer.train(num_epochs=num_epochs, subdir=subdir)

        # 평가 수행 (선택적)
        if evaluate_model:
            # 테스트 데이터 로드
            from ...train.data_loader import MNISTDataLoader

            test_images, test_labels = MNISTDataLoader.load_mnist_test()

            # 동적으로 평가 함수 임포트
            from ...evaluation.models.jax_evaluator import evaluate_jax_model

            # 상세 평가 수행
            metrics, _, _ = evaluate_jax_model(
                training_state.params, test_images, test_labels, with_probs=True
            )
            print(f"평가 결과: {metrics}")

        # 학습된 파라미터 반환
        return training_state.params
    else:
        # FLAX 모델 트레이너
        trainer = FlaxTrainer(model_or_manager, learning_rate=learning_rate)

        # 학습 수행
        training_state = trainer.train(num_epochs=num_epochs, subdir=subdir)

        # 평가 수행 (선택적)
        if evaluate_model:
            # 테스트 데이터 로드
            from ...train.data_loader import MNISTDataLoader

            test_images, test_labels = MNISTDataLoader.load_mnist_test()

            # 동적으로 평가 함수 임포트
            from ...evaluation.models.flax_evaluator import evaluate_flax_model

            # 상세 평가 수행
            metrics, _, _ = evaluate_flax_model(
                training_state.train_state, test_images, test_labels, with_probs=True
            )
            print(f"평가 결과: {metrics}")

        # 학습 상태 반환
        return training_state.train_state
