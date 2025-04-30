"""
FLAX 모델 평가자 구현

FLAX 모델의 평가를 위한 구체적인 평가자 구현을 제공합니다.
"""

import jax.numpy as jnp
from flax.training import train_state
import numpy as np
from typing import Tuple
from tqdm import tqdm

from .base_evaluator import BaseEvaluator
from .data_loader import MNISTDataLoader


class FlaxEvaluator(BaseEvaluator):
    """
    FLAX 모델 평가 클래스

    FLAX 모델의 평가, 메트릭 계산, 예측 등의 기능을 제공합니다.
    """

    def __init__(self, model_manager):
        """
        모델 평가 클래스 초기화

        Args:
            model_manager: 평가할 FLAX 모델 매니저 인스턴스
        """
        self.model_manager = model_manager

    def evaluate(self, state: train_state.TrainState) -> Tuple[float, jnp.ndarray]:
        """
        MNIST 테스트 세트에 대한 평가

        Args:
            state: 학습 상태

        Returns:
            accuracy, predictions: 정확도와 예측값
        """
        # 테스트 데이터셋 로드
        test_images, test_labels = MNISTDataLoader.load_mnist_test()
        print(f"테스트 데이터셋 로드 완료: {test_images.shape[0]} 샘플")

        # 배치 크기
        batch_size = 100
        num_samples = test_images.shape[0]
        num_batches = num_samples // batch_size

        correct = 0
        all_predictions = []

        for i in tqdm(range(num_batches), desc="모델 평가"):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_images = test_images[start_idx:end_idx]
            batch_labels = test_labels[start_idx:end_idx]

            # 예측
            logits = self.model_manager.model.apply(
                {"params": state.params}, batch_images
            )
            predictions = jnp.argmax(logits, axis=1)

            # 정확도 계산
            correct += jnp.sum(predictions == batch_labels)
            all_predictions.append(predictions)

        accuracy = float(correct) / num_samples
        predictions = jnp.concatenate(all_predictions)

        # 결과 출력
        print(f"테스트 정확도: {accuracy:.4f} ({int(correct)}/{num_samples})")

        # 클래스별 정확도, 혼동 행렬 등 상세 메트릭 계산 및 출력
        self._print_detailed_metrics(predictions, test_labels)

        return accuracy, predictions

    def _print_detailed_metrics(self, predictions: jnp.ndarray, labels: jnp.ndarray):
        """
        상세 메트릭 계산 및 출력

        Args:
            predictions: 모델 예측값
            labels: 정답 레이블
        """
        # 클래스별 정확도 계산
        class_correct = np.zeros(10)
        class_total = np.zeros(10)

        num_samples = len(labels)
        for i in range(num_samples):
            label = int(labels[i])
            class_total[label] += 1
            if predictions[i] == labels[i]:
                class_correct[label] += 1

        print("\n클래스별 정확도:")
        for i in range(10):
            if class_total[i] > 0:
                class_accuracy = class_correct[i] / class_total[i]
                print(
                    f"클래스 {i}: {class_accuracy:.4f} ({int(class_correct[i])}/{int(class_total[i])})"
                )

        # 혼동 행렬 계산
        confusion_matrix = np.zeros((10, 10), dtype=np.int32)
        for i in range(num_samples):
            confusion_matrix[int(labels[i]), int(predictions[i])] += 1

        print("\n혼동 행렬:")
        print(confusion_matrix)
