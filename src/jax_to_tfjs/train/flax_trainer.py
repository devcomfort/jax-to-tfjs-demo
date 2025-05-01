"""
FLAX 모델 트레이너 구현

FLAX 모델의 학습을 위한 구체적인 트레이너 구현을 제공합니다.
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import orbax.checkpoint as ocp
from typing import (
    Any,
    Dict,
    Tuple,
    NamedTuple,
    Optional,
    Callable,
)
from tqdm import tqdm
import json

from ..conf.paths import get_flax_checkpoint_path
from .base_trainer import BaseTrainer
from .data_loader import MNISTDataLoader


class FlaxTrainingState(NamedTuple):
    """FLAX 모델 학습 상태를 저장하는 NamedTuple"""

    train_state: train_state.TrainState
    step: int
    epoch: int
    loss: float


class FlaxTrainer(BaseTrainer):
    """
    FLAX 모델 학습 클래스

    FLAX 모델의 학습, 손실 계산, 파라미터 업데이트 등의 기능을 제공합니다.
    """

    def __init__(self, model_manager, learning_rate: float = 0.001):
        """
        FLAX 모델 학습 클래스 초기화

        Args:
            model_manager: 학습할 FLAX 모델 매니저 인스턴스
            learning_rate: 학습률 (기본값: 0.001)
        """
        super().__init__(learning_rate)
        self.model_manager = model_manager
        self.update_step = self._create_update_step()

    def _compute_loss(self, logits: jnp.ndarray, labels: Any) -> jnp.ndarray:
        """
        손실 함수 계산

        Args:
            logits: 모델 출력
            labels: 정답 레이블 (Any 타입 허용)

        Returns:
            계산된 손실값
        """
        one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
        return loss.mean()

    def _create_update_step(self) -> Callable:
        """
        학습 스텝 함수 생성

        Returns:
            학습 스텝 함수
        """

        @jax.jit
        def loss_fn(params, batch):
            """
            손실 계산 함수

            Args:
                params: 모델 파라미터
                batch: 데이터 배치

            Returns:
                손실값
            """
            logits = self.model_manager.model.apply({"params": params}, batch["image"])
            return self._compute_loss(logits, batch["label"])

        @jax.jit
        def update_step(
            state: train_state.TrainState, batch: Dict[str, jnp.ndarray]
        ) -> Tuple[train_state.TrainState, jnp.ndarray]:
            """
            모델 파라미터 업데이트 단계

            Args:
                state: 현재 학습 상태
                batch: 데이터 배치

            Returns:
                업데이트된 학습 상태와 손실값
            """
            # 손실 값과 그라디언트 계산
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params, batch)

            # 그라디언트로 파라미터 업데이트
            state = state.apply_gradients(grads=grads)

            return state, loss

        return update_step

    def train(
        self,
        num_epochs: int = 5,
        subdir: Optional[str] = None,
        evaluate_after_training: bool = True,
    ) -> FlaxTrainingState:
        """
        모델 학습 수행

        Args:
            num_epochs: 학습 에포크 수 (기본값: 5)
            subdir: 체크포인트 저장 하위 디렉토리 (기본값: None)
            evaluate_after_training: 학습 후 상세 평가 수행 여부 (기본값: True)

        Returns:
            최종 학습 상태
        """
        # 학습 상태 초기화
        state = self.model_manager.create_train_state(self.learning_rate)

        # 체크포인트 경로 설정
        checkpoint_dir = get_flax_checkpoint_path(subdir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Orbax 체크포인트 매니저 생성
        checkpointer = ocp.PyTreeCheckpointer()
        options = ocp.CheckpointManagerOptions(max_to_keep=3)
        checkpoint_manager = ocp.CheckpointManager(
            directory=str(checkpoint_dir),
            checkpointers={"model": checkpointer},
            options=options,
        )

        current_step = 0
        test_loss = 0.0  # 기본값 초기화

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("=" * 50)

            # 학습
            train_loss = 0
            train_batches = 0

            # 매 에포크마다 새로운 이터레이터 생성
            train_data, test_data = MNISTDataLoader.load_mnist()

            # MNIST 훈련 데이터 크기 60,000개를 배치 크기 32로 나누면 약 1875 배치
            estimated_train_batches = 1875

            # 진행 표시를 위한 tqdm 사용 (JAX 트레이너와 동일한 형식으로 통일)
            progress_bar = tqdm(
                total=estimated_train_batches,
                desc="Training",
                bar_format="{l_bar}{bar:30}{r_bar}",
                ascii=True,  # ASCII 문자만 사용하여 일관된 표시
            )

            try:
                for batch_idx in range(estimated_train_batches):
                    try:
                        batch = next(train_data)
                        state, loss = self.update_step(state, batch)

                        # JAX 값을 Python float로 변환 (jit 외부)
                        loss_value = float(loss)
                        train_loss += loss_value
                        train_batches += 1
                        current_step += 1

                        # 진행 상태 업데이트
                        progress_bar.update(1)
                        # loss 형식을 .6f로 변경하여 일정한 폭으로 표시
                        loss_formatted = f"{loss_value:.6f}"
                        progress_bar.set_postfix({"loss": loss_formatted})
                    except StopIteration:
                        break
            finally:
                progress_bar.close()

            # 평균 손실 계산
            if train_batches > 0:
                train_loss /= train_batches
            print(f"Training Loss: {train_loss:.6f}")

            # 테스트 데이터로 손실 계산
            test_loss = 0.0
            test_batches = 0

            # 매 에포크마다 새로운 테스트 이터레이터 생성
            _, test_data = MNISTDataLoader.load_mnist()

            # MNIST 테스트 데이터 10,000개를 배치 크기 32로 나누면 약 312 배치
            estimated_test_batches = 312

            # 테스트 진행 표시를 위한 tqdm 사용
            progress_bar = tqdm(
                total=estimated_test_batches,
                desc="Evaluating",
                bar_format="{l_bar}{bar:30}{r_bar}",
                ascii=True,  # ASCII 문자만 사용하여 일관된 표시
            )

            try:
                for batch_idx in range(estimated_test_batches):
                    try:
                        batch = next(test_data)
                        logits = self.model_manager.model.apply(
                            {"params": state.params}, batch["image"]
                        )
                        batch_loss = self._compute_loss(logits, batch["label"])

                        # JAX 값을 Python float로 변환 (jit 외부)
                        batch_loss_float = float(batch_loss)
                        test_loss += batch_loss_float
                        test_batches += 1

                        # 진행 상태 업데이트
                        progress_bar.update(1)
                        # loss 형식을 .6f로 변경하여 일정한 폭으로 표시
                        loss_formatted = f"{batch_loss_float:.6f}"
                        progress_bar.set_postfix({"loss": loss_formatted})
                    except StopIteration:
                        break
            finally:
                progress_bar.close()

            # 평균 손실 계산
            if test_batches > 0:
                test_loss /= test_batches
            print(f"Test Loss: {test_loss:.6f}")

            # 체크포인트 저장
            checkpoint_manager.save(epoch + 1, {"model": state})
            print(f"Checkpoint saved: {checkpoint_dir / str(epoch + 1) / 'model'}")

        # 최종 손실을 float으로 변환하여 반환
        final_loss = float(test_loss)

        # 학습 완료 후 상세 평가 수행
        if evaluate_after_training:
            print("\n" + "=" * 50)
            print("학습 완료 후 상세 평가 수행")
            print("=" * 50)

            # 테스트 데이터 로드
            test_images, test_labels = MNISTDataLoader.load_mnist_test()

            # evaluation 모듈에서 평가 함수 가져오기
            from ..evaluation.models.flax_evaluator import evaluate_flax_model

            # 상세 평가 수행 - 결과를 분리
            result = evaluate_flax_model(  # type: ignore
                state, test_images, test_labels, with_probs=True
            )

            # 결과 분리
            metrics_obj = result[0]
            predictions = result[1]

            # 딕셔너리로 변환 (타입 체커 우회)
            metrics = {}
            for field in ["accuracy", "precision", "recall", "f1"]:
                try:
                    # 객체 속성으로 접근
                    metrics[field] = getattr(metrics_obj, field)  # type: ignore
                except AttributeError:
                    # 딕셔너리로 접근하거나 기본값 사용
                    try:
                        if isinstance(metrics_obj, dict):
                            metrics[field] = metrics_obj.get(field, 0.0)  # type: ignore
                        else:
                            metrics[field] = 0.0
                    except (TypeError, ValueError):
                        metrics[field] = 0.0

            # 상세 메트릭 계산
            try:
                from sklearn.metrics import confusion_matrix

                # 혼동 행렬 계산 및 출력
                cm = confusion_matrix(test_labels, predictions)
                print("\n혼동 행렬:")
                print(cm)

                # 메트릭 출력
                print(
                    f"\n정확도(Accuracy): {metrics['accuracy']:.4f} (전체 중 올바르게 분류한 비율)"
                )
                print(
                    f"정밀도(Precision): {metrics['precision']:.4f} (양성으로 예측한 것 중 실제 양성 비율)"
                )
                print(
                    f"재현율(Recall): {metrics['recall']:.4f} (실제 양성 중 양성으로 예측한 비율)"
                )
                print(
                    f"F1 점수: {metrics['f1']:.4f} (정밀도와 재현율의 조화평균, 두 지표의 균형)"
                )

                # 메트릭 저장
                evaluation_dir = checkpoint_dir / "evaluation"
                evaluation_dir.mkdir(exist_ok=True)

                # 혼동 행렬 추가
                metrics_dict = dict(metrics)
                metrics_dict["confusion_matrix"] = cm.tolist()

                # 메트릭을 JSON 파일로 저장
                metrics_path = evaluation_dir / "metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(metrics_dict, f, indent=2)

                print(f"\n평가 메트릭 저장 완료: {metrics_path}")

            except ImportError as e:
                print(f"상세 메트릭 계산을 위한 라이브러리를 찾을 수 없습니다: {e}")
                print(f"기본 메트릭만 사용합니다: 정확도 {metrics['accuracy']:.4f}")

        return FlaxTrainingState(
            train_state=state, step=current_step, epoch=num_epochs, loss=final_loss
        )
