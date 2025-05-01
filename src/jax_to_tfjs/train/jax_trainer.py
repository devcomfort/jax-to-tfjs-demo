"""
JAX 모델 트레이너 구현

JAX 모델의 학습을 위한 구체적인 트레이너 구현을 제공합니다.
"""

import jax
import jax.numpy as jnp
import optax
from typing import (
    Any,
    Dict,
    Tuple,
    NamedTuple,
    Optional,
    Iterator,
    Union,
    cast,
)
import numpy as np
from tqdm import tqdm
import json

from ..checkpoint_utils.jax_checkpointer import JAXCheckpointer
from ..conf.paths import get_jax_checkpoint_path
from .base_trainer import BaseTrainer
from .data_loader import MNISTDataLoader

# 타입 정의
# Params를 optax에서 사용하는 PyTree 타입과 호환되도록 재정의
PyTree = Any  # Any type that can be a PyTree leaf
Params = PyTree  # 모델 파라미터 타입 (PyTree 타입 사용)
Loss = Union[float, jnp.ndarray]


class JAXTrainingState(NamedTuple):
    """JAX 모델 학습 상태를 저장하는 NamedTuple"""

    params: Dict[str, Any]
    opt_state: Any
    step: int
    epoch: int
    loss: float


class JAXTrainer(BaseTrainer):
    """
    JAX 모델 학습 클래스

    JAX 모델의 학습, 손실 계산, 파라미터 업데이트 등의 기능을 제공합니다.
    """

    def __init__(self, model, learning_rate: float = 0.001):
        """
        JAX 모델 학습 클래스 초기화

        Args:
            model: 학습할 JAX 모델 인스턴스
            learning_rate: 학습률 (기본값: 0.001)
        """
        super().__init__(learning_rate)
        self.model = model
        self.optimizer = optax.adam(learning_rate)
        self.update_step = self._create_update_step()
        self.checkpointer = JAXCheckpointer(max_to_keep=3)

    def _compute_loss(self, logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """
        손실 함수 계산

        Args:
            logits: 모델 출력
            labels: 정답 레이블

        Returns:
            계산된 손실값
        """
        return optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        ).mean()

    def _loss_fn(
        self, params: Params, batch: Dict[str, np.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        손실 함수

        Args:
            params: 모델 파라미터
            batch: 데이터 배치

        Returns:
            loss, logits: 손실값과 모델 출력
        """
        # numpy 배열을 jax 배열로 변환
        images = jnp.array(batch["image"])
        logits = self.model.forward(params, images)
        loss = self._compute_loss(logits, jnp.array(batch["label"]))
        return loss, logits

    def _create_update_step(self):
        """
        optimizer를 클로저로 처리하는 update_step 생성

        Returns:
            update_step 함수
        """

        @jax.jit
        def update_step(
            params: Params,
            opt_state: optax.OptState,
            batch: Dict[str, np.ndarray],
        ) -> Tuple[Params, optax.OptState, jnp.ndarray]:
            """
            학습 스텝

            Args:
                params: 모델 파라미터
                opt_state: 옵티마이저 상태
                batch: 데이터 배치

            Returns:
                updated_params, updated_opt_state, loss: 업데이트된 파라미터, 옵티마이저 상태, 손실값
            """
            (loss, _), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(
                params, batch
            )
            updates, opt_state = self.optimizer.update(grads, opt_state)
            # optax.apply_updates 반환 타입을 Params 타입으로 캐스팅
            params = cast(Params, optax.apply_updates(params, updates))
            return params, opt_state, loss

        return update_step

    def _process_epoch(
        self,
        params: Params,
        opt_state: optax.OptState,
        train_data_iterator: Iterator[Dict[str, np.ndarray]],
        is_training: bool = True,
        estimated_batches: int = 1875,  # MNIST 훈련 데이터 60,000개를 배치 크기 32로 나누면 약 1875
    ) -> Tuple[Params, optax.OptState, float, int]:
        """
        한 에포크 처리

        Args:
            params: 모델 파라미터
            opt_state: 옵티마이저 상태
            train_data_iterator: 학습 데이터 반복자
            is_training: 학습 여부
            estimated_batches: 예상 배치 수

        Returns:
            params, opt_state, avg_loss, batch_count: 파라미터, 옵티마이저 상태, 평균 손실, 배치 수
        """
        total_loss = 0.0
        batch_count = 0

        # 진행 상태 표시를 위한 tqdm 사용
        desc = "Training" if is_training else "Evaluating"

        # 테스트 배치 수를 312로 고정 (MNIST 테스트 데이터 10,000개를 배치 크기 32로 나누면 약 312)
        if not is_training:
            estimated_batches = 312

        progress_bar = tqdm(
            total=estimated_batches,
            desc=desc,
            bar_format="{l_bar}{bar:30}{r_bar}",
            ascii=True,  # ASCII 문자만 사용하여 일관된 표시
        )

        try:
            while True:
                batch = next(train_data_iterator)
                batch_count += 1

                if is_training:
                    # 학습 모드
                    params, opt_state, loss = self.update_step(params, opt_state, batch)
                    total_loss += float(loss)
                else:
                    # 평가 모드
                    loss, _ = self._loss_fn(params, batch)
                    total_loss += float(loss)

                # 진행 상태 업데이트
                progress_bar.update(1)
                # loss 형식을 .6f로 변경하여 일정한 폭으로 표시
                loss_formatted = f"{float(loss):.6f}"
                progress_bar.set_postfix({"loss": loss_formatted})

                # 배치 수가 예상 배치 수를 초과하면 중단
                if batch_count >= estimated_batches:
                    break
        except StopIteration:
            # 반복자 소진
            pass
        finally:
            progress_bar.close()

        avg_loss = total_loss / max(batch_count, 1)
        return params, opt_state, avg_loss, batch_count

    def train(
        self,
        num_epochs: int = 5,
        subdir: Optional[str] = None,
        evaluate_after_training: bool = True,
    ) -> JAXTrainingState:
        """
        모델 학습 및 평가

        Args:
            num_epochs: 학습 에포크 수 (기본값: 5)
            subdir: 체크포인트 저장 하위 디렉토리 (기본값: None)
            evaluate_after_training: 학습 후 상세 평가 수행 여부 (기본값: True)

        Returns:
            최종 학습 상태 (파라미터, 옵티마이저 상태, 스텝, 에포크, 손실)
        """
        if self.model.params is None:
            self.model.init_params()

        params = self.model.params
        opt_state = self.optimizer.init(params)

        # 데이터 로드 - 반복자로 반환됨
        train_data, test_data = MNISTDataLoader.load_mnist()

        # 체크포인트 경로 설정
        checkpoint_dir = get_jax_checkpoint_path(subdir)
        # 체크포인트 디렉토리가 존재하지 않으면 생성
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        current_step = 0
        final_test_loss = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("=" * 50)

            # 학습 데이터셋 새로 로드 (반복자가 소진되었을 수 있음)
            train_data, _ = MNISTDataLoader.load_mnist()

            # 학습
            params, opt_state, train_loss, train_batches = self._process_epoch(
                params, opt_state, train_data, is_training=True
            )
            current_step += train_batches
            print(f"Training Loss: {train_loss:.6f}")

            # 테스트 데이터셋 새로 로드
            _, test_data = MNISTDataLoader.load_mnist()

            # 평가
            params, opt_state, test_loss, _ = self._process_epoch(
                params, opt_state, test_data, is_training=False, estimated_batches=312
            )
            print(f"Test Loss: {test_loss:.6f}")
            final_test_loss = test_loss

            # 체크포인트 저장
            checkpoint_state = {"params": params, "opt_state": opt_state}
            save_path = self.checkpointer.save(
                epoch + 1, checkpoint_state, checkpoint_dir
            )
            print(f"Checkpoint saved: {save_path}")

        self.model.params = params  # 학습된 파라미터 업데이트

        # 학습 완료 후 상세 평가 수행
        if evaluate_after_training:
            print("\n" + "=" * 50)
            print("학습 완료 후 상세 평가 수행")
            print("=" * 50)

            # 테스트 데이터 로드
            test_images, test_labels = MNISTDataLoader.load_mnist_test()

            # evaluation 모듈에서 평가 함수 가져오기
            from ..evaluation.models.jax_evaluator import evaluate_jax_model

            # 상세 평가 수행 - 결과를 딕셔너리로 변환
            result = evaluate_jax_model(
                params, test_images, test_labels, with_probs=True
            )

            # 결과 분리
            metrics_obj = result[0]
            predictions = result[1]

            # 딕셔너리로 변환 (타입 체커 우회)
            metrics = {}
            for field in ["accuracy", "precision", "recall", "f1"]:
                try:
                    # 객체 속성으로 접근
                    metrics[field] = getattr(metrics_obj, field)
                except:
                    # 딕셔너리로 접근하거나 기본값 사용
                    if hasattr(metrics_obj, "__getitem__"):
                        try:
                            metrics[field] = metrics_obj[field]
                        except:
                            metrics[field] = 0.0
                    else:
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

        return JAXTrainingState(
            params=params,
            opt_state=opt_state,
            step=current_step,
            epoch=num_epochs,
            loss=final_test_loss,
        )
