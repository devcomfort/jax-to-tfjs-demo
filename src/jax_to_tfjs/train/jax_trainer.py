"""
JAX 모델 트레이너 구현

JAX 모델의 학습을 위한 구체적인 트레이너 구현을 제공합니다.
"""

import jax
import jax.numpy as jnp
import optax
import os
import orbax.checkpoint as ocp
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

    params: Params
    opt_state: optax.OptState
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
        self, num_epochs: int = 5, subdir: Optional[str] = None
    ) -> JAXTrainingState:
        """
        모델 학습 및 평가

        Args:
            num_epochs: 학습 에포크 수 (기본값: 5)
            subdir: 체크포인트 저장 하위 디렉토리 (기본값: None)

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
            checkpoint_manager.save(
                epoch + 1, {"model": {"params": params, "opt_state": opt_state}}
            )
            print(f"Checkpoint saved: {checkpoint_dir / str(epoch + 1) / 'model'}")

        self.model.params = params  # 학습된 파라미터 업데이트

        return JAXTrainingState(
            params=params,
            opt_state=opt_state,
            step=current_step,
            epoch=num_epochs,
            loss=final_test_loss,
        )
