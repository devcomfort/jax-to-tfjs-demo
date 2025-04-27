"""
FLAX 모델 트레이너 구현

FLAX 모델의 학습을 위한 구체적인 트레이너 구현을 제공합니다.
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from datetime import datetime
import os
import orbax.checkpoint as ocp
from typing import Any, Dict, Tuple, NamedTuple, Optional, Callable
from tqdm import tqdm

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
        self, num_epochs: int = 5, subdir: Optional[str] = None
    ) -> FlaxTrainingState:
        """
        모델 학습 수행

        Args:
            num_epochs: 학습 에포크 수 (기본값: 5)
            subdir: 체크포인트 저장 하위 디렉토리 (기본값: None)

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

            # 진행 표시를 위한 tqdm 사용 (반복자의 크기를 알 수 없으므로 total은 지정하지 않음)
            for batch_idx, batch in enumerate(tqdm(train_data, desc="학습 중")):
                state, loss = self.update_step(state, batch)
                # JAX 값을 Python float로 변환 (jit 외부)
                loss_value = float(loss)
                train_loss += loss_value
                train_batches += 1
                current_step += 1

            # 평균 손실 계산
            if train_batches > 0:
                train_loss /= train_batches
            print(f"\n에포크 평균 학습 손실: {train_loss:.6f}")

            # 테스트 데이터로 손실 계산
            test_loss = 0.0
            test_batches = 0

            # 매 에포크마다 새로운 테스트 이터레이터 생성
            _, test_data = MNISTDataLoader.load_mnist()

            for batch_idx, batch in enumerate(tqdm(test_data, desc="평가 중")):
                logits = self.model_manager.model.apply(
                    {"params": state.params}, batch["image"]
                )
                batch_loss = self._compute_loss(logits, batch["label"])
                # JAX 값을 Python float로 변환 (jit 외부)
                batch_loss_float = float(batch_loss)
                test_loss += batch_loss_float
                test_batches += 1

            # 평균 손실 계산
            if test_batches > 0:
                test_loss /= test_batches
            print(f"\n에포크 평균 테스트 손실: {test_loss:.6f}")

            # 체크포인트 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 체크포인트 저장
            checkpoint_manager.save(epoch + 1, {"model": state})
            print(f"Checkpoint saved: {checkpoint_dir / str(epoch + 1) / 'model'}")

        # 최종 손실을 float으로 변환하여 반환
        final_loss = float(test_loss)
        return FlaxTrainingState(
            train_state=state, step=current_step, epoch=num_epochs, loss=final_loss
        )
