"""
FLAX 모델 매니저 구현

FLAX 모델의 초기화와 체크포인트 관리를 담당하는 클래스를 정의합니다.
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import orbax.checkpoint as ocp
from typing import Dict, Any, Sequence, Optional
import os

from ...conf.path import Path
from .cnn_model import CNN


class FlaxModelManager:
    """
    FLAX 모델 관리 클래스

    모델 초기화, 로드, 저장 등의 기능을 제공합니다.
    """

    def __init__(self, rng: jax.random.PRNGKey = None):
        """
        모델 관리자 초기화

        Args:
            rng: JAX 난수 생성을 위한 키 (기본값: None, 이 경우 자동 생성)
        """
        self.rng = rng if rng is not None else jax.random.PRNGKey(0)
        self.model = CNN()
        self.variables = None
        self.path_manager = Path()

    def init_model(self, input_shape: Sequence[int] = (1, 28, 28, 1)) -> Dict[str, Any]:
        """
        모델 변수 초기화

        Args:
            input_shape: 입력 텐서 형태 (기본값: (1, 28, 28, 1))

        Returns:
            초기화된 모델 변수
        """
        # 더미 입력으로 모델 초기화
        self.rng, init_rng = jax.random.split(self.rng)
        dummy_input = jnp.ones(input_shape)
        self.variables = self.model.init(init_rng, dummy_input)

        return self.variables

    def get_params(self) -> Dict[str, Any]:
        """
        모델 파라미터 반환

        Returns:
            모델 파라미터
        """
        if self.variables is None:
            self.init_model()

        return self.variables["params"]

    def create_train_state(
        self, learning_rate: float = 0.001
    ) -> train_state.TrainState:
        """
        학습 상태 생성

        Args:
            learning_rate: 학습률 (기본값: 0.001)

        Returns:
            초기화된 학습 상태
        """
        if self.variables is None:
            self.init_model()

        # 옵티마이저 생성
        tx = optax.adam(learning_rate)

        # 학습 상태 초기화
        return train_state.TrainState.create(
            apply_fn=self.model.apply, params=self.variables["params"], tx=tx
        )

    @staticmethod
    def load_checkpoint(
        checkpoint_dir: Optional[str] = None, step: int = None
    ) -> Dict[str, Any]:
        """
        체크포인트 로드

        Args:
            checkpoint_dir: 체크포인트 디렉토리 경로 (None이면 기본 경로 사용)
            step: 로드할 체크포인트 스텝 (None이면 가장 최근 체크포인트)

        Returns:
            로드된 학습 상태
        """
        # 체크포인트 경로 설정
        if checkpoint_dir is None:
            path_manager = Path()
            checkpoint_dir = path_manager.flax_checkpoint_dir

        print(f"FLAX 모델 체크포인트 로드 중: {checkpoint_dir}")

        # Orbax 체크포인트 구조 확인
        model_dir = os.path.join(checkpoint_dir, "model")
        if os.path.isdir(model_dir):
            # model 디렉토리가 있는 경우 해당 디렉토리를 체크포인트 경로로 사용
            checkpoint_dir = model_dir

        checkpointer = ocp.PyTreeCheckpointer()

        try:
            # Orbax API 사용
            options = ocp.CheckpointManagerOptions(max_to_keep=3)
            checkpoint_manager = ocp.CheckpointManager(
                directory=str(checkpoint_dir),
                checkpointers={"model": checkpointer},
                options=options,
            )

            if step is None:
                step = checkpoint_manager.latest_step()
                if step is None:
                    # 모델 디렉토리 자체가 체크포인트인 경우
                    step = 0

            if step == 0:
                # 모델 디렉토리 자체가 체크포인트인 경우 직접 로드
                try:
                    checkpoint = checkpointer.restore(checkpoint_dir)
                    print(
                        f"모델 디렉토리로부터 체크포인트를 직접 로드했습니다: {checkpoint_dir}"
                    )
                    return checkpoint
                except Exception as e:
                    print(f"체크포인트 직접 로드 실패: {e}")
                    raise ValueError(f"No checkpoint found in {checkpoint_dir}")
            else:
                # 체크포인트 복원
                checkpoint = checkpoint_manager.restore(step, items={"model": None})[
                    "model"
                ]
                print(f"Loaded checkpoint from step {step} in {checkpoint_dir}")
                return checkpoint

        except Exception as e:
            print(f"체크포인트 로드 오류: {e}")

            # 대체 방법 시도: 직접 체크포인트 로드
            try:
                checkpoint = checkpointer.restore(checkpoint_dir)
                print(f"대체 방법으로 체크포인트를 로드했습니다: {checkpoint_dir}")
                return checkpoint
            except Exception as e2:
                print(f"대체 체크포인트 로드 실패: {e2}")
                raise ValueError(f"No checkpoint found in {checkpoint_dir}")
