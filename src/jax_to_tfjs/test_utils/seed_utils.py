"""
테스트 결정성 유틸리티

테스트의 재현 가능성과 안정적인 실행을 위한 유틸리티 함수들을 제공합니다.
"""

import os
import random
import numpy as np
import tensorflow as tf
import jax


def set_deterministic_mode(seed=42):
    """
    테스트의 결정적 실행을 위한 모든 난수 생성기 시드 설정

    Args:
        seed: 시드 값 (기본값: 42)

    Returns:
        JAX PRNGKey: JAX 난수 생성을 위한 키
    """
    # Python 기본 난수 생성기
    random.seed(seed)

    # NumPy 난수 생성기
    np.random.seed(seed)

    # TensorFlow 난수 생성기
    tf.random.set_seed(seed)

    # TensorFlow 연산 결정성 활성화 (TF 2.8 이상)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        # 구 버전 TensorFlow에서는 이 옵션 사용 불가
        print(
            "경고: TensorFlow 버전이 낮아 enable_op_determinism을 적용할 수 없습니다."
        )

    # JAX 초기화 시드
    return jax.random.PRNGKey(seed)


def prepare_deterministic_dataset(dataset, batch_size=32, is_training=True, seed=42):
    """
    결정적 데이터셋 준비

    Args:
        dataset: TensorFlow 데이터셋
        batch_size: 배치 크기
        is_training: 학습용 데이터셋 여부
        seed: 시드 값

    Returns:
        준비된 데이터셋
    """
    # 데이터셋 준비 (셔플링 없음)
    if is_training:
        dataset = dataset.cache()
        # 테스트에서는 셔플링 비활성화하거나 고정 시드 사용
        dataset = dataset.shuffle(10000, seed=seed)

    # 배치 처리 및 프리페치
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def setup_test_environment():
    """
    테스트 시작 시 호출하여 결정적 테스트 환경 설정

    Returns:
        JAX PRNGKey: JAX 난수 생성을 위한 키
    """
    # 환경 변수로 시드 설정 가능
    seed = int(os.environ.get("TEST_SEED", "42"))
    return set_deterministic_mode(seed)
