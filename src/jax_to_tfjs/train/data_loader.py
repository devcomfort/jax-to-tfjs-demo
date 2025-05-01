"""
데이터 로딩 공통 모듈

데이터셋 로딩과 관련된 공통 기능을 제공합니다.
"""

import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Tuple, Dict, Iterator, cast


class BaseDataLoader:
    """
    데이터 로딩 기본 클래스

    데이터 로딩과 배치 처리 기능을 제공합니다.
    """

    @staticmethod
    def normalize_image(image, label):
        """
        이미지 정규화 함수

        Args:
            image: 입력 이미지
            label: 레이블

        Returns:
            정규화된 이미지와 레이블
        """
        # TensorFlow의 나눗셈 함수 사용
        return {
            "image": tf.math.divide(tf.cast(image, tf.float32), 255.0),
            "label": label,
        }

    @classmethod
    def prepare_dataset(
        cls, dataset, batch_size: int = 32, is_training: bool = True
    ) -> tf.data.Dataset:
        """
        데이터셋 전처리 및 배치 처리

        Args:
            dataset: 원본 데이터셋
            batch_size: 배치 크기
            is_training: 학습용 데이터셋 여부

        Returns:
            전처리된 데이터셋
        """
        # 데이터 전처리
        dataset = dataset.map(cls.normalize_image, num_parallel_calls=tf.data.AUTOTUNE)

        # 학습 데이터셋인 경우 셔플
        if is_training:
            dataset = dataset.cache()
            dataset = dataset.shuffle(10000)

        # 배치 처리 및 프리페치
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


class MNISTDataLoader(BaseDataLoader):
    """
    MNIST 데이터셋 로더

    MNIST 데이터셋의 로딩과 전처리를 담당합니다.
    """

    @classmethod
    def _convert_to_numpy_iterator(
        cls, dataset: tf.data.Dataset
    ) -> Iterator[Dict[str, np.ndarray]]:
        """
        TensorFlow 데이터셋을 NumPy 배치 반복자로 변환

        Args:
            dataset: TensorFlow 데이터셋

        Returns:
            NumPy 배치 반복자
        """
        for batch in dataset:
            # TensorFlow Eager Tensor를 NumPy 배열로 변환
            batch_dict = cast(Dict[str, tf.Tensor], batch)
            yield {
                "image": np.array(batch_dict["image"]),
                "label": np.array(batch_dict["label"]),
            }

    @classmethod
    def load_mnist(
        cls, batch_size: int = 32
    ) -> Tuple[Iterator[Dict[str, np.ndarray]], Iterator[Dict[str, np.ndarray]]]:
        """
        MNIST 데이터셋 로드

        Args:
            batch_size: 배치 크기

        Returns:
            train_dataset, test_dataset: 학습 및 테스트 데이터셋
        """
        # MNIST 데이터셋 로드
        # 타입 주석: tfds.load의 반환값을 Dict로 처리
        datasets = tfds.load(  # type: ignore  # tfds.load는 동적 타입을 반환하며 타입 어노테이션이 불완전하여 타입 체커가 반환 형식을 정확히 추론할 수 없음
            name="mnist",
            with_info=True,
            as_supervised=True,
        )

        # 타입 안전을 위해 명시적 변수 할당
        # ObjectProxy 객체에 대한 인덱스 접근에 타입 무시 주석 추가
        mnist_dataset = datasets[0]  # type: ignore  # datasets는 런타임에는 튜플처럼 작동하지만 TensorFlow의 동적 타입 시스템으로 인해 정적 타입 체커가 이를 인식할 수 없음
        # mnist_info = datasets[1]  # type: ignore  # 사용하지 않는 변수 제거

        # 학습 및 테스트 데이터셋 분리
        # 타입 주석: ObjectProxy 타입 무시
        mnist_train = mnist_dataset["train"]  # type: ignore  # mnist_dataset은 tf.data.Dataset을 포함하는 특수 딕셔너리 같은 객체이지만 정적 타입 체커는 __getitem__ 메서드를 인식하지 못함
        mnist_test = mnist_dataset["test"]  # type: ignore  # mnist_dataset은 tf.data.Dataset을 포함하는 특수 딕셔너리 같은 객체이지만 정적 타입 체커는 __getitem__ 메서드를 인식하지 못함

        # 데이터셋 준비
        train_dataset = cls.prepare_dataset(mnist_train, batch_size, is_training=True)
        test_dataset = cls.prepare_dataset(mnist_test, batch_size, is_training=False)

        # TensorFlow 데이터셋을 NumPy 반복자로 변환
        train_iterator = cls._convert_to_numpy_iterator(train_dataset)
        test_iterator = cls._convert_to_numpy_iterator(test_dataset)

        return train_iterator, test_iterator

    @classmethod
    def load_mnist_test(cls) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        MNIST 테스트 데이터셋 로드 (넘파이 배열 형태)

        Returns:
            test_images, test_labels: 테스트 이미지와 레이블
        """
        # 테스트 데이터셋 로드
        mnist_dataset = tfds.load(name="mnist", as_supervised=True)  # type: ignore  # tfds.load는 동적 타입을 반환하며 타입 어노테이션이 불완전하여 타입 체커가 반환 형식을 정확히 추론할 수 없음

        # 타입 안전 처리
        # 타입 주석: ObjectProxy 타입 무시
        mnist_test = mnist_dataset["test"]  # type: ignore  # mnist_dataset은 내부적으로 ObjectProxy를 사용하는 특수 객체로 런타임에는 딕셔너리처럼 작동하지만 정적 타입 체커는 이를 인식할 수 없음

        # 이미지와 레이블 분리
        images = []
        labels = []

        # 데이터 정규화 및 수집
        for image, label in mnist_test:
            images.append(image.numpy() / 255.0)
            labels.append(label.numpy())

        # JAX 배열로 변환
        test_images = jnp.array(images)
        test_labels = jnp.array(labels)

        return test_images, test_labels
