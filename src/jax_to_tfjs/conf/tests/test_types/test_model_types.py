"""
모델 관련 타입 모듈 테스트

모델 파라미터, 상태, 함수 시그니처 등의 타입을 테스트합니다.
"""

import unittest
import numpy as np

# 기본 모델 타입 임포트
from jax_to_tfjs.conf.types.model_types import (
    BatchData,
    ModelProtocol,
    TrainState,
    LogData,
)

# 추가 필요한 타입 임포트 - 실제 정의된 위치에서 가져옴
from jax_to_tfjs.conf.config.training import ModelType
from jax_to_tfjs.utils.checkpoint_schema import ModelFramework, ModelConfig


class SimpleModel:
    """ModelProtocol을 구현한 간단한 클래스"""

    def apply(self, params, x):
        return x

    def init_params(self):
        return {"weights": np.ones((10, 10))}


class TestModelTypes(unittest.TestCase):
    """모델 타입 모듈 테스트"""

    def test_batch_data_type(self):
        """BatchData 타입 테스트"""
        # BatchData는 TypedDict이므로 일반 딕셔너리처럼 사용
        batch = BatchData(image=np.zeros((32, 28, 28, 1)), label=np.zeros((32, 10)))

        self.assertIn("image", batch)
        self.assertIn("label", batch)
        self.assertEqual(batch["image"].shape, (32, 28, 28, 1))
        self.assertEqual(batch["label"].shape, (32, 10))

    def test_model_protocol(self):
        """ModelProtocol 테스트"""
        # 프로토콜을 구현한 클래스
        model = SimpleModel()
        self.assertIsInstance(model, ModelProtocol)

        # 프로토콜을 구현하지 않은 클래스
        class NotModel:
            pass

        not_model = NotModel()
        self.assertNotIsInstance(not_model, ModelProtocol)

        # 일부만 구현한 클래스
        class PartialModel:
            def apply(self, params, x):
                return x

        partial_model = PartialModel()
        self.assertNotIsInstance(partial_model, ModelProtocol)

    def test_train_state(self):
        """TrainState 테스트"""
        # NamedTuple 생성
        state = TrainState(
            step=10,
            epoch=2,
            params={"layer1": np.ones((5, 5))},
            opt_state={"momentum": np.zeros((5, 5))},
            loss=0.1,
        )

        # 속성 접근
        self.assertEqual(state.step, 10)
        self.assertEqual(state.epoch, 2)
        self.assertEqual(state.loss, 0.1)
        self.assertIn("layer1", state.params)
        self.assertIn("momentum", state.opt_state)

        # 불변성 테스트 - runtime에서 확인
        try:
            print("\n[디버깅] NamedTuple 불변성 테스트")
            print(f"변경 전 state.step = {state.step}")
            state.step = 11  # TypeError 발생해야 함
            print(f"변경 후 state.step = {state.step}")  # 이 부분은 실행되지 않아야 함
        except AttributeError as e:
            print(f"정상적인 오류 발생: {e}")

    def test_log_data(self):
        """LogData 테스트"""
        # LogData는 TypedDict이며 total=False로 선언되어 모든 키가 선택적
        print("\n[디버깅] LogData 테스트")

        # 기본 로그 데이터
        log1 = LogData(epoch=1, step=100, train_loss=0.5, learning_rate=0.001)
        print(f"log1 내용: {log1}")

        # get() 메서드 사용
        self.assertEqual(log1.get("epoch"), 1)
        self.assertEqual(log1.get("step"), 100)
        self.assertEqual(log1.get("train_loss"), 0.5)
        self.assertEqual(log1.get("learning_rate"), 0.001)

        # 존재하지 않는 키는 None 반환
        self.assertIsNone(log1.get("val_loss"))

        # 확장된 로그 데이터
        log2 = LogData(
            epoch=2,
            step=200,
            train_loss=0.3,
            val_loss=0.4,
            accuracy=0.85,
            learning_rate=0.001,
            checkpoint_path="models/checkpoint.h5",
            elapsed_time=120.5,
        )
        print(f"log2 내용: {log2}")

        # get() 메서드 사용
        self.assertEqual(log2.get("epoch"), 2)
        self.assertEqual(log2.get("val_loss"), 0.4)
        self.assertEqual(log2.get("accuracy"), 0.85)
        self.assertEqual(log2.get("checkpoint_path"), "models/checkpoint.h5")
        self.assertEqual(log2.get("elapsed_time"), 120.5)

    def test_model_type_enum(self):
        """ModelType Enum 테스트"""
        # 기본 열거형 값 확인
        self.assertIsInstance(ModelType.JAX, ModelType)
        self.assertIsInstance(ModelType.FLAX, ModelType)

        # 열거형 값들은 문자열이어야 함
        for model_type in ModelType:
            self.assertIsInstance(model_type.value, str)

    def test_model_framework_enum(self):
        """ModelFramework Enum 테스트"""
        # 기본 열거형 값 확인
        self.assertIsInstance(ModelFramework.JAX, ModelFramework)
        self.assertIsInstance(ModelFramework.TENSORFLOW, ModelFramework)
        self.assertIsInstance(ModelFramework.PYTORCH, ModelFramework)

        # 열거형 값들은 문자열이어야 함
        for framework in ModelFramework:
            self.assertIsInstance(framework.value, str)

    def test_model_config(self):
        """ModelConfig 클래스 테스트"""
        # 기본 생성
        model_config = ModelConfig(name="test_model")

        # 타입 검사
        self.assertIsInstance(model_config.name, str)
        self.assertIsInstance(model_config.framework, ModelFramework)

        # 기본값 검증
        self.assertEqual(model_config.name, "test_model")
        self.assertEqual(model_config.version, "1.0.0")
        self.assertEqual(model_config.framework, ModelFramework.UNKNOWN)

        # 커스텀 값으로 생성
        test_config = ModelConfig(
            name="custom_model",
            version="2.0.0",
            framework=ModelFramework.JAX,
            input_shape=[28, 28, 1],
            output_shape=[10],
            additional_params={"optimizer": "adam"},
        )

        # 값 검증
        self.assertEqual(test_config.name, "custom_model")
        self.assertEqual(test_config.version, "2.0.0")
        self.assertEqual(test_config.framework, ModelFramework.JAX)
        self.assertEqual(test_config.input_shape, [28, 28, 1])
        self.assertEqual(test_config.output_shape, [10])
        self.assertEqual(test_config.additional_params["optimizer"], "adam")


if __name__ == "__main__":
    unittest.main()
