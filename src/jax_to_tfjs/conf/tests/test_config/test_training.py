"""
학습 설정 모듈 테스트

TrainingConfig 클래스 및 ModelType 열거형을 테스트합니다.
"""

import unittest
from jax_to_tfjs.conf.config.training import TrainingConfig, ModelType


class TestTrainingConfig(unittest.TestCase):
    """TrainingConfig 클래스 테스트"""

    def test_default_values(self):
        """기본값 테스트"""
        config = TrainingConfig()
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.epochs, 5)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertTrue(config.evaluate_after_training)
        self.assertEqual(config.validation_split, 0.1)
        self.assertFalse(config.early_stopping)
        self.assertEqual(config.early_stopping_patience, 3)
        self.assertEqual(config.model_type, ModelType.JAX)

    def test_custom_values(self):
        """사용자 정의값 테스트"""
        config = TrainingConfig(
            batch_size=64,
            epochs=10,
            learning_rate=0.01,
            evaluate_after_training=False,
            validation_split=0.2,
            early_stopping=True,
            early_stopping_patience=5,
            model_type=ModelType.FLAX,
        )

        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.epochs, 10)
        self.assertEqual(config.learning_rate, 0.01)
        self.assertFalse(config.evaluate_after_training)
        self.assertEqual(config.validation_split, 0.2)
        self.assertTrue(config.early_stopping)
        self.assertEqual(config.early_stopping_patience, 5)
        self.assertEqual(config.model_type, ModelType.FLAX)

    def test_to_dict(self):
        """to_dict 메서드 테스트"""
        config = TrainingConfig(batch_size=64, epochs=10, model_type=ModelType.FLAX)

        config_dict = config.to_dict()

        self.assertEqual(config_dict["batch_size"], 64)
        self.assertEqual(config_dict["epochs"], 10)
        self.assertEqual(config_dict["model_type"], "flax_mnist")  # Enum 값 변환 확인

    def test_from_dict(self):
        """from_dict 메서드 테스트"""
        config_dict = {"batch_size": 128, "epochs": 20, "model_type": "jax"}

        config = TrainingConfig.from_dict(config_dict)

        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.epochs, 20)
        self.assertEqual(config.model_type, ModelType.JAX)

        # 대/소문자 무시 및 약어 테스트
        config_dict["model_type"] = "FlAx_MnIsT"
        config = TrainingConfig.from_dict(config_dict)
        self.assertEqual(config.model_type, ModelType.FLAX)


if __name__ == "__main__":
    unittest.main()
