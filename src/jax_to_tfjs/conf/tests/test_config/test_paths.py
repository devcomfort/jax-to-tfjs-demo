"""
경로 설정 모듈 테스트

PathConfig 클래스를 테스트합니다.
"""

import unittest
from jax_to_tfjs.conf.config.paths import PathConfig


class TestPathConfig(unittest.TestCase):
    """PathConfig 클래스 테스트"""

    def test_default_values(self):
        """기본값 테스트"""
        config = PathConfig()
        self.assertEqual(config.checkpoint_dir, "checkpoints")
        self.assertEqual(config.jax_checkpoint_dir, "jax_mnist")
        self.assertEqual(config.flax_checkpoint_dir, "flax_mnist")
        self.assertEqual(config.results_dir, "results")
        self.assertEqual(config.web_model_dir, "model")
        self.assertEqual(config.data_cache_dir, "data_cache")
        self.assertEqual(config.log_dir, "logs")

    def test_custom_values(self):
        """사용자 정의값 테스트"""
        config = PathConfig(
            checkpoint_dir="custom_checkpoints",
            jax_checkpoint_dir="jax_models",
            results_dir="output",
            web_model_dir="tfjs_models",
            log_dir="logging",
        )

        self.assertEqual(config.checkpoint_dir, "custom_checkpoints")
        self.assertEqual(config.jax_checkpoint_dir, "jax_models")
        self.assertEqual(config.flax_checkpoint_dir, "flax_mnist")  # 기본값 유지
        self.assertEqual(config.results_dir, "output")
        self.assertEqual(config.web_model_dir, "tfjs_models")
        self.assertEqual(config.data_cache_dir, "data_cache")  # 기본값 유지
        self.assertEqual(config.log_dir, "logging")

    def test_to_dict(self):
        """to_dict 메서드 테스트"""
        config = PathConfig(checkpoint_dir="models", log_dir="logs/training")

        config_dict = config.to_dict()

        self.assertEqual(config_dict["checkpoint_dir"], "models")
        self.assertEqual(config_dict["log_dir"], "logs/training")
        self.assertEqual(config_dict["jax_checkpoint_dir"], "jax_mnist")  # 기본값

    def test_from_dict(self):
        """from_dict 메서드 테스트"""
        config_dict = {
            "checkpoint_dir": "saved_models",
            "web_model_dir": "web_output",
            "log_dir": "logs/custom",
        }

        config = PathConfig.from_dict(config_dict)

        # 지정한 값 확인
        self.assertEqual(config.checkpoint_dir, "saved_models")
        self.assertEqual(config.web_model_dir, "web_output")
        self.assertEqual(config.log_dir, "logs/custom")

        # 지정하지 않은 값은 기본값 유지
        self.assertEqual(config.jax_checkpoint_dir, "jax_mnist")
        self.assertEqual(config.data_cache_dir, "data_cache")


if __name__ == "__main__":
    unittest.main()
