"""
설정 로더 모듈 테스트

Config 클래스 및 통합 설정 기능을 테스트합니다.
"""

import unittest
import os
import tempfile
import json
from pathlib import Path

from jax_to_tfjs.conf.config.loader import Config
from jax_to_tfjs.conf.config.training import ModelType


class TestConfigLoader(unittest.TestCase):
    """Config 클래스 테스트"""

    def test_default_values(self):
        """기본값 테스트"""
        config = Config()

        # 각 섹션 객체가 생성되었는지 확인
        self.assertIsNotNone(config.training)
        self.assertIsNotNone(config.data)
        self.assertIsNotNone(config.paths)
        self.assertIsNotNone(config.conversion)
        self.assertIsNotNone(config.custom_settings)

        # 기본값 확인 (각 섹션에서 하나씩만 확인)
        self.assertEqual(config.training.batch_size, 32)
        self.assertTrue(config.data.cache_data)
        self.assertEqual(config.paths.checkpoint_dir, "checkpoints")
        self.assertFalse(config.conversion.quantize)
        self.assertEqual(config.custom_settings, {})

    def test_environment_variables(self):
        """환경 변수 로드 테스트"""
        # 환경 변수 설정
        os.environ["JAX_TFJS_BATCH_SIZE"] = "64"
        os.environ["JAX_TFJS_MODEL_TYPE"] = "flax"
        os.environ["JAX_TFJS_CACHE_DATA"] = "false"

        try:
            config = Config()

            # 환경 변수에서 로드된 값 확인
            self.assertEqual(config.training.batch_size, 64)
            self.assertEqual(config.training.model_type, ModelType.FLAX)
            self.assertFalse(config.data.cache_data)

            # 환경 변수에 설정되지 않은 값은 기본값 유지
            self.assertEqual(config.training.epochs, 5)
        finally:
            # 테스트 후 환경 변수 제거
            del os.environ["JAX_TFJS_BATCH_SIZE"]
            del os.environ["JAX_TFJS_MODEL_TYPE"]
            del os.environ["JAX_TFJS_CACHE_DATA"]

    def test_update_from_dict(self):
        """딕셔너리에서 설정 업데이트 테스트"""
        config = Config()

        # 업데이트할 설정 딕셔너리
        config_dict = {
            "training": {"batch_size": 128, "epochs": 20, "model_type": "flax"},
            "data": {"normalize_method": "standard"},
            "paths": {"checkpoint_dir": "saved_models"},
            "custom_settings": {"experiment_name": "test_run", "debug": True},
        }

        config.update_from_dict(config_dict)

        # 업데이트된 값 확인
        self.assertEqual(config.training.batch_size, 128)
        self.assertEqual(config.training.epochs, 20)
        self.assertEqual(config.training.model_type, ModelType.FLAX)
        self.assertEqual(config.data.normalize_method, "standard")
        self.assertEqual(config.paths.checkpoint_dir, "saved_models")
        self.assertEqual(config.custom_settings["experiment_name"], "test_run")
        self.assertTrue(config.custom_settings["debug"])

    def test_load_save_file(self):
        """파일 로드/저장 테스트"""
        config = Config()

        # 설정 수정
        config.training.batch_size = 64
        config.training.epochs = 15
        config.data.normalize_method = "min_max"
        config.paths.log_dir = "logs/test"
        config.conversion.quantize = True
        config.custom_settings["version"] = "1.0.0"

        # 임시 파일에 저장
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            config.save_to_file(temp_path)

            # 새 설정 객체 생성 및 파일에서 로드
            new_config = Config()
            new_config.load_from_file(temp_path)

            # 로드된 값 확인
            self.assertEqual(new_config.training.batch_size, 64)
            self.assertEqual(new_config.training.epochs, 15)
            self.assertEqual(new_config.data.normalize_method, "min_max")
            self.assertEqual(new_config.paths.log_dir, "logs/test")
            self.assertTrue(new_config.conversion.quantize)
            self.assertEqual(new_config.custom_settings["version"], "1.0.0")
        finally:
            # 임시 파일 삭제
            os.unlink(temp_path)

    def test_to_dict_from_dict(self):
        """to_dict 및 from_dict 메서드 테스트"""
        # 초기 설정 생성 및 수정
        original_config = Config()
        original_config.training.batch_size = 64
        original_config.data.cache_data = False
        original_config.paths.checkpoint_dir = "models"
        original_config.conversion.quantize = True
        original_config.custom_settings["author"] = "test_user"

        # 딕셔너리로 변환
        config_dict = original_config.to_dict()

        # 새 객체 생성
        new_config = Config.from_dict(config_dict)

        # 값 비교
        self.assertEqual(new_config.training.batch_size, 64)
        self.assertFalse(new_config.data.cache_data)
        self.assertEqual(new_config.paths.checkpoint_dir, "models")
        self.assertTrue(new_config.conversion.quantize)
        self.assertEqual(new_config.custom_settings["author"], "test_user")


if __name__ == "__main__":
    unittest.main()
