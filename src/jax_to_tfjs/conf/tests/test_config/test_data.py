"""
데이터 설정 모듈 테스트

DataConfig 클래스를 테스트합니다.
"""

import unittest
from jax_to_tfjs.conf.config.data import DataConfig


class TestDataConfig(unittest.TestCase):
    """DataConfig 클래스 테스트"""

    def test_default_values(self):
        """기본값 테스트"""
        config = DataConfig()
        self.assertTrue(config.cache_data)
        self.assertEqual(config.shuffle_buffer_size, 10000)
        self.assertEqual(config.prefetch_buffer_size, 4)
        self.assertEqual(config.num_parallel_calls, -1)  # AUTOTUNE 값
        self.assertEqual(config.normalize_method, "divide_by_255")

    def test_custom_values(self):
        """사용자 정의값 테스트"""
        config = DataConfig(
            cache_data=False,
            shuffle_buffer_size=5000,
            prefetch_buffer_size=8,
            num_parallel_calls=2,
            normalize_method="standard",
        )

        self.assertFalse(config.cache_data)
        self.assertEqual(config.shuffle_buffer_size, 5000)
        self.assertEqual(config.prefetch_buffer_size, 8)
        self.assertEqual(config.num_parallel_calls, 2)
        self.assertEqual(config.normalize_method, "standard")

    def test_to_dict(self):
        """to_dict 메서드 테스트"""
        config = DataConfig(
            cache_data=False, shuffle_buffer_size=5000, normalize_method="min_max"
        )

        config_dict = config.to_dict()

        self.assertFalse(config_dict["cache_data"])
        self.assertEqual(config_dict["shuffle_buffer_size"], 5000)
        self.assertEqual(config_dict["normalize_method"], "min_max")

        # 변경되지 않은 기본값도 포함되어 있는지 확인
        self.assertEqual(config_dict["prefetch_buffer_size"], 4)
        self.assertEqual(config_dict["num_parallel_calls"], -1)

    def test_from_dict(self):
        """from_dict 메서드 테스트"""
        config_dict = {
            "cache_data": False,
            "prefetch_buffer_size": 16,
            "normalize_method": "min_max",
        }

        config = DataConfig.from_dict(config_dict)

        # 지정한 값 확인
        self.assertFalse(config.cache_data)
        self.assertEqual(config.prefetch_buffer_size, 16)
        self.assertEqual(config.normalize_method, "min_max")

        # 지정하지 않은 값은 기본값 유지
        self.assertEqual(config.shuffle_buffer_size, 10000)
        self.assertEqual(config.num_parallel_calls, -1)

        # 부분 업데이트 테스트 (일부 필드만 제공)
        minimal_dict = {"normalize_method": "standard"}

        config = DataConfig.from_dict(minimal_dict)
        self.assertEqual(config.normalize_method, "standard")
        self.assertTrue(config.cache_data)  # 기본값 유지


if __name__ == "__main__":
    unittest.main()
