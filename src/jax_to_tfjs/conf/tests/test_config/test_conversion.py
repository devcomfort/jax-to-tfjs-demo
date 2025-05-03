"""
변환 설정 모듈 테스트

ConversionConfig 클래스를 테스트합니다.
"""

import unittest
from jax_to_tfjs.conf.config.conversion import ConversionConfig


class TestConversionConfig(unittest.TestCase):
    """ConversionConfig 클래스 테스트"""

    def test_default_values(self):
        """기본값 테스트"""
        config = ConversionConfig()
        self.assertFalse(config.quantize)
        self.assertEqual(config.quantization_type, "uint8")
        self.assertEqual(config.web_model_name, "model.json")
        self.assertFalse(config.include_optimizer)
        self.assertEqual(config.save_format, "tfjs")

    def test_custom_values(self):
        """사용자 정의값 테스트"""
        config = ConversionConfig(
            quantize=True,
            quantization_type="uint16",
            web_model_name="mnist_model.json",
            include_optimizer=True,
            save_format="keras_saved_model",
        )

        self.assertTrue(config.quantize)
        self.assertEqual(config.quantization_type, "uint16")
        self.assertEqual(config.web_model_name, "mnist_model.json")
        self.assertTrue(config.include_optimizer)
        self.assertEqual(config.save_format, "keras_saved_model")

    def test_to_dict(self):
        """to_dict 메서드 테스트"""
        config = ConversionConfig(quantize=True, web_model_name="quantized_model.json")

        config_dict = config.to_dict()

        self.assertTrue(config_dict["quantize"])
        self.assertEqual(config_dict["web_model_name"], "quantized_model.json")
        self.assertEqual(config_dict["quantization_type"], "uint8")  # 기본값
        self.assertFalse(config_dict["include_optimizer"])  # 기본값
        self.assertEqual(config_dict["save_format"], "tfjs")  # 기본값

    def test_from_dict(self):
        """from_dict 메서드 테스트"""
        config_dict = {
            "quantize": True,
            "quantization_type": "uint16",
            "web_model_name": "custom_model.json",
        }

        config = ConversionConfig.from_dict(config_dict)

        # 지정한 값 확인
        self.assertTrue(config.quantize)
        self.assertEqual(config.quantization_type, "uint16")
        self.assertEqual(config.web_model_name, "custom_model.json")

        # 지정하지 않은 값은 기본값 유지
        self.assertFalse(config.include_optimizer)
        self.assertEqual(config.save_format, "tfjs")

        # 부분 업데이트 테스트
        minimal_dict = {"save_format": "keras_saved_model"}

        config = ConversionConfig.from_dict(minimal_dict)
        self.assertEqual(config.save_format, "keras_saved_model")
        self.assertFalse(config.quantize)  # 기본값 유지


if __name__ == "__main__":
    unittest.main()
