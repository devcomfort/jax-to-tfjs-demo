"""
타입 변환 유틸리티 모듈 테스트

경로 관련 함수 및 타입 변환 함수를 테스트합니다.
"""

import unittest
import os
from pathlib import Path

from jax_to_tfjs.conf.types.utils import ensure_path, ensure_str_path, safe_cast


class TestTypeUtils(unittest.TestCase):
    """타입 유틸리티 함수 테스트"""

    def test_ensure_path(self):
        """ensure_path 함수 테스트"""
        # 문자열에서 Path 객체로 변환
        path_str = "models/checkpoint"
        path_obj = ensure_path(path_str)
        self.assertIsInstance(path_obj, Path)
        self.assertEqual(str(path_obj), path_str)

        # 이미 Path 객체인 경우
        path_obj2 = Path("data/images")
        result = ensure_path(path_obj2)
        self.assertIs(result, path_obj2)  # 동일 객체 반환

    def test_ensure_str_path(self):
        """ensure_str_path 함수 테스트"""
        # Path 객체에서 문자열로 변환
        path_obj = Path("models/weights")
        path_str = ensure_str_path(path_obj)
        self.assertIsInstance(path_str, str)
        self.assertEqual(path_str, str(path_obj))

        # 이미 문자열인 경우
        path_str2 = "data/cache"
        result = ensure_str_path(path_str2)
        self.assertIs(result, path_str2)  # 동일 객체 반환

    def test_safe_cast(self):
        """safe_cast 함수 테스트"""
        # 성공적인 캐스팅
        self.assertEqual(safe_cast("42", int), 42)
        self.assertEqual(safe_cast(3.14, float), 3.14)
        self.assertEqual(safe_cast(42, str), "42")

        # 이미 대상 타입인 경우
        i = 100
        self.assertIs(safe_cast(i, int), i)  # 동일 객체 반환

        # 변환 불가능한 경우
        with self.assertRaises(TypeError):
            safe_cast("hello", int)

        # 복잡한 타입 변환
        # 리스트로 변환
        self.assertEqual(safe_cast((1, 2, 3), list), [1, 2, 3])

        # 세트로 변환
        self.assertEqual(safe_cast([1, 2, 2, 3], set), {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
