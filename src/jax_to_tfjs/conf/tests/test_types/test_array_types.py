"""
배열 및 텐서 타입 관련 모듈 테스트

ArrayLike 프로토콜 및 배열 변환 함수를 테스트합니다.
"""

import unittest
import numpy as np
import inspect
import pytest
import sys

from jax_to_tfjs.conf.types.array_types import ArrayLike, ensure_numpy


class SimpleArrayLike:
    """ArrayLike 프로토콜을 구현한 간단한 클래스"""

    def __init__(self, data):
        self._data = np.array(data)

    def __array__(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype


class TrueNonConvertibleObject:
    """변환 시 TypeError를 발생시키는 클래스"""

    def __init__(self):
        self.data = "문자열 데이터"

    def __repr__(self):
        return f"TrueNonConvertibleObject(data={self.data!r})"

    def __array__(self):
        # np.array 변환 시 TypeError 발생
        raise TypeError("이 객체는 배열로 변환할 수 없습니다")


class TestArrayTypes(unittest.TestCase):
    """배열 타입 모듈 테스트"""

    def test_array_like_protocol(self):
        """ArrayLike 프로토콜 테스트"""
        # 테스트 데이터 정의
        test_array_data = [1, 2, 3]
        test_custom_data = [4, 5, 6]

        # NumPy 배열은 ArrayLike 프로토콜을 만족함
        arr = np.array(test_array_data)
        self.assertIsInstance(arr, ArrayLike)

        # 커스텀 클래스도 프로토콜을 만족하면 통과
        custom_arr = SimpleArrayLike(test_custom_data)
        self.assertIsInstance(custom_arr, ArrayLike)

        # 프로토콜을 만족하지 않는 클래스
        class NotArrayLike:
            pass

        not_arr = NotArrayLike()
        self.assertNotIsInstance(not_arr, ArrayLike)

    @pytest.mark.debug_types
    def test_ensure_numpy(self, debug_logger=None):
        """ensure_numpy 함수 테스트"""
        # 테스트 데이터 정의
        test_array_data = [1, 2, 3]
        test_custom_data = [4, 5, 6]

        # 함수 소스코드 확인
        print("\n[디버깅] ensure_numpy 함수 소스코드:")
        try:
            source = inspect.getsource(ensure_numpy)
            print(source)
        except Exception as e:
            print(f"소스코드 추출 오류: {e}")

        # 이미 NumPy 배열인 경우
        arr = np.array(test_array_data)
        np_arr = ensure_numpy(arr)
        self.assertIs(np_arr, arr)  # 동일 객체 반환

        # 리스트에서 변환
        lst = test_array_data
        np_arr = ensure_numpy(lst)
        self.assertIsInstance(np_arr, np.ndarray)
        self.assertTrue(np.array_equal(np_arr, np.array(lst)))

        # 커스텀 ArrayLike 객체에서 변환
        custom_arr = SimpleArrayLike(test_custom_data)
        np_arr = ensure_numpy(custom_arr)
        self.assertIsInstance(np_arr, np.ndarray)
        self.assertTrue(np.array_equal(np_arr, np.array(test_custom_data)))

        # 변환 불가능한 객체
        print("\n[디버깅] 진짜 변환 불가능한 객체 테스트:")
        non_convertible = TrueNonConvertibleObject()
        print(f"테스트 객체: {non_convertible}")

        try:
            print("ensure_numpy 호출 전...")
            result = ensure_numpy(non_convertible)
            print(f"예상치 못한 성공! 결과: {result}, 타입: {type(result)}")
            self.fail("TypeError가 발생하지 않았습니다")
        except TypeError as e:
            print(f"성공적으로 TypeError 발생: {e}")
            # 정상적으로 TypeError가.발생함
            pass


if __name__ == "__main__":
    unittest.main()
